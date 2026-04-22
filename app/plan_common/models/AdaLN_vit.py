# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
from functools import partial
from typing import Callable

import torch
import torch.nn as nn

from src.models.utils.modules import (
    MLP,
    Attention,
    DropPath,
    RoPEAttention,
    build_action_block_causal_attention_mask,
)
from src.utils.logging import get_logger
from src.utils.tensors import trunc_normal_

logger = get_logger(__name__)

BLOCK_SIZE = 64


def bipartite_soft_matching(metric: torch.Tensor, r: int, metric_dim: int = 64) -> Callable:
    """Bipartite soft matching for Token Merging (ToMe).

    Args:
        metric: [B, N, C] tensor of token features used to compute similarity.
        r: Number of tokens to merge (remove) in this step.
        metric_dim: Number of channels to use for similarity (lower = faster). 0 = use all.

    Returns:
        (merge_fn, merge_positions_fn, unmerge_fn): callables that reduce
        [B, N, C] -> [B, N-r, C] and [B, N] -> [B, N-r] positions respectively.
    """
    B, N, C = metric.shape

    r = min(r, N // 2)

    if r <= 0:
        return lambda x: x, lambda p: p, lambda x: x

    with torch.no_grad():
        if metric_dim > 0 and metric_dim < C:
            metric = metric[:, :, :metric_dim]

        a = metric[:, ::2, :]
        b = metric[:, 1::2, :]

        a_norm = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        b_norm = b / b.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        scores = a_norm @ b_norm.transpose(-1, -2)

        node_max, node_idx = scores.max(dim=-1)

        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r:, :].sort(dim=-2)[0]
        src_idx = edge_idx[..., :r, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    num_A = a.shape[1]

    def merge(x: torch.Tensor) -> torch.Tensor:
        x_a = x[:, ::2, :]
        x_b = x[:, 1::2, :]
        Bx, _, C = x.shape
        n_a = x_a.shape[1]

        x_unm = x_a.gather(-2, unm_idx.expand(Bx, n_a - r, C))
        x_src = x_a.gather(-2, src_idx.expand(Bx, r, C))
        x_dst = x_b.scatter_add(-2, dst_idx.expand(Bx, r, C), x_src)
        return torch.cat([x_unm, x_dst], dim=-2)

    def merge_positions(pos: torch.Tensor) -> torch.Tensor:
        pos_a = pos[:, ::2]
        pos_b = pos[:, 1::2]
        Bx = pos.shape[0]
        pos_unm = pos_a.gather(-1, unm_idx.squeeze(-1).expand(Bx, num_A - r))
        return torch.cat([pos_unm, pos_b], dim=-1)

    num_B = N // 2

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        Bx, _, C = x.shape
        x_unm = x[:, : num_A - r, :]
        x_dst = x[:, num_A - r :, :]
        x_src = x_dst.gather(-2, dst_idx.expand(Bx, r, C))
        x_a = torch.empty(Bx, num_A, C, device=x.device, dtype=x.dtype)
        x_a.scatter_(-2, unm_idx.expand(Bx, num_A - r, C), x_unm)
        x_a.scatter_(-2, src_idx.expand(Bx, r, C), x_src)
        x_full = torch.empty(Bx, N, C, device=x.device, dtype=x.dtype)
        x_full[:, ::2, :] = x_a
        x_full[:, 1::2, :] = x_dst
        return x_full

    return merge, merge_positions, unmerge


def _build_causal_block_mask(T: int, N_per_ts: int, device: torch.device) -> torch.Tensor:
    N = T * N_per_ts
    mask = torch.zeros(N, N, dtype=torch.bool, device=device)
    for t in range(T):
        mask[t * N_per_ts : (t + 1) * N_per_ts, : (t + 1) * N_per_ts] = True
    return mask


def _build_asymmetric_causal_block_mask(token_counts: list, device: torch.device) -> torch.Tensor:
    N = sum(token_counts)
    mask = torch.zeros(N, N, dtype=torch.bool, device=device)
    cum = 0
    for nt in token_counts:
        mask[cum : cum + nt, : cum + nt] = True
        cum += nt
    return mask


class FWAdaLNBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        wide_silu=True,
        norm_layer=nn.LayerNorm,
        use_sdpa=True,
        is_causal=False,
        grid_size=16,
        use_rope=False,
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.grid_size = grid_size
        if use_rope:
            self.attn = RoPEAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                grid_size=grid_size,
                proj_drop=drop,
            )
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                use_sdpa=use_sdpa,
                is_causal=is_causal,
                proj_drop=drop,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if act_layer is nn.SiLU:
            self.mlp = modules.SwiGLUFFN(
                in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, wide_silu=wide_silu, drop=drop
            )
        else:
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

    def forward_attn(self, x, z, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None, cond_tokens=0, token_counts=None):
        mod_raw = self.adaLN_modulation(z)
        if token_counts is not None:
            repeats = torch.tensor(token_counts, device=mod_raw.device)
            expanded = torch.repeat_interleave(mod_raw, repeats, dim=1)
        else:
            N_per_ts = x.size(1) // T
            expanded = mod_raw.repeat_interleave(N_per_ts, dim=1)
        shift_msa, scale_msa, gate_msa, _, _, _ = expanded.chunk(6, dim=2)
        if isinstance(self.attn, RoPEAttention):
            y = self.attn(
                modulate(self.norm1(x), shift_msa, scale_msa),
                mask=mask,
                attn_mask=attn_mask,
                T=T,
                H=H_patches,
                W=W_patches,
                action_tokens=cond_tokens,
            )
        else:
            y = self.attn(
                modulate(self.norm1(x), shift_msa, scale_msa),
                mask=mask,
                attn_mask=attn_mask,
            )
        x = x + self.drop_path(y * gate_msa)
        return x, mod_raw

    def forward_mlp(self, x, mod_raw, T, token_counts=None):
        if token_counts is not None:
            repeats = torch.tensor(token_counts, device=mod_raw.device)
            expanded = torch.repeat_interleave(mod_raw, repeats, dim=1)
        else:
            N_per_ts = x.size(1) // T
            expanded = mod_raw.repeat_interleave(N_per_ts, dim=1)
        _, _, _, shift_mlp, scale_mlp, gate_mlp = expanded.chunk(6, dim=2)
        x = x + self.drop_path(gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)))
        return x

    def forward(self, x, z, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None, cond_tokens=0):
        """
        Input:
            x : B, N, C with N = T*H*W
            z : B, T, D
        Returns:
            B, N, D
        """
        x, mod_raw = self.forward_attn(x, z, mask, attn_mask, T, H_patches, W_patches, cond_tokens)
        x = self.forward_mlp(x, mod_raw, T)
        return x


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class VisionTransformerAdaLN(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        use_silu=False,
        wide_silu=True,
        is_causal=False,
        use_activation_checkpointing=False,
        local_window=(-1, -1, -1),
        use_rope=True,
        action_dim=20,
        proprio_dim=10,
        use_proprio=True,
        act_mlp=False,
        prop_mlp=False,
        init_scale_factor_adaln=10,
        # AdaLN-predictor specific
        proprio_encoding="feature",  # 'feature' or 'token'
        proprio_emb_dim=0,  # if proprio_encoding='feature', proprio_emb_dim>0 will be used to encode the proprio input
        proprio_encoder_inpred=True,
        proprio_tokens=0,  # if proprio_encoding='token', proprio_tokens>0 will be used to encode the proprio input
        action_encoder_inpred=True,
        tome_r=0,
        tome_mode="uniform",
        **kwargs,
    ):
        super().__init__()
        self.tome_r = tome_r
        self.tome_mode = tome_mode
        self.attn_depth, self.attn_height, self.attn_width = local_window
        self.predictor_embed_dim = predictor_embed_dim
        self.proprio_encoder_inpred = proprio_encoder_inpred
        self.action_encoder_inpred = action_encoder_inpred

        # Map input to predictor dimension
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)

        # Determine positional embedding
        if type(img_size) is int:
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        # --
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1

        self.grid_height = img_size[0] // self.patch_size
        self.grid_width = img_size[1] // self.patch_size
        self.grid_depth = num_frames // self.tubelet_size
        self.use_activation_checkpointing = use_activation_checkpointing

        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.proprio_emb_dim = proprio_emb_dim
        self.proprio_tokens = proprio_tokens
        self.use_proprio = use_proprio
        self.proprio_encoding = proprio_encoding
        self.act_mlp = act_mlp
        self.prop_mlp = prop_mlp

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Embed proprio and action
        if self.use_proprio and self.proprio_encoding == "feature":
            self.predictor_total_embed_dim = predictor_embed_dim + proprio_emb_dim
        else:
            self.predictor_total_embed_dim = predictor_embed_dim

        # Initialize encoders
        if self.action_encoder_inpred:
            self.action_encoder = nn.Linear(action_dim, self.predictor_total_embed_dim, bias=True)
        if self.proprio_encoder_inpred:
            if self.proprio_encoding == "token" and self.proprio_tokens > 0:
                self.proprio_encoder = nn.Linear(proprio_dim, predictor_embed_dim, bias=True)
            elif self.proprio_encoding == "feature" and self.proprio_emb_dim > 0:
                self.proprio_encoder = nn.Linear(proprio_dim, proprio_emb_dim, bias=True)

        # Attention Blocks
        self.use_rope = use_rope
        self.predictor_blocks = nn.ModuleList(
            [
                FWAdaLNBlock(
                    use_rope=use_rope,
                    grid_size=self.grid_height,
                    grid_depth=self.grid_depth,
                    dim=self.predictor_total_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    is_causal=is_causal,
                    wide_silu=wide_silu,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # Normalize & project back to input dimension
        self.predictor_norm = norm_layer(self.predictor_total_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        attn_mask = None
        self.cond_tokens = 0
        if self.attn_depth > 0 or self.attn_height > 0 or self.attn_width > 0:
            grid_depth = self.num_frames // self.tubelet_size
            grid_height = self.img_height // self.patch_size
            grid_width = self.img_width // self.patch_size
            if self.proprio_tokens > 0 and self.proprio_encoding == "token":
                self.cond_tokens += 1
            attn_mask = build_action_block_causal_attention_mask(
                grid_depth,
                grid_height,
                grid_width,
                add_tokens=self.cond_tokens,
            )
        self.attn_mask = attn_mask

        # ------ initialize weights
        self.init_std = init_std
        self.init_scale_factor_adaln = init_scale_factor_adaln
        self.apply(self._init_weights)
        self._rescale_blocks()

        # Initialize for better gradient flow
        with torch.no_grad():
            for block in self.predictor_blocks:
                linear_layer = block.adaLN_modulation[1]
                if self.init_scale_factor_adaln == 0:
                    nn.init.constant_(linear_layer.weight, 0)
                else:
                    trunc_normal_(linear_layer.weight, std=self.init_std * self.init_scale_factor_adaln)
                nn.init.constant_(linear_layer.bias, 0)
            # Log once after initializing all blocks
            if self.init_scale_factor_adaln == 0:
                logger.info(f"🔧 Initialized {len(self.predictor_blocks)} AdaLN-zero blocks")
            else:
                logger.info(
                    f"🔧 Initialized {len(self.predictor_blocks)} AdaLN blocks (scale_factor={self.init_scale_factor_adaln})"
                )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def concat_obs(self, z_vis, proprio):
        """
        input :
            z_vis: B T H*W D
            proprio: B T 1 P
        output:    z (tensor): (b, num_frames, num_patches, emb_dim)
        """
        z = torch.cat([z_vis, proprio], dim=3)
        return z

    def forward(self, x, actions, proprio=None):
        """
        Input:
            x: B T V H W D
            actions: B T A
            proprio: B T P
        Returns:
            x: B T H*W D
            proprio: B T 1 P (P=D if proprio_encoding='token' else P=proprio_dim)
        """
        # Map context tokens to pedictor dimensions
        x = self.predictor_embed(x)
        x = x.flatten(2, 4)  # [B, T, H*W, D]
        B, T, N, D = x.shape

        # Encode actions if needed
        if self.action_encoder_inpred:
            z = self.action_encoder(actions)
        else:
            z = actions.squeeze(2)  # (b t 1 a) -> (b t a)

        if self.use_proprio and proprio is not None:
            if self.proprio_encoder_inpred:
                proprio = self.proprio_encoder(proprio).unsqueeze(2)
            if self.proprio_encoding == "token":
                x = torch.cat([proprio, x], dim=2).flatten(1, 2)  # [B, T*(H*W+1), D]
            elif self.proprio_encoding == "feature":
                x = self.concat_obs(x, proprio).flatten(1, 2)  # [B, T*(H*W), D+P]
        else:
            x = x.flatten(1, 2)  # [B, T*(H*W), D]
        attn_mask = (
            self.attn_mask[: x.size(1), : x.size(1)].to(x.device, non_blocking=True)
            if self.attn_mask is not None
            else None
        )

        # Initialize ToMe tracking
        N_per_ts_init = x.size(1) // T
        unmerge_fns = []
        tome_history = self.tome_r > 0 and self.tome_mode == "history" and T >= 2
        tome_uniform = self.tome_r > 0 and self.tome_mode == "uniform"

        if tome_history:
            T_hist = T - 1
            per_ts_counts = [N_per_ts_init] * T
            s_hist = torch.ones(B * T_hist, N_per_ts_init, 1, device=x.device, dtype=x.dtype)
            if self.use_rope:
                N_frame_orig = N_per_ts_init - self.cond_tokens
                pos_ids_hist = torch.arange(N_per_ts_init, device=x.device).unsqueeze(0).expand(B * T_hist, -1).contiguous()
                pos_ids_curr = torch.arange(N_per_ts_init, device=x.device).unsqueeze(0).expand(B, -1).contiguous()
        elif tome_uniform:
            s = torch.ones(B * T, N_per_ts_init, 1, device=x.device, dtype=x.dtype)
            if self.use_rope:
                pos_ids = torch.arange(N_per_ts_init, device=x.device).unsqueeze(0).expand(B * T, -1).contiguous()
                N_frame_orig = N_per_ts_init - self.cond_tokens
                t_offsets = (torch.arange(B * T, device=x.device) % T * N_frame_orig).unsqueeze(1)
            else:
                pos_ids = None
        else:
            s = None
            pos_ids = None

        # Fwd prop
        num_blocks = len(self.predictor_blocks)
        for i, blk in enumerate(self.predictor_blocks):
            # --- Build RoPE position mask ---
            if tome_history and self.use_rope:
                N_frame_orig = N_per_ts_init - self.cond_tokens
                parts = []
                n_hist_t = per_ts_counts[0]
                for t in range(T_hist):
                    pid = pos_ids_hist[t * B : (t + 1) * B, self.cond_tokens:]
                    parts.append(pid - self.cond_tokens + t * N_frame_orig)
                curr_frame = pos_ids_curr[:, self.cond_tokens:]
                parts.append(curr_frame - self.cond_tokens + T_hist * N_frame_orig)
                rope_mask = torch.cat(parts, dim=1)
            elif tome_uniform and self.use_rope and pos_ids is not None:
                frame_local = pos_ids[:, self.cond_tokens:]
                global_frame_pos = frame_local - self.cond_tokens + t_offsets
                rope_mask = global_frame_pos.view(B, -1)
            else:
                rope_mask = None

            tc = list(per_ts_counts) if tome_history else None

            # --- ATTENTION HALF ---
            if self.use_activation_checkpointing:
                x, mod_raw = torch.utils.checkpoint.checkpoint(
                    blk.forward_attn,
                    x,
                    z,
                    rope_mask,
                    attn_mask,
                    T,
                    self.grid_height,
                    self.grid_width,
                    self.cond_tokens,
                    tc,
                    use_reentrant=False,
                )
            else:
                x, mod_raw = blk.forward_attn(
                    x,
                    z,
                    mask=rope_mask,
                    attn_mask=attn_mask,
                    T=T,
                    H_patches=self.grid_height,
                    W_patches=self.grid_width,
                    cond_tokens=self.cond_tokens,
                    token_counts=tc,
                )

            # --- TOKEN MERGING (between attention and MLP) ---
            if (tome_history or tome_uniform) and i < num_blocks - 1:
                if tome_history:
                    N_hist_total = sum(per_ts_counts[:T_hist])
                    x_hist_all = x[:, :N_hist_total, :]
                    x_curr = x[:, N_hist_total:, :]
                    n_hist_t = per_ts_counts[0]
                    x_hist_bt = x_hist_all.reshape(B, T_hist, n_hist_t, -1).reshape(B * T_hist, n_hist_t, -1)
                    merge_fn, merge_fn_pos, unmerge_fn = bipartite_soft_matching(x_hist_bt, self.tome_r)
                    unmerge_fns.append(unmerge_fn)
                    x_hist_bt = merge_fn(x_hist_bt * s_hist)
                    s_hist = merge_fn(s_hist)
                    x_hist_bt = x_hist_bt / s_hist
                    n_hist_new = x_hist_bt.size(1)
                    x_hist_all = x_hist_bt.reshape(B, T_hist * n_hist_new, -1)
                    x = torch.cat([x_hist_all, x_curr], dim=1)
                    for t in range(T_hist):
                        per_ts_counts[t] = n_hist_new
                    if self.use_rope:
                        pos_ids_hist = merge_fn_pos(pos_ids_hist)
                    if self.attn_mask is not None:
                        attn_mask = _build_asymmetric_causal_block_mask(per_ts_counts, device=x.device)
                    tc = list(per_ts_counts)
                else:
                    N_cur = x.size(1) // T
                    x_t = x.view(B * T, N_cur, -1)
                    merge_fn, merge_fn_pos, unmerge_fn = bipartite_soft_matching(x_t, self.tome_r)
                    unmerge_fns.append(unmerge_fn)
                    x_t = merge_fn(x_t * s)
                    s = merge_fn(s)
                    x_t = x_t / s
                    N_new = x_t.size(1)
                    x = x_t.view(B, T * N_new, -1)
                    if self.use_rope:
                        pos_ids = merge_fn_pos(pos_ids)
                    if self.attn_mask is not None:
                        attn_mask = _build_causal_block_mask(T, N_new, device=x.device)

            # --- MLP HALF ---
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    blk.forward_mlp,
                    x,
                    mod_raw,
                    T,
                    tc,
                    use_reentrant=False,
                )
            else:
                x = blk.forward_mlp(x, mod_raw, T, token_counts=tc)

        # Unmerge to restore original token counts
        if tome_history and unmerge_fns:
            N_hist_total = sum(per_ts_counts[:T_hist])
            x_hist_all = x[:, :N_hist_total, :]
            x_curr = x[:, N_hist_total:, :]
            n_hist_t = per_ts_counts[0]
            x_hist_bt = x_hist_all.reshape(B, T_hist, n_hist_t, -1).reshape(B * T_hist, n_hist_t, -1)
            for ufn in reversed(unmerge_fns):
                x_hist_bt = ufn(x_hist_bt)
            x_hist_all = x_hist_bt.reshape(B, T_hist * N_per_ts_init, -1)
            x = torch.cat([x_hist_all, x_curr], dim=1)
        elif tome_uniform and unmerge_fns:
            N_cur = x.size(1) // T
            x_t = x.view(B * T, N_cur, -1)
            for ufn in reversed(unmerge_fns):
                x_t = ufn(x_t)
            x = x_t.view(B, T * N_per_ts_init, -1)

        x = self.predictor_norm(x)

        N_final = x.size(1) // T
        if self.use_proprio and proprio is not None:
            if self.proprio_encoding == "token":
                x = x.view(B, T, N_final, D)
                x, proprio_features = x[:, :, self.cond_tokens :, :], x[:, :, : self.cond_tokens, :]
            elif self.proprio_encoding == "feature":
                x = x.view(B, T, N_final, self.predictor_total_embed_dim)
                x, proprio_features = x[:, :, :, : -self.proprio_emb_dim], x[:, :, :, -self.proprio_emb_dim :]
        else:
            x = x.view(B, T, N_final, self.predictor_total_embed_dim)
            proprio_features = None

        x = self.predictor_proj(x)
        return x, None, proprio_features


def vit_predictor_AdaLN(**kwargs):
    model = VisionTransformerAdaLN(qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
