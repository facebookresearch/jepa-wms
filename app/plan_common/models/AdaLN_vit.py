import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from app.plan_common.models.modules import (
    MLP,
    Attention,
    DropPath,
    RoPEAttention,
    build_action_block_causal_attention_mask,
)
from src.utils.tensors import trunc_normal_
from src.utils.logging import get_logger

logger = get_logger(__name__)

BLOCK_SIZE = 64


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

    def forward(self, x, z, mask=None, attn_mask=None, T=None, H_patches=None, W_patches=None, cond_tokens=0):
        """
        Input:
            x : B, N, C with N = T*H*W
            z : B, T, D
        Returns:
            B, N, D
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(z).repeat_interleave((self.grid_size**2 + cond_tokens), dim=1).chunk(6, dim=2)
        )
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
        x = x + self.drop_path(gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)))
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
        **kwargs,
    ):
        super().__init__()
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
                chunk_size = predictor_embed_dim
                if self.init_scale_factor_adaln == 0:
                    nn.init.constant_(linear_layer.weight, 0)
                    logger.info(f"Initialized AdaLN-zero with {self.init_scale_factor_adaln=}")
                else:
                    trunc_normal_(linear_layer.weight, std=self.init_std * self.init_scale_factor_adaln)
                    logger.info(f"Initialized AdaLN with {self.init_scale_factor_adaln=}")
                nn.init.constant_(linear_layer.bias, 0)

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
            # TODO: if proprio, encode it either by sequence or feature conditioning on the visual x,
            # then separate visual and proprio output after AdaLN blocks
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

        # Fwd prop
        for i, blk in enumerate(self.predictor_blocks):
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    blk,
                    x,
                    z,
                    None,
                    attn_mask,
                    T=T,
                    H_patches=self.grid_height,
                    W_patches=self.grid_width,
                    use_reentrant=False,
                    cond_tokens=self.cond_tokens,
                )
            else:
                x = blk(
                    x,
                    z,
                    mask=None,
                    attn_mask=attn_mask,
                    T=T,
                    H_patches=self.grid_height,
                    W_patches=self.grid_width,
                    cond_tokens=self.cond_tokens,
                )
        x = self.predictor_norm(x)

        if self.use_proprio and proprio is not None:
            if self.proprio_encoding == "token":
                x = x.view(B, T, self.cond_tokens + self.grid_height * self.grid_width, D)  # [B, T, K+H*W, D]
                x, proprio_features = x[:, :, self.cond_tokens :, :], x[:, :, : self.cond_tokens, :]
            elif self.proprio_encoding == "feature":
                x = x.view(B, T, self.grid_height * self.grid_width, self.predictor_total_embed_dim)
                x, proprio_features = x[:, :, :, : -self.proprio_emb_dim], x[:, :, :, -self.proprio_emb_dim :]
        else:
            x = x.view(B, T, self.grid_height * self.grid_width, self.predictor_total_embed_dim)
            proprio_features = None

        x = self.predictor_proj(x)
        # TODO: if proprio, encode it either by sequence or feature conditioning on the visual x,
        # then separate visual and proprio output after AdaLN blocks
        return x, None, proprio_features


def vit_predictor_AdaLN(**kwargs):
    model = VisionTransformerAdaLN(qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
