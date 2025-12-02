# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from functools import partial

import torch
import torch.nn as nn
from einops import repeat

from app.plan_common.models.modules import Block, build_action_block_causal_attention_mask
from src.utils.tensors import trunc_normal_


class VisionTransformerPredictorAC(nn.Module):
    """Action Conditioned Vision Transformer Predictor"""

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        embed_dim=768,
        predictor_embed_dim=1024,
        depth=24,
        num_heads=16,
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
        is_frame_causal=True,
        use_activation_checkpointing=False,
        use_rope=True,
        action_dim=7,
        action_emb_dim=None,
        proprio_emb_dim=None,
        proprio_dim=7,
        action_conditioning="token",
        use_extrinsics=False,
        proprio_tokens=1,
        proprio_encoder_inpred=False,
        action_encoder_inpred=True,
        **kwargs
    ):
        super().__init__()
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.is_frame_causal = is_frame_causal
        self.use_extrinsics = use_extrinsics
        self.proprio_tokens = proprio_tokens
        self.action_conditioning = action_conditioning
        self.proprio_emb_dim = proprio_emb_dim if proprio_emb_dim is not None else 0
        self.action_emb_dim = action_emb_dim
        self.predictor_embed_dim = predictor_embed_dim
        self.proprio_encoder_inpred = proprio_encoder_inpred
        self.action_encoder_inpred = action_encoder_inpred

        # Map visual input to predictor dimension
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        if self.action_conditioning == "token":
            if self.action_encoder_inpred:
                self.action_encoder = nn.Linear(action_dim, predictor_embed_dim, bias=True)
            if self.proprio_tokens > 0 and self.proprio_encoder_inpred:
                self.proprio_encoder = nn.Linear(proprio_dim, predictor_embed_dim, bias=True)
            if self.use_extrinsics:
                self.extrinsics_encoder = nn.Linear(action_dim - 1, predictor_embed_dim, bias=True)
        elif self.action_conditioning == "feature":
            assert action_emb_dim is not None, "action_emb_dim must be specified for feature conditioning"
            if self.action_encoder_inpred:
                self.action_encoder = nn.Linear(action_dim, action_emb_dim, bias=True)
            if self.proprio_emb_dim > 0 and self.proprio_encoder_inpred:
                self.proprio_encoder = nn.Linear(proprio_dim, proprio_emb_dim, bias=True)
            if self.use_extrinsics:
                self.extrinsics_encoder = nn.Linear(action_dim - 1, action_emb_dim, bias=True)

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
        self.use_activation_checkpointing = use_activation_checkpointing

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Attention Blocks
        self.use_rope = use_rope

        if self.action_conditioning == "feature":
            self.predictor_total_embed_dim = (
                predictor_embed_dim + action_emb_dim + proprio_emb_dim + (action_emb_dim if use_extrinsics else 0)
            )
        else:
            self.predictor_total_embed_dim = predictor_embed_dim

        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    use_rope=use_rope,
                    grid_size=self.grid_height,
                    dim=self.predictor_total_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
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
        # proj only for visual part of the prediction
        self.predictor_proj = nn.Linear(self.predictor_embed_dim, embed_dim, bias=True)

        # ------ initialize weights
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

        attn_mask = None
        if self.is_frame_causal:
            grid_depth = self.num_frames // self.tubelet_size
            grid_height = self.img_height // self.patch_size
            grid_width = self.img_width // self.patch_size
            if self.action_conditioning == "feature":
                self.cond_tokens = 0
            else:
                self.cond_tokens = 1 + self.proprio_tokens + (1 if use_extrinsics else 0)
            attn_mask = build_action_block_causal_attention_mask(
                grid_depth, grid_height, grid_width, add_tokens=self.cond_tokens
            )
        self.attn_mask = attn_mask

    def concat_obs_act(self, z_vis, proprio, act):
        """
        input :
            z_vis: B T H*W D
            act: [B T 1 A] or [B T H*W A]
            proprio: [B T 1 P] or [B T H*W P]
        output:    z (tensor): (b, num_frames, num_patches, emb_dim)
        """
        if self.action_encoder_inpred:
            act = repeat(act, "b t 1 a -> b t f a", f=z_vis.shape[2])
        if self.proprio_emb_dim > 0:
            if self.proprio_encoder_inpred:
                proprio = repeat(proprio, "b t 1 p -> b t f p", f=z_vis.shape[2])
            z = torch.cat([z_vis, proprio, act], dim=3)  # (b, num_frames, num_patches, dim + action_dim)
        else:
            z = torch.cat([z_vis, act], dim=3)  # (b, num_frames, num_patches, dim + action_dim)
        return z

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

    def forward(self, x, actions, states, extrinsics=None):
        """
        Input:
            x: (B, N_ctxt, embed_dim) tensor of context tokens
            states: (B, T, state_dim) tensor of states if not proprio_encoder_in pred else (B T 1 state_dim)
            actions: (B, T, action_dim) tensor of actions if not action_encoder_in pred else (B T 1 action_dim)
            extrinsics: (B, T, extrinsics_dim) tensor of extrinsics
        Output:
            x: (B, T*H*W, embed_dim) tensor of predicted tokens
            action_features: (B, T, H*W, action_emb_dim) if action_conditioning='feature' else (B, T, 1, action_emb_dim)
            proprio_features: (B, T, H*W, proprio_emb_dim) if proprio_encoding='feature' else (B, T, 1, proprio_emb_dim)
        """
        # only for visual embedding input, map tokens to predictor dimensions
        x = self.predictor_embed(x)
        B, N_ctxt, D = x.size()
        T = N_ctxt // (self.grid_height * self.grid_width)

        # Encode action if action_encoder_inpred is True, else actions are pre-encoded
        if self.action_encoder_inpred:
            a = self.action_encoder(actions).unsqueeze(2) # B T A -> B T 1 A
        else:
            a = actions

        x = x.view(B, T, self.grid_height * self.grid_width, D)  # [B, T, H*W, D]
        if self.action_conditioning == "token":
            if self.proprio_tokens > 0:
                if self.proprio_encoder_inpred:
                    s = self.proprio_encoder(states).unsqueeze(2) # B T P -> B T 1 P
                else:
                    s = states
            else:
                s = torch.zeros(B, T, 0, D, device=x.device)
            if self.use_extrinsics:
                e = self.extrinsics_encoder(extrinsics).unsqueeze(2)
                x = torch.cat([a, s, e, x], dim=2).flatten(1, 2)  # [B, T*(H*W+3), D]
            else:
                x = torch.cat([a, s, x], dim=2).flatten(1, 2)  # [B, T*(H*W+2), D]
        elif self.action_conditioning == "feature":
            if self.proprio_emb_dim > 0:
                if self.proprio_encoder_inpred:
                    s = self.proprio_encoder(states).unsqueeze(2)
                else:
                    s = states
            else:
                s = None
            x = self.concat_obs_act(x, s, a).flatten(1, 2)  # [B, T*(H*W), D+A+P]
        attn_mask = self.attn_mask[: x.size(1), : x.size(1)].to(x.device, non_blocking=True)

        # Fwd prop
        for i, blk in enumerate(self.predictor_blocks):
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    blk,
                    x,
                    mask=None,
                    attn_mask=attn_mask,
                    T=T,
                    H=self.grid_height,
                    W=self.grid_width,
                    action_tokens=self.cond_tokens,
                    use_reentrant=False,
                )
            else:
                x = blk(
                    x,
                    mask=None,
                    attn_mask=attn_mask,
                    T=T,
                    H=self.grid_height,
                    W=self.grid_width,
                    action_tokens=self.cond_tokens,
                )
        x = self.predictor_norm(x)

        # Split out cond and frame tokens
        if self.action_conditioning == "token":
            x = x.view(B, T, self.cond_tokens + self.grid_height * self.grid_width, D)  # [B, T, K+H*W, D]
            action_features, x = x[:, :, : self.cond_tokens, :], x[:, :, self.cond_tokens :, :]
            if self.proprio_tokens > 0:
                action_features, proprio_features = (
                    action_features[:, :, 0:1, :],
                    action_features[:, :, 1 : 1 + self.proprio_tokens, :],
                )
            else:
                action_features, proprio_features = action_features[:, :, 0:1, :], None
            x = x.flatten(1, 2)
        elif self.action_conditioning == "feature":
            x = x.view(B, T, self.grid_height * self.grid_width, self.predictor_total_embed_dim)
            x, action_features = x[:, :, :, : -self.action_emb_dim :], x[:, :, :, -self.action_emb_dim :]
            if self.proprio_emb_dim > 0:
                x, proprio_features = x[:, :, :, : -self.proprio_emb_dim], x[:, :, :, -self.proprio_emb_dim :]
            else:
                proprio_features = None
            x = x.flatten(1, 2)
        # only for visual embedding output, apply projector
        x = self.predictor_proj(x)  # [B, T*(H*W+K), D] or [B, T*(H*W), D]
        return x, action_features, proprio_features


def vit_ac_predictor(**kwargs):
    model = VisionTransformerPredictorAC(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model
