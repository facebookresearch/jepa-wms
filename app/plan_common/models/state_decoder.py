# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math

import torch
import torch.nn as nn

# from app.plan_common.models.utils.modules_all import Block
from app.plan_common.models.modules import Block
from src.models.utils.pos_embs import get_2d_sincos_pos_embed
from src.utils.tensors import trunc_normal_


class StateReadoutViT(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        grid_size=16,
        embed_dim=1408,
        decoder_embed_dim=384,
        depth=3,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        state_dim=7,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        use_silu=False,
        wide_silu=True,
        use_activation_checkpointing=False,
        use_rope=False,
        use_camera_embed=False,
        attn_impl="fmha",
        **kwargs
    ):
        super().__init__()
        self.attn_impl = attn_impl
        # Map input to decoder dimension
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        # Mask tokens
        self.mask_tokens = None
        self.num_mask_tokens = 0
        self.use_camera_embed = use_camera_embed
        self.state_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        if use_camera_embed:
            self.camera_embed = nn.Linear(6, decoder_embed_dim, bias=True)
        # Determine positional embedding
        # --
        self.grid_size = grid_size
        self.grid_height = grid_size
        self.grid_width = grid_size
        self.use_activation_checkpointing = use_activation_checkpointing
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, grid_size * grid_size, decoder_embed_dim), requires_grad=False
        )
        # Attention Blocks
        self.use_rope = use_rope
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    use_rope=use_rope,
                    grid_size=self.grid_height,
                    grid_depth=1,
                    dim=decoder_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    is_causal=False,
                    # use_sdpa=False,
                    attn_impl=attn_impl,
                    wide_silu=wide_silu,
                    block_size=None,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        # Normalize & project back to input dimension
        # self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_embed_dim = decoder_embed_dim
        self.state_proj = nn.Linear(decoder_embed_dim, state_dim)
        # ------ initialize weights
        if self.decoder_pos_embed is not None:
            self._init_pos_embed(self.decoder_pos_embed.data)  # sincos pos-embed
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_pos_embed(self, pos_embed):
        embed_dim = pos_embed.size(-1)
        grid_size = self.grid_size  # TODO: update; currently assumes square input
        sincos = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False)
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

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

        for layer_id, layer in enumerate(self.decoder_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def forward(self, video_features, camera_extrinsics):
        D = video_features.shape[-1]
        x = video_features.detach()
        B, T, V, H, W, C = x.shape

        # TODO: implement view token
        x = x[:, :, 0].reshape(B * T, H * W, C)
        x = self.decoder_embed(x)
        x = x + self.decoder_pos_embed.repeat(B * T, 1, 1)
        if self.use_camera_embed:
            camera_extrinsics_embedding = self.camera_embed(camera_extrinsics)
            camera_extrinsics_embedding = camera_extrinsics_embedding.reshape(B * T, 1, -1)
            x = torch.concatenate([camera_extrinsics_embedding, x], dim=1)
        x = torch.concatenate([self.state_token.repeat(B * T, 1, 1), x], dim=1)
        # Fwd prop
        for i, blk in enumerate(self.decoder_blocks):
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        # x = self.decoder_norm(x)
        x = x.reshape(B, T, -1, self.decoder_embed_dim)[:, :, 0]
        return self.state_proj(x)
