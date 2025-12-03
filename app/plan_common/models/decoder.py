# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
from functools import partial

import torch
import torch.nn as nn

from src.models.utils.modules import Block
from src.models.utils.pos_embs import get_2d_sincos_pos_embed
from src.utils.tensors import trunc_normal_


class VisionTransformerDecoder(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        img_size=(256, 256),
        patch_size=16,
        embed_dim=384,
        in_chans=3,
        decoder_embed_dim=1024,
        regression_dim=None,
        depth=3,
        num_heads=32,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        num_views=1,
        use_silu=False,
        wide_silu=True,
        is_causal=False,
        use_activation_checkpointing=False,
        use_rope=False,
        attn_impl="fmha",
        **kwargs
    ):
        super().__init__()
        self.attn_impl = attn_impl
        # Map input to decoder dimension
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # Determine positional embedding
        # --
        self.grid_size = img_size[0] // patch_size
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.grid_height = self.grid_size
        self.grid_width = self.grid_size
        self.use_activation_checkpointing = use_activation_checkpointing

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.num_patches = num_patches = (self.grid_size) * (self.grid_size)

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)

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
                    is_causal=is_causal,
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
        self.decoder_norm = norm_layer(decoder_embed_dim)
        if regression_dim is None:
            self.decoder_proj = nn.Linear(decoder_embed_dim, patch_size * patch_size * in_chans, bias=True)
        else:
            self.decoder_proj = nn.Linear(decoder_embed_dim, regression_dim, bias=True)
        self.num_views = num_views
        if num_views != 0:
            self.view_tokens = nn.Parameter(torch.zeros(1, num_views, decoder_embed_dim))

        # ------ initialize weights
        if self.decoder_pos_embed is not None:
            self._init_pos_embed(self.decoder_pos_embed.data)  # sincos pos-embed
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

    def interpolate_pos_encoding(self, x, pos_embed):
        _, N, dim = pos_embed.shape
        # If pos_embed already corret size, just return
        _, _, H, W, _ = x.shape
        if H == self.grid_size and W == self.grid_size:
            return pos_embed

        # Compute scale factor for spatial interpolation
        npatch = (H) * (W)
        scale_factor = math.sqrt(npatch / N)

        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=scale_factor,
            mode="bicubic",
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return pos_embed

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

    def forward(self, video_features, pos_enc_resize_strategy="interpolate"):
        """
        :param x: context tokens
        """
        x = video_features
        B, T, V, H, W, C = x.shape
        if self.num_patches != (H * W):
            x = (
                nn.functional.interpolate(
                    x.reshape(B * T * V, H, W, C).permute(0, 3, 1, 2),
                    scale_factor=(self.grid_size / H, self.grid_size / W),
                    mode="bilinear",
                )
                .permute(0, 2, 3, 1)
                .reshape(B, T, V, self.grid_size, self.grid_size, C)
            )
        x = self.decoder_embed(x)

        if not self.use_rope:
            B, T, V, H, W, C = x.shape
            if pos_enc_resize_strategy == "interpolate":
                pos_embed = self.interpolate_pos_encoding(x[:, :, 0], self.decoder_pos_embed)
            elif pos_enc_resize_strategy == "slice":
                pos_embed = self.decoder_pos_embed.view(1, self.num_frames, H, W, C)
                pos_embed = pos_embed[:, :T].reshape(1, T * H * W, C)
            pos_embed = pos_embed.repeat(B, 1, 1)
            if T > 1:
                pos_embed = pos_embed.unsqueeze(1).repeat(1, T, 1, 1)
            pos_embed = pos_embed.view(B, T, 1, H, W, C)

            view_tokens = self.view_tokens.view(1, 1, V, 1, 1, C)
            view_tokens = view_tokens.repeat(B, T, 1, H, W, 1)
            pos_embed = pos_embed.repeat(1, 1, V, 1, 1, 1)
            pos_embed = pos_embed + view_tokens
            x = x + pos_embed

        N = V * H * W
        x = x.view((B * T, N, C))

        # Fwd prop
        for i, blk in enumerate(self.decoder_blocks):
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_proj(x)

        x = x.view(B, T, V, self.grid_size, self.grid_size, -1)

        return x


def vit_decoder(**kwargs):
    model = VisionTransformerDecoder(mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
