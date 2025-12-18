# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch.nn as nn


class ProprioceptiveEmbedding(nn.Module):

    def __init__(
        self,
        num_frames=16,
        tubelet_size=2,
        in_chans=7,
        tokens_per_step=1,
        embed_dim=384,
        use_3d_pos=False,
        use_embedding=True,
        shift_input=True,
        use_mlp=False,
        mlp_dims=[384],
    ):
        super().__init__()
        self.num_frames = num_frames
        self.use_embedding = use_embedding
        self.tubelet_size = tubelet_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.shift_input = shift_input
        self.tokens_per_step = tokens_per_step
        self.use_mlp = use_mlp
        self.patch_embed = nn.Conv1d(
            in_chans, embed_dim * tokens_per_step, kernel_size=tubelet_size, stride=tubelet_size
        )
        if self.use_mlp:
            self.mlp = mlp(in_dim=mlp_dims[0], mlp_dims=[], out_dim=mlp_dims[-1], act=nn.GELU(), dropout=0.0)

    def forward(self, x):
        """
        Input:
            x: [B T D]
        """
        B, T, D = x.shape
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        if self.use_mlp:
            x = self.mlp(x)
        x = x.reshape(B, T, self.tokens_per_step, -1)
        return x


def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0.0):
    """
    MLP with LayerNorm, Mish activations, and optionally dropout.
    """
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]
    mlp = nn.ModuleList()
    for i in range(len(dims) - 2):
        mlp.append(NormedLinear(dims[i], dims[i + 1], dropout=dropout * (i == 0)))
    mlp.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*mlp)


class NormedLinear(nn.Linear):
    """
    Linear layer with LayerNorm, activation, and optionally dropout.
    """

    def __init__(self, *args, dropout=0.0, act=nn.Mish(inplace=True), **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features)
        self.act = act
        self.dropout = nn.Dropout(dropout, inplace=True) if dropout else None

    def forward(self, x):
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)
        return self.act(self.ln(x))

    def __repr__(self):
        repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
        return (
            f"NormedLinear(in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}{repr_dropout}, "
            f"act={self.act.__class__.__name__})"
        )
