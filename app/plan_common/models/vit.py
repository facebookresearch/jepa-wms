# Taken from https://github.com/gaoyuezhou/dino_wm,
# itself adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
#
# Modified to support SDPA (Scaled Dot Product Attention) and Flash Attention
# for faster training while maintaining numerical equivalence with the original.

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

# helpers
NUM_FRAMES = 1
NUM_PATCHES = 1

# SDPA backends to enable (includes Flash Attention and memory-efficient attention)
ALL_SDPA_BACKENDS = [
    SDPBackend.MATH,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.CUDNN_ATTENTION,
]


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def generate_mask_matrix(npatch, nwindow):
    """Generate causal attention mask.

    Returns a mask where 1 = attend, 0 = mask out.
    Shape: (1, 1, nwindow * npatch, nwindow * npatch)
    """
    zeros = torch.zeros(npatch, npatch)
    ones = torch.ones(npatch, npatch)
    rows = []
    for i in range(nwindow):
        row = torch.cat([ones] * (i + 1) + [zeros] * (nwindow - i - 1), dim=1)
        rows.append(row)
    mask = torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)
    return mask


def generate_sdpa_mask(npatch, nwindow, device=None):
    """Generate causal attention mask for SDPA.

    For SDPA with boolean attn_mask:
    - True = attend
    - False = mask out

    This is dtype-agnostic and works with both float32 and bfloat16 queries.

    Shape: (nwindow * npatch, nwindow * npatch)
    """
    # Build block-causal boolean mask
    total_len = npatch * nwindow
    mask = torch.zeros(total_len, total_len, device=device).bool()
    mask_block = torch.ones(npatch, npatch, device=device).bool()

    for i in range(nwindow):
        for j in range(i + 1):
            # Block (i, j) should attend (i can attend to j where j <= i)
            row_start = i * npatch
            row_end = (i + 1) * npatch
            col_start = j * npatch
            col_end = (j + 1) * npatch
            mask[row_start:row_end, col_start:col_end] = mask_block

    return mask


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, use_sdpa=True):
        """
        Args:
            dim: Input dimension
            heads: Number of attention heads
            dim_head: Dimension per head
            dropout: Dropout probability
            use_sdpa: If True, use PyTorch's scaled_dot_product_attention for
                     faster computation (Flash Attention / memory-efficient attention).
                     If False, use the original manual attention implementation.
        """
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.use_sdpa = use_sdpa
        self.dropout_p = dropout

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

        # Register both mask formats for flexibility
        # Original format (1 = attend, 0 = mask)
        self.register_buffer("bias", generate_mask_matrix(NUM_PATCHES, NUM_FRAMES))
        # SDPA format (0.0 = attend, -inf = mask) - will be created lazily
        self._sdpa_mask = None
        self._sdpa_mask_size = 0

    def _get_sdpa_mask(self, T, device):
        """Get or create SDPA-compatible boolean attention mask.

        Boolean masks are dtype-agnostic and work with both float32 and bfloat16 queries.
        The mask is created once and cached, then sliced and moved to the correct device.
        """
        if self._sdpa_mask is None or self._sdpa_mask_size < T:
            # Create boolean mask (True = attend, False = mask out)
            self._sdpa_mask = generate_sdpa_mask(NUM_PATCHES, NUM_FRAMES, device=device)
            self._sdpa_mask_size = NUM_FRAMES * NUM_PATCHES

        mask = self._sdpa_mask[:T, :T]
        # Ensure mask is on the correct device
        if mask.device != device:
            mask = mask.to(device=device, non_blocking=True)
        return mask

    def forward(self, x):
        B, T, C = x.size()
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        if self.use_sdpa:
            # Use PyTorch's optimized scaled_dot_product_attention
            # Get the attention mask in SDPA format (boolean mask is dtype-agnostic)
            attn_mask = self._get_sdpa_mask(T, x.device)

            with sdpa_kernel(ALL_SDPA_BACKENDS):
                # Note: SDPA applies dropout internally during training
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    scale=self.scale,
                )
        else:
            # Original manual attention implementation
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            # apply causal mask
            dots = dots.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

            attn = self.attend(dots)
            attn = self.dropout(attn)

            out = torch.matmul(attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0, use_sdpa=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, use_sdpa=use_sdpa),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class ViTPredictor(nn.Module):
    def __init__(
        self,
        *,
        num_patches,
        num_frames,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        use_sdpa=True,
    ):
        """
        ViT-based predictor for world models.

        Args:
            num_patches: Number of patches per frame
            num_frames: Number of frames in the sequence
            dim: Model dimension
            depth: Number of transformer layers
            heads: Number of attention heads
            mlp_dim: MLP hidden dimension
            pool: Pooling type ('cls' or 'mean')
            dim_head: Dimension per attention head
            dropout: Dropout probability
            emb_dropout: Embedding dropout probability
            use_sdpa: If True, use SDPA/Flash Attention for faster training.
                     If False, use original manual attention (for exact reproducibility
                     with DINO-WM baseline).
        """
        super().__init__()
        assert pool in {"cls", "mean"}, "pool type must be either cls (cls token) or mean (mean pooling)"

        # update params for adding causal attention masks
        global NUM_FRAMES, NUM_PATCHES
        NUM_FRAMES = num_frames
        NUM_PATCHES = num_patches

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames * (num_patches), dim)
        )  # dim for the pos encodings # don't add class for now!
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, use_sdpa=use_sdpa)
        self.pool = pool

    def forward(self, x, **kwargs):  # x: (b, window_size * H/patch_size * W/patch_size, 384)
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        x = self.transformer(x)
        return x
