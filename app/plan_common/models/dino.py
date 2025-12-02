import os

import torch
import torch.nn as nn

torch.hub._validate_not_a_forked_repo = lambda a, b, c: True


class DinoEncoder(nn.Module):
    def __init__(self, name, feature_key, causal_enc=False):
        super().__init__()
        self.name = name
        if self.name.startswith("dinov2"):
            self.base_model = torch.hub.load("facebookresearch/dinov2", name)
        elif self.name.startswith("dinov3"):
            pretrained_ckpt_root = os.environ.get("PRETRAINED_CKPT_ROOT")
            dinov3_path = os.path.join(os.environ.get("JEPA_HOME", os.path.expanduser("~")), "dinov3")
            if "vitl16" in self.name:
                self.base_model = torch.hub.load(
                    dinov3_path,
                    name,
                    source="local",
                    backbone_weights=f"{pretrained_ckpt_root}/dinov3/{name}_pretrain_lvd1689m-7c1da9a5.pth",
                    weights=f"{pretrained_ckpt_root}/dinov3/{name}_pretrain_lvd1689m-7c1da9a5.pth",
                )
            else:
                self.base_model = torch.hub.load(
                    dinov3_path,
                    name,
                    source="local",
                    weights=f"{pretrained_ckpt_root}/dinov3/{name}_pretrain_lvd1689m.pth",
                )
        self.feature_key = feature_key
        self.emb_dim = self.base_model.num_features
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")

        self.patch_size = self.base_model.patch_size

    def forward(self, x):
        emb = self.base_model.forward_features(x)[self.feature_key]
        if self.latent_ndim == 1:
            emb = emb.unsqueeze(1)  # dummy patch dim
        return emb
