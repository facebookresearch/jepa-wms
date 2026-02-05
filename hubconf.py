# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
PyTorch Hub configuration for JEPA-WMs pretrained models.

Example usage:
    import torch

    # Load a pretrained JEPA-WM model for Metaworld
    model, preprocessor = torch.hub.load('facebookresearch/jepa-wms', 'jepa_wm_metaworld')

    # Load a pretrained JEPA-WM model for DROID
    model, preprocessor = torch.hub.load('facebookresearch/jepa-wms', 'jepa_wm_droid')
"""

import os

dependencies = ["torch", "torchvision", "yaml", "omegaconf"]

# Hugging Face repository for model checkpoints
HF_REPO_ID = "facebook/jepa-wms"

# Model weight URLs - Hugging Face Hub is the primary source, fbaipublicfiles is fallback
# Filenames on Hugging Face Hub follow the pattern: {model_name}.pth.tar
MODEL_URLS = {
    # JEPA-WM models
    "jepa_wm_droid": "https://dl.fbaipublicfiles.com/jepa-wms/droid_jepa-wm_noprop.pth.tar",
    "jepa_wm_metaworld": "https://dl.fbaipublicfiles.com/jepa-wms/mw_jepa-wm.pth.tar",
    "jepa_wm_pointmaze": "https://dl.fbaipublicfiles.com/jepa-wms/mz_jepa-wm.pth.tar",
    "jepa_wm_pusht": "https://dl.fbaipublicfiles.com/jepa-wms/pt_jepa-wm.pth.tar",
    "jepa_wm_wall": "https://dl.fbaipublicfiles.com/jepa-wms/wall_jepa-wm.pth.tar",
    # DINO-WM baseline models
    "dino_wm_droid": "https://dl.fbaipublicfiles.com/jepa-wms/droid_dino-wm_noprop.pth.tar",
    "dino_wm_metaworld": "https://dl.fbaipublicfiles.com/jepa-wms/mw_dino-wm.pth.tar",
    "dino_wm_pointmaze": "https://dl.fbaipublicfiles.com/jepa-wms/mz_dino-wm.pth.tar",
    "dino_wm_pusht": "https://dl.fbaipublicfiles.com/jepa-wms/pt_dino-wm.pth.tar",
    "dino_wm_wall": "https://dl.fbaipublicfiles.com/jepa-wms/wall_dino-wm.pth.tar",
    # V-JEPA-2-AC baseline models
    "vjepa2_ac_droid": "https://dl.fbaipublicfiles.com/jepa-wms/droid_vj2ac_noprop.pth.tar",
    "vjepa2_ac_oss": "https://dl.fbaipublicfiles.com/jepa-wms/droid_vj2ac_oss-prop.pth.tar",
}

# Image decoder URLs (loaded via config's heads_cfg.pretrain_dec_path.image_head field)
# These are not standalone models, but decoder heads trained on frozen backbones.
# Decoder assignment: DINO-WM uses dinov2_vits_224 (05norm), JEPA-WM uses INet variants
# (dv2vits for sim envs, dv3vitl for real-robot), VJ2AC uses vjepa2_vitg.
IMAGE_DECODER_URLS = {
    "dinov2_vits_224": "https://dl.fbaipublicfiles.com/jepa-wms/vm2m_lpips_dv2vits_vitldec_224_05norm.pth.tar",
    "dinov2_vits_224_INet": "https://dl.fbaipublicfiles.com/jepa-wms/vm2m_lpips_dv2vits_vitldec_224_INet.pth.tar",
    "dinov3_vitl_256_INet": "https://dl.fbaipublicfiles.com/jepa-wms/vm2m_lpips_dv3vitl_256_INet.pth.tar",
    "vjepa2_vitg_256_INet": "https://dl.fbaipublicfiles.com/jepa-wms/vm2m_lpips_vj2vitgnorm_vitldec_dup_256_INet.pth.tar",
}


def _get_checkpoint_path(model_name: str, use_hf: bool = True) -> str:
    """
    Get the checkpoint path for a model, trying Hugging Face Hub first.

    Args:
        model_name: Name of the model (e.g., 'jepa_wm_droid')
        use_hf: If True, try to download from Hugging Face Hub first

    Returns:
        Path to the downloaded checkpoint file, or URL for fallback download
    """
    if use_hf:
        try:
            from huggingface_hub import hf_hub_download

            # Try to download from Hugging Face Hub
            checkpoint_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=f"{model_name}.pth.tar",
            )
            return checkpoint_path
        except Exception:
            # Fall back to fbaipublicfiles URL
            pass

    # Return the fallback URL
    return MODEL_URLS.get(model_name)

# Model configurations: maps model name to (config_path, weight_key)
# weight_key is used to look up the URL in MODEL_URLS (may differ from model name for shared weights)
_MODEL_CONFIGS = {
    # JEPA-WM models
    "jepa_wm_metaworld": (
        "configs/evals/simu_env_planning/mw/jepa-wm/reach-wall_L2_cem_sourcexp_H6_nas3_ctxt2_r256_alpha0.1_ep48_decode.yaml",
        "jepa_wm_metaworld",
    ),
    "jepa_wm_droid": (
        "configs/evals/simu_env_planning/droid/jepa-wm/droid_L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r256_alpha0_ep64_decode.yaml",
        "jepa_wm_droid",
    ),
    "jepa_wm_pusht": (
        "configs/evals/simu_env_planning/pt/jepa-wm/pt_L2_cem_sourcedset_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml",
        "jepa_wm_pusht",
    ),
    "jepa_wm_pointmaze": (
        "configs/evals/simu_env_planning/mz/jepa-wm/mz_L2_cem_sourcerandstate_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml",
        "jepa_wm_pointmaze",
    ),
    "jepa_wm_wall": (
        "configs/evals/simu_env_planning/wall/jepa-wm/wall_L2_cem_sourcerandstate_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml",
        "jepa_wm_wall",
    ),
    # DINO-WM baseline models
    "dino_wm_metaworld": (
        "configs/evals/simu_env_planning/mw/dino-wm/reach_L2_cem_sourcexp_H6_nas3_ctxt2_r224_alpha0.1_ep48_decode.yaml",
        "dino_wm_metaworld",
    ),
    "dino_wm_pusht": (
        "configs/evals/simu_env_planning/pt/dino-wm/pt_L2_cem_sourcedset_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml",
        "dino_wm_pusht",
    ),
    "dino_wm_pointmaze": (
        "configs/evals/simu_env_planning/mz/dino-wm/mz_L2_cem_sourcerandstate_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml",
        "dino_wm_pointmaze",
    ),
    "dino_wm_wall": (
        "configs/evals/simu_env_planning/wall/dino-wm/wall_L2_cem_sourcerandstate_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml",
        "dino_wm_wall",
    ),
    "dino_wm_droid": (
        "configs/evals/simu_env_planning/droid/dino-wm/droid_L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r224_alpha0_ep64_decode.yaml",
        "dino_wm_droid",
    ),
    # V-JEPA-2-AC baseline models
    "vjepa2_ac_droid": (
        "configs/evals/simu_env_planning/droid/vj2ac/droid_L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r256_alpha0_ep64_decode.yaml",
        "vjepa2_ac_droid",
    ),
    # V-JEPA-2-AC OSS model from https://github.com/facebookresearch/vjepa2 with bug in loss (see JEPA-WMs paper's appendix)
    "vjepa2_ac_oss": (
        "configs/evals/simu_env_planning/droid/vj2ac_oss/droid_L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r256_alpha0_ep64_decode.yaml",
        "vjepa2_ac_oss",
    ),
}


def _load_model_with_config(config_path, model_name, device="cuda:0", pretrained=True):
    """
    Helper function to load model and preprocessor from config file.

    Uses init_module() from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds
    which can load checkpoints from either URLs or local paths.

    This function uses hardcoded action_dim, proprio_dim, and normalization statistics
    from app.plan_common.datasets.DATA_STATS, so datasets do not need to be downloaded
    to load pretrained models via torchhub.

    Args:
        config_path (str): Path to the config YAML file
        model_name (str): Name of the model (used to look up weight URL when pretrained=True)
        device (str): Device to load model on ('cuda:0' or 'cpu')
        pretrained (bool): If True, download and load pretrained weights from URL.
                          If False, load from local checkpoint path in config.

    Returns:
        tuple: (model, preprocessor) where model is EncPredWM and preprocessor handles normalization
    """
    import logging

    import torch
    import yaml

    from app.plan_common.datasets import get_data_stats
    from app.plan_common.datasets.preprocessor import Preprocessor
    from app.plan_common.datasets.transforms import make_inverse_transforms, make_transforms
    from evals.simu_env_planning.eval import init_module
    from src.utils.yaml_utils import expand_env_vars

    logging.basicConfig()
    log = logging.getLogger()

    # Load config
    with open(config_path, "r") as f:
        args_eval = yaml.safe_load(f)

    # Expand environment variables
    args_eval = expand_env_vars(args_eval)

    # Extract model kwargs
    model_kwargs = args_eval["model_kwargs"]
    cfgs_data = model_kwargs.get("data", {})
    cfgs_data_aug = model_kwargs.get("data_aug", {})
    wrapper_kwargs = model_kwargs.get("wrapper_kwargs", {})
    pretrain_kwargs = model_kwargs.get("pretrain_kwargs", {})

    # Set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(device)

    # Extract environment name from model_name (e.g., "jepa_wm_metaworld" -> "metaworld")
    # Handle special cases like "droid" which applies to both droid and robocasa
    env_name = model_name.split("_")[-1]
    if env_name == "oss":
        # vjepa2_ac_oss is a DROID model
        env_name = "droid"

    # Get hardcoded dimensions and normalization stats from registry (no dataset download needed)
    data_stats = get_data_stats(env_name)
    action_dim = data_stats["action_dim"]
    proprio_dim = data_stats["proprio_dim"]
    log.info(f"Using hardcoded dimensions for {env_name}: action_dim={action_dim}, proprio_dim={proprio_dim}")

    # Build transforms from config
    img_size = cfgs_data.get("img_size", 224)
    transform = make_transforms(
        img_size=img_size,
        normalize=cfgs_data_aug.get("normalize", [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]),
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
    )
    inverse_transform = make_inverse_transforms(
        img_size=img_size,
        **cfgs_data_aug,
    )

    # Build preprocessor with hardcoded normalization stats (no dataset access needed)
    preprocessor = Preprocessor(
        action_mean=torch.tensor(data_stats["action_mean"]),
        action_std=torch.tensor(data_stats["action_std"]),
        state_mean=torch.tensor(data_stats["state_mean"]),
        state_std=torch.tensor(data_stats["state_std"]),
        proprio_mean=torch.tensor(data_stats["proprio_mean"]),
        proprio_std=torch.tensor(data_stats["proprio_std"]),
        transform=transform,
        inverse_transform=inverse_transform,
    )
    log.info(f"Preprocessor attributes for {env_name}:")
    log.info(f"  action_mean: {preprocessor.action_mean}")
    log.info(f"  action_std: {preprocessor.action_std}")
    log.info(f"  proprio_mean: {preprocessor.proprio_mean}")
    log.info(f"  proprio_std: {preprocessor.proprio_std}")
    log.info(f"  state_mean: {preprocessor.state_mean}")
    log.info(f"  state_std: {preprocessor.state_std}")

    # Determine checkpoint source: try HF Hub first (if pretrained), then URL, then local path
    if pretrained and model_name in MODEL_URLS:
        checkpoint = _get_checkpoint_path(model_name, use_hf=True)
    else:
        checkpoint = model_kwargs.get("checkpoint")

    # Get folder (only needed for local checkpoint paths)
    pretrain_folder = args_eval.get("folder", None)
    module_name = model_kwargs.get("module_name")

    # Initialize model using init_module (handles both URLs and local paths)
    model = init_module(
        folder=pretrain_folder,
        checkpoint=checkpoint,
        module_name=module_name,
        model_kwargs=pretrain_kwargs,
        wrapper_kwargs=wrapper_kwargs,
        cfgs_data=cfgs_data,
        device=device,
        action_dim=action_dim,
        proprio_dim=proprio_dim,
        preprocessor=preprocessor,
    )

    log.info("Loaded encoder and predictor")
    return model, preprocessor


def _load_model(model_name, pretrained=True, device="cuda:0"):
    """
    Generic model loader that uses the model registry.

    Args:
        model_name (str): Name of the model to load
        pretrained (bool): If True, download and load pretrained weights from URL
        device (str): Device to load model on ('cuda:0' or 'cpu')

    Returns:
        tuple: (model, preprocessor) where model is EncPredWM ready for inference
    """
    if model_name not in _MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(_MODEL_CONFIGS.keys())}")

    config_rel_path, weight_key = _MODEL_CONFIGS[model_name]
    config_path = os.path.join(os.path.dirname(__file__), config_rel_path)
    return _load_model_with_config(config_path, model_name=weight_key, device=device, pretrained=pretrained)


def _make_model_fn(model_name):
    """Factory to create model loading functions for torch.hub."""

    def model_fn(pretrained=True, device="cuda:0", **kwargs):
        return _load_model(model_name, pretrained=pretrained, device=device)

    return model_fn


# Dynamically generate all model loading functions for torch.hub
jepa_wm_metaworld = _make_model_fn("jepa_wm_metaworld")
jepa_wm_robocasa = _make_model_fn("jepa_wm_robocasa")
jepa_wm_droid = _make_model_fn("jepa_wm_droid")
jepa_wm_pusht = _make_model_fn("jepa_wm_pusht")
jepa_wm_pointmaze = _make_model_fn("jepa_wm_pointmaze")
jepa_wm_wall = _make_model_fn("jepa_wm_wall")

dino_wm_metaworld = _make_model_fn("dino_wm_metaworld")
dino_wm_pusht = _make_model_fn("dino_wm_pusht")
dino_wm_pointmaze = _make_model_fn("dino_wm_pointmaze")
dino_wm_wall = _make_model_fn("dino_wm_wall")
dino_wm_droid = _make_model_fn("dino_wm_droid")

vjepa2_ac_droid = _make_model_fn("vjepa2_ac_droid")
vjepa2_ac_oss = _make_model_fn("vjepa2_ac_oss")
