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

# Model weight URLs from https://dl.fbaipublicfiles.com/jepa-wms/
MODEL_URLS = {
    # JEPA-WM models
    "jepa_wm_droid": "https://dl.fbaipublicfiles.com/jepa-wms/jepa_wm_droid.pth",
    "jepa_wm_metaworld": "https://dl.fbaipublicfiles.com/jepa-wms/jepa_wm_metaworld.pth",
    "jepa_wm_pusht": "https://dl.fbaipublicfiles.com/jepa-wms/jepa_wm_pusht.pth",
    "jepa_wm_pointmaze": "https://dl.fbaipublicfiles.com/jepa-wms/jepa_wm_pointmaze.pth",
    "jepa_wm_wall": "https://dl.fbaipublicfiles.com/jepa-wms/jepa_wm_wall.pth",
    # DINO-WM baseline models
    "dino_wm_droid": "https://dl.fbaipublicfiles.com/jepa-wms/droid/dino-wm/jepa-latest.pth.tar",
    "dino_wm_metaworld": "https://dl.fbaipublicfiles.com/jepa-wms/dino_wm_metaworld.pth",
    "dino_wm_pusht": "https://dl.fbaipublicfiles.com/jepa-wms/dino_wm_pusht.pth",
    "dino_wm_pointmaze": "https://dl.fbaipublicfiles.com/jepa-wms/dino_wm_pointmaze.pth",
    "dino_wm_wall": "https://dl.fbaipublicfiles.com/jepa-wms/dino_wm_wall.pth",
    # V-JEPA-2-AC baseline models
    "vjepa2_ac_droid": "https://dl.fbaipublicfiles.com/jepa-wms/vjepa2_ac_droid.pth",
}

# Model configurations: maps model name to (config_path, weight_key)
# weight_key is used to look up the URL in MODEL_URLS (may differ from model name for shared weights)
_MODEL_CONFIGS = {
    # JEPA-WM models
    "jepa_wm_metaworld": (
        "configs/evals/simu_env_planning/mw/jepa-wm/reach-wall_L2_ng_sourcexp_H6_nas3_ctxt2_r256_alpha0.1_ep48.yaml",
        "jepa_wm_metaworld",
    ),
    "jepa_wm_droid": (
        "configs/evals/simu_env_planning/droid/droid_L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3.yaml",
        "jepa_wm_droid",
    ),
    "jepa_wm_pusht": (
        "configs/evals/simu_env_planning/pt/jepa-wm/pt_L2_cem_sourcedset_H6_nas6_ctxt2_r224_alpha0.1_ep96.yaml",
        "jepa_wm_pusht",
    ),
    "jepa_wm_pointmaze": (
        "configs/evals/simu_env_planning/mz/jepa-wm/mz_L2_cem_sourcerandstate_H6_nas6_ctxt2_r224_alpha0.1_ep96.yaml",
        "jepa_wm_pointmaze",
    ),
    "jepa_wm_wall": (
        "configs/evals/simu_env_planning/wall/jepa-wm/wall_L2_cem_sourcerandstate_H6_nas6_ctxt2_r224_alpha0.1_ep96.yaml",
        "jepa_wm_wall",
    ),
    # DINO-WM baseline models
    "dino_wm_metaworld": (
        "configs/evals/simu_env_planning/mw/dino-wm/reach_L2_ng_sourcexp_H6_nas3_ctxt2_r224_alpha0.1_ep48_decode.yaml",
        "dino_wm_metaworld",
    ),
    "dino_wm_pusht": (
        "configs/evals/simu_env_planning/pt/dino-wm/pt_L2_cem_sourcedset_H6_nas6_ctxt2_r224_alpha0.1_ep96.yaml",
        "dino_wm_pusht",
    ),
    "dino_wm_pointmaze": (
        "configs/evals/simu_env_planning/mz/dino-wm/mz_L2_cem_sourcerandstate_H6_nas6_ctxt2_r224_alpha0.0_ep96.yaml",
        "dino_wm_pointmaze",
    ),
    "dino_wm_wall": (
        "configs/evals/simu_env_planning/wall/dino-wm/wall_L2_cem_sourcerandstate_H6_nas6_ctxt2_r224_alpha0.1_ep96.yaml",
        "dino_wm_wall",
    ),
    "dino_wm_droid": (
        "configs/evals/simu_env_planning/droid/dino-wm/droid_L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r224_alpha0_ep64_decode.yaml",
        "dino_wm_droid",
    ),
    # V-JEPA-2-AC baseline models
    "vjepa2_ac_droid": (
        "configs/evals/simu_env_planning/droid/vj2ac/droid_L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r256_alpha0.0_ep64_decode.yaml",
        "vjepa2_ac_droid",
    ),
}


def _load_model_with_config(config_path, model_name, device="cuda:0", pretrained=True):
    """
    Helper function to load model and preprocessor from config file.

    Uses init_module() from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds
    which can load checkpoints from either URLs or local paths.

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

    from evals.simu_env_planning.eval import init_module, make_datasets
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

    # Build dataset and preprocessor
    dset, preprocessor = make_datasets(cfgs_data, cfgs_data_aug, world_size=1, rank=0)

    # Determine checkpoint source: URL (if pretrained) or local path from config
    if pretrained and model_name in MODEL_URLS:
        checkpoint = MODEL_URLS[model_name]
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
        action_dim=dset.action_dim,
        proprio_dim=dset.proprio_dim,
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
