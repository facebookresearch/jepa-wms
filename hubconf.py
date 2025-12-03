"""
PyTorch Hub configuration for JEPA-WMs pretrained models.

Example usage:
    import torch

    # Load a pretrained JEPA-WM model for Metaworld
    model = torch.hub.load('facebookresearch/jepa-wms', 'jepa_wm_metaworld')

    # Load a pretrained JEPA-WM model for Push-T
    model = torch.hub.load('facebookresearch/jepa-wms', 'jepa_wm_pusht')
"""

dependencies = ['torch', 'torchvision']


def _load_model(url, model_name):
    """Helper function to load model from URL."""
    import torch
    from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import EncPredWM

    checkpoint = torch.hub.load_state_dict_from_url(url, map_location='cpu')

    # Extract model from checkpoint
    # The exact loading logic depends on your checkpoint structure
    # You may need to adjust this based on your actual checkpoint format
    model = EncPredWM()  # Initialize with appropriate config
    model.load_state_dict(checkpoint['model'])

    return model


def jepa_wm_metaworld(pretrained=True, **kwargs):
    """
    JEPA-WM model pretrained on Metaworld dataset.

    Resolution: 256x256
    Encoder: DINOv3 ViT-S/16
    Predictor depth: 6

    Args:
        pretrained (bool): Load pretrained weights
        **kwargs: Additional arguments passed to the model

    Returns:
        model: JEPA-WM model
    """
    if pretrained:
        url = 'https://dl.fbaipublicfiles.com/jepa-wms/jepa_wm_metaworld.pth'
        return _load_model(url, 'jepa_wm_metaworld')
    else:
        from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import EncPredWM
        return EncPredWM(**kwargs)


def jepa_wm_pusht(pretrained=True, **kwargs):
    """
    JEPA-WM model pretrained on Push-T dataset.

    Resolution: 224x224
    Predictor depth: 6

    Args:
        pretrained (bool): Load pretrained weights
        **kwargs: Additional arguments passed to the model

    Returns:
        model: JEPA-WM model
    """
    if pretrained:
        url = 'https://dl.fbaipublicfiles.com/jepa-wms/jepa_wm_pusht.pth'
        return _load_model(url, 'jepa_wm_pusht')
    else:
        from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import EncPredWM
        return EncPredWM(**kwargs)


def jepa_wm_pointmaze(pretrained=True, **kwargs):
    """
    JEPA-WM model pretrained on PointMaze dataset.

    Resolution: 224x224
    Predictor depth: 6

    Args:
        pretrained (bool): Load pretrained weights
        **kwargs: Additional arguments passed to the model

    Returns:
        model: JEPA-WM model
    """
    if pretrained:
        url = 'https://dl.fbaipublicfiles.com/jepa-wms/jepa_wm_pointmaze.pth'
        return _load_model(url, 'jepa_wm_pointmaze')
    else:
        from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import EncPredWM
        return EncPredWM(**kwargs)


def jepa_wm_wall(pretrained=True, **kwargs):
    """
    JEPA-WM model pretrained on Wall dataset.

    Resolution: 224x224
    Predictor depth: 6

    Args:
        pretrained (bool): Load pretrained weights
        **kwargs: Additional arguments passed to the model

    Returns:
        model: JEPA-WM model
    """
    if pretrained:
        url = 'https://dl.fbaipublicfiles.com/jepa-wms/jepa_wm_wall.pth'
        return _load_model(url, 'jepa_wm_wall')
    else:
        from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import EncPredWM
        return EncPredWM(**kwargs)


def jepa_wm_robocasa(pretrained=True, **kwargs):
    """
    JEPA-WM model pretrained on RoboCasa dataset.

    Resolution: 256x256
    Encoder: DINOv3 ViT-S/16
    Predictor depth: 6

    Args:
        pretrained (bool): Load pretrained weights
        **kwargs: Additional arguments passed to the model

    Returns:
        model: JEPA-WM model
    """
    if pretrained:
        url = 'https://dl.fbaipublicfiles.com/jepa-wms/jepa_wm_robocasa.pth'
        return _load_model(url, 'jepa_wm_robocasa')
    else:
        from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import EncPredWM
        return EncPredWM(**kwargs)


def jepa_wm_droid(pretrained=True, **kwargs):
    """
    JEPA-WM model pretrained on DROID dataset.

    Resolution: 256x256
    Encoder: DINOv3 ViT-S/16
    Predictor depth: 6

    Args:
        pretrained (bool): Load pretrained weights
        **kwargs: Additional arguments passed to the model

    Returns:
        model: JEPA-WM model
    """
    if pretrained:
        url = 'https://dl.fbaipublicfiles.com/jepa-wms/jepa_wm_droid.pth'
        return _load_model(url, 'jepa_wm_droid')
    else:
        from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import EncPredWM
        return EncPredWM(**kwargs)


def dino_wm_metaworld(pretrained=True, **kwargs):
    """
    DINO-WM baseline model pretrained on Metaworld dataset.

    Resolution: 224x224
    Predictor depth: 6

    Args:
        pretrained (bool): Load pretrained weights
        **kwargs: Additional arguments passed to the model

    Returns:
        model: DINO-WM model
    """
    if pretrained:
        url = 'https://dl.fbaipublicfiles.com/jepa-wms/dino_wm_metaworld.pth'
        return _load_model(url, 'dino_wm_metaworld')
    else:
        from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import EncPredWM
        return EncPredWM(**kwargs)


def dino_wm_pusht(pretrained=True, **kwargs):
    """
    DINO-WM baseline model pretrained on Push-T dataset.

    Resolution: 224x224
    Predictor depth: 6

    Args:
        pretrained (bool): Load pretrained weights
        **kwargs: Additional arguments passed to the model

    Returns:
        model: DINO-WM model
    """
    if pretrained:
        url = 'https://dl.fbaipublicfiles.com/jepa-wms/dino_wm_pusht.pth'
        return _load_model(url, 'dino_wm_pusht')
    else:
        from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import EncPredWM
        return EncPredWM(**kwargs)


def dino_wm_pointmaze(pretrained=True, **kwargs):
    """
    DINO-WM baseline model pretrained on PointMaze dataset.

    Resolution: 224x224
    Predictor depth: 6

    Args:
        pretrained (bool): Load pretrained weights
        **kwargs: Additional arguments passed to the model

    Returns:
        model: DINO-WM model
    """
    if pretrained:
        url = 'https://dl.fbaipublicfiles.com/jepa-wms/dino_wm_pointmaze.pth'
        return _load_model(url, 'dino_wm_pointmaze')
    else:
        from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import EncPredWM
        return EncPredWM(**kwargs)


def dino_wm_wall(pretrained=True, **kwargs):
    """
    DINO-WM baseline model pretrained on Wall dataset.

    Resolution: 224x224
    Predictor depth: 6

    Args:
        pretrained (bool): Load pretrained weights
        **kwargs: Additional arguments passed to the model

    Returns:
        model: DINO-WM model
    """
    if pretrained:
        url = 'https://dl.fbaipublicfiles.com/jepa-wms/dino_wm_wall.pth'
        return _load_model(url, 'dino_wm_wall')
    else:
        from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import EncPredWM
        return EncPredWM(**kwargs)


def dino_wm_droid(pretrained=True, **kwargs):
    """
    DINO-WM baseline model pretrained on DROID dataset.

    Resolution: 224x224
    Predictor depth: 6

    Args:
        pretrained (bool): Load pretrained weights
        **kwargs: Additional arguments passed to the model

    Returns:
        model: DINO-WM model
    """
    if pretrained:
        url = 'https://dl.fbaipublicfiles.com/jepa-wms/dino_wm_droid.pth'
        return _load_model(url, 'dino_wm_droid')
    else:
        from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import EncPredWM
        return EncPredWM(**kwargs)
