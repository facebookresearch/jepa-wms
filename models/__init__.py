# This module provides compatibility for loading checkpoints
# saved from the dino_wm codebase which uses 'models.*' import paths
from . import vit
from . import vqvae
from . import proprio

__all__ = ["vit", "vqvae", "proprio"]
