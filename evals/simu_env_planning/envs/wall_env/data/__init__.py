from enum import Enum, auto

from .configs import ConfigBase
from .single import DotDataset, DotDatasetConfig, Sample
from .wall import WallDataset, WallDatasetConfig


class DatasetType(Enum):
    Single = auto()
    Multiple = auto()
    Wall = auto()
    WallExpert = auto()
    WallEigenfunc = auto()
