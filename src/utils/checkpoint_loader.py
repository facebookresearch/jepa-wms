import os
import random
import time
from typing import Any

import torch
from torch.serialization import MAP_LOCATION

from src.utils.logging import get_logger

logger = get_logger(os.path.basename(__file__))


def robust_checkpoint_loader(r_path: str, map_location: MAP_LOCATION = "cpu", max_retries: int = 3) -> Any:
    """
    Loads a checkpoint from a path, retrying up to max_retries times if the checkpoint is not found.
    """
    retries = 0

    while retries < max_retries:
        try:
            return torch.load(r_path, map_location=map_location)
        except Exception as e:
            logger.warning(f"Encountered exception when loading checkpoint {e}")
            retries += 1
            if retries < max_retries:
                sleep_time_s = (2**retries) * random.uniform(1.0, 1.1)
                logger.warning(f"Sleeping {sleep_time_s}s and trying again, count {retries}/{max_retries}")
                time.sleep(sleep_time_s)
                continue
            else:
                raise e
