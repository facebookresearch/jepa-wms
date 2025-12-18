# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Common utilities for evaluation scripts.
"""
import logging
import os

import imageio
import matplotlib.pyplot as plt
import numpy as np

from app.plan_common.datasets.preprocessor import Preprocessor
from app.plan_common.datasets.transforms import make_inverse_transforms, make_transforms
from app.plan_common.datasets.utils import init_data
from src.datasets.utils.utils import get_dataset_paths

logger = logging.getLogger(__name__)


def make_datasets(cfgs_data, cfgs_data_aug, world_size=1, rank=0, filter_first_episodes=20):
    """
    Initialize dataset and preprocessor for evaluation.

    This is a shared utility function used by both simu_env_planning and unroll_decode evaluations.

    Args:
        cfgs_data: Data configuration dictionary with nested structure:
            - datasets: List of dataset names
            - img_size: Image size
            - validation: Validation config with val_datasets, etc.
            - loader: DataLoader config
            - custom: Custom dataset parameters
            - droid: DROID-specific parameters
        cfgs_data_aug: Data augmentation configuration with normalize, etc.
        world_size: Number of distributed processes
        rank: Current process rank
        filter_first_episodes: Number of episodes to filter to (for faster loading)

    Returns:
        tuple: (dset, data_preprocessor) where dset is the validation trajectory dataset
            and data_preprocessor is a Preprocessor instance
    """
    img_size = cfgs_data["img_size"]

    transform = make_transforms(
        img_size=img_size,
        normalize=cfgs_data_aug["normalize"],
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
    logger.info(f"{inverse_transform.std=}, {inverse_transform.mean=}")

    # Extract nested structures (matching train.py structure)
    cfgs_validation = cfgs_data.get("validation", {})
    cfgs_loader = cfgs_data.get("loader", {})
    cfgs_custom = cfgs_data.get("custom", {})
    cfgs_droid = cfgs_data.get("droid", {})

    datasets = cfgs_data.get("datasets", [])
    dataset_paths = get_dataset_paths(datasets)
    val_datasets = cfgs_validation.get("val_datasets", [])
    val_dataset_paths = get_dataset_paths(val_datasets) if val_datasets else None

    # Prepare data kwargs from config, flattening nested structures and filtering out non-init_data fields
    excluded_keys = [
        "datasets",
        "val_datasets",
        "img_size",
        # subfields of cfgs_data
        "validation",
        "loader",
        "custom",
        "droid",
    ]
    data_kwargs = {k: v for k, v in cfgs_data.items() if k not in excluded_keys}

    data_kwargs.update(cfgs_validation)
    data_kwargs.update(cfgs_loader)
    data_kwargs.update(cfgs_custom)
    data_kwargs.update(cfgs_droid)

    data_kwargs.update(
        {
            "data_paths": dataset_paths,
            "val_data_paths": val_dataset_paths,
            "transform": transform,
            "world_size": world_size,
            "rank": rank,
            "num_workers": 0,
            "filter_first_episodes": filter_first_episodes,
            # robocasa
            "output_rcasa_state": True,
            "output_rcasa_info": True,
        }
    )

    (
        dataset,
        val_dataset,
        traj_dataset,
        val_traj_dataset,
        unsupervised_loader,
        val_unsupervised_loader,
        unsupervised_sampler,
        viz_val_data_loader,
    ) = init_data(**data_kwargs)

    dset = val_traj_dataset
    data_preprocessor = Preprocessor(
        action_mean=dset.action_mean,
        action_std=dset.action_std,
        state_mean=dset.state_mean,
        state_std=dset.state_std,
        proprio_mean=dset.proprio_mean,
        proprio_std=dset.proprio_std,
        transform=transform,
        inverse_transform=inverse_transform,
    )
    return dset, data_preprocessor


def prepare_obs(obs_type, td):
    """Prepare observation based on observation type."""
    if obs_type == "rgb_state":
        return td
    elif obs_type == "state":
        return td["proprio"]
    elif obs_type == "rgb":
        return td["visual"]
    else:
        raise ValueError(f"Unknown observation type: {obs_type}")


def save_image(img_array, path):
    """Save image array to PDF file."""
    img_normalized = img_array / 255.0
    plt.figure(figsize=(10, 10))
    plt.imshow(img_normalized)
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", pad_inches=0, format="pdf")
    plt.close()
    logger.info(f"Saved image to {path}")


def save_video(frames, path, fps=10):
    """Save video frames to GIF."""
    frames = np.transpose(frames, (0, 2, 3, 1))
    if frames.dtype != np.uint8:
        frames = frames.astype(np.uint8)
    imageio.mimsave(path, frames, fps=fps, loop=0)
    logger.info(f"Saved GIF to {path}")


def log_media_local(media_dict, folder, step=0):
    """Log media files locally."""
    for key, value in media_dict.items():
        subfolder = os.path.join(folder, "/".join(key.split("/")))
        os.makedirs(subfolder, exist_ok=True)
        if key.endswith("_rollouts") or key.endswith("_noisy_actions"):
            image_path = os.path.join(subfolder, f"{step}.pdf")
            save_image(value, image_path)
        elif key.endswith("_rollout"):
            video_path = os.path.join(subfolder, f"{step}.gif")
            save_video(value, video_path)
