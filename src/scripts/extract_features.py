#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pre-extract DINO encoder features from trajectory datasets.

Extracts per-frame encoder features and saves them as tensors,
eliminating video decoding and encoder forward passes during training.

Usage:
    python src/scripts/extract_features.py \
        --dataset pusht \
        --feature-key x_norm_clstoken \
        --output-dir $JEPAWM_DSET/pusht_noise_clstoken
"""

import argparse
import os
from pathlib import Path

import torch
from torchvision import transforms
from tqdm import tqdm

from app.plan_common.models.dino import DinoEncoder
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Dataset-specific loaders
DATASET_LOADERS = {
    "pusht": {
        "class": "app.plan_common.datasets.pusht_dset.PushTDataset",
        "default_subdir": "pusht_noise",
        "splits": ["train", "val"],
    },
}

# Standard normalization transforms matching training configs
NORMALIZE_TRANSFORMS = {
    "imagenet": transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    "half": transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
}


def load_dataset(name, data_path, split, transform):
    """Load a trajectory dataset by name."""
    if name == "pusht":
        from app.plan_common.datasets.pusht_dset import PushTDataset

        return PushTDataset(
            n_rollout=None,
            transform=transform,
            data_path=str(Path(data_path) / split),
            normalize_action=True,
            with_velocity=True,
        )
    else:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_LOADERS.keys())}")


@torch.no_grad()
def extract_episode_features(encoder, dataset, idx, img_size, batch_size=64, device="cuda"):
    """Extract encoder features for all frames of a single episode.

    Args:
        encoder: Frozen DINO encoder
        dataset: TrajDataset instance
        idx: Episode index
        img_size: Target image size for resizing
        batch_size: Frames to process per forward pass
        device: Torch device

    Returns:
        features: (T, num_tokens, D) tensor of encoder features
    """
    seq_len = dataset.get_seq_length(idx)
    frames = list(range(seq_len))

    # Load all frames via get_frames (handles video decoding)
    obs, act, state, reward, meta = dataset.get_frames(idx, frames)
    images = obs["visual"]  # (T, C, H, W), already [0,1] and transformed

    # Resize if needed (dataset may return different resolution)
    if images.shape[-1] != img_size or images.shape[-2] != img_size:
        images = torch.nn.functional.interpolate(images, size=(img_size, img_size), mode="bilinear")

    # Extract features in batches
    all_features = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].to(device)
        features = encoder(batch)  # (batch, num_tokens, D)
        all_features.append(features.cpu())

    return torch.cat(all_features, dim=0)  # (T, num_tokens, D)


def extract_and_save(encoder, dataset, output_dir, img_size, batch_size=64, device="cuda"):
    """Extract features for all episodes and save to disk.

    Saves per-episode files: features_ep00000.pt, features_ep00001.pt, ...
    Each file contains a tensor of shape (T, num_tokens, D).
    Also saves metadata.pt with encoder config and feature shapes.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_episodes = len(dataset)
    logger.info(f"Extracting features for {num_episodes} episodes...")

    sample_features = None
    for idx in tqdm(range(num_episodes), desc="Extracting"):
        features = extract_episode_features(encoder, dataset, idx, img_size, batch_size, device)
        torch.save(features, output_dir / f"features_ep{idx:05d}.pt")
        if sample_features is None:
            sample_features = features

    # Save metadata
    metadata = {
        "num_episodes": num_episodes,
        "feature_dim": sample_features.shape[-1],
        "num_tokens": sample_features.shape[1],
        "encoder_name": encoder.name,
        "feature_key": encoder.feature_key,
        "patch_size": encoder.patch_size,
    }
    torch.save(metadata, output_dir / "metadata.pt")
    logger.info(f"Saved {num_episodes} episodes to {output_dir}")
    logger.info(f"Feature shape per frame: ({metadata['num_tokens']}, {metadata['feature_dim']})")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-extract DINO encoder features from trajectory datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, required=True, choices=list(DATASET_LOADERS.keys()))
    parser.add_argument("--dataset-root", type=str, default=os.environ.get("JEPAWM_DSET"))
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save extracted features")
    parser.add_argument("--enc-version", type=str, default="dinov2_vits14", help="DINO encoder version")
    parser.add_argument("--feature-key", type=str, default="x_norm_clstoken", choices=["x_norm_clstoken", "x_norm_patchtokens"])
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64, help="Frames per encoder forward pass")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--normalize", type=str, default="imagenet", choices=["imagenet", "half", "none"])

    args = parser.parse_args()

    if not args.dataset_root:
        parser.error("Dataset root not specified. Set JEPAWM_DSET or use --dataset-root")

    device = torch.device(args.device)

    # Build transform (resize + normalize, matching training config)
    transform_list = [transforms.Resize((args.img_size, args.img_size), antialias=True)]
    if args.normalize != "none":
        transform_list.append(NORMALIZE_TRANSFORMS[args.normalize])
    transform = transforms.Compose(transform_list)

    # Load encoder
    logger.info(f"Loading encoder: {args.enc_version} with feature_key={args.feature_key}")
    encoder = DinoEncoder(name=args.enc_version, feature_key=args.feature_key).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Extract for each split
    dataset_config = DATASET_LOADERS[args.dataset]
    data_path = str(Path(args.dataset_root) / dataset_config["default_subdir"])

    for split in dataset_config["splits"]:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing split: {split}")
        logger.info(f"{'='*50}")

        dataset = load_dataset(args.dataset, data_path, split, transform)
        output_dir = Path(args.output_dir) / split
        extract_and_save(encoder, dataset, output_dir, args.img_size, args.batch_size, device)

    logger.info("\nDone! Features saved to: " + args.output_dir)


if __name__ == "__main__":
    main()
