#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Script to generate the DROID dataset paths CSV file.

This script walks through the DROID dataset directory structure and generates
a CSV file where each line contains a path to a valid episode directory
(containing trajectory.h5) and its index.

Usage:
    python src/scripts/generate_droid_paths.py \
        --droid_root /path/to/droid_raw/1.0.1 \
        --output_path /path/to/output/droid_paths.csv

    # With parallelization (recommended for large datasets)
    python src/scripts/generate_droid_paths.py \
        --droid_root /path/to/droid_raw/1.0.1 \
        --output_path /path/to/output/droid_paths_gen.csv \
        --num_workers 16

The DROID dataset structure is expected to be:
    droid_raw/
    └── 1.0.1/
        ├── <LAB_NAME>/
        │   └── success/
        │       └── <DATE>/
        │           └── <DATETIME_FOLDER>/
        │               ├── trajectory.h5
        │               ├── metadata.json
        │               └── recordings/
        │                   └── MP4/
        │                       └── *.mp4
        └── ...

Output CSV format (space-separated):
    /path/to/episode_folder 0
    /path/to/episode_folder 1
    ...
"""

import argparse
import os
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional

from tqdm import tqdm


def _find_episodes_in_subdir(
    subdir: Path,
    trajectory_filename: str,
) -> list[str]:
    """
    Find all episode directories within a subdirectory.
    This function is designed to be called in parallel.
    """
    episode_paths = []
    for root, dirs, files in os.walk(subdir):
        if trajectory_filename in files:
            episode_paths.append(root)
    return episode_paths


def find_droid_episodes(
    droid_root: str | Path,
    trajectory_filename: str = "trajectory.h5",
    num_workers: int = 1,
) -> list[str]:
    """
    Walk through the DROID dataset directory and find all valid episode directories.

    An episode directory is considered valid if it contains a trajectory.h5 file.

    Args:
        droid_root: Root directory of the DROID dataset (e.g., droid_raw/1.0.1)
        trajectory_filename: Name of the trajectory file to look for
        num_workers: Number of parallel workers to use

    Returns:
        List of absolute paths to valid episode directories
    """
    droid_root = Path(droid_root)

    if not droid_root.exists():
        raise FileNotFoundError(f"DROID root directory not found: {droid_root}")

    print(f"Searching for episodes in: {droid_root}")

    subdirs = [d for d in droid_root.iterdir() if d.is_dir()]
    print(f"Found {len(subdirs)} top-level directories (labs)")

    if num_workers <= 1:
        episode_paths = []
        for subdir in tqdm(subdirs, desc="Scanning directories"):
            episode_paths.extend(_find_episodes_in_subdir(subdir, trajectory_filename))
    else:
        print(f"Using {num_workers} parallel workers")
        worker_fn = partial(_find_episodes_in_subdir, trajectory_filename=trajectory_filename)

        episode_paths = []
        with Pool(num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(worker_fn, subdirs),
                    total=len(subdirs),
                    desc="Scanning directories",
                )
            )
        for result in results:
            episode_paths.extend(result)

    episode_paths.sort()
    return episode_paths


def write_paths_csv(
    episode_paths: list[str],
    output_path: str | Path,
    train_val_split: Optional[float] = None,
) -> None:
    """
    Write episode paths to a CSV file.

    Args:
        episode_paths: List of paths to episode directories
        output_path: Path to output CSV file
        train_val_split: If provided, fraction of data to use for training.
                        Will create two files: output_train.csv and output_val.csv
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if train_val_split is not None:
        assert 0 < train_val_split < 1, "train_val_split must be between 0 and 1"
        num_train = int(len(episode_paths) * train_val_split)

        train_paths = episode_paths[:num_train]
        val_paths = episode_paths[num_train:]

        train_output = output_path.parent / f"{output_path.stem}_train{output_path.suffix}"
        val_output = output_path.parent / f"{output_path.stem}_val{output_path.suffix}"

        _write_single_csv(train_paths, train_output)
        _write_single_csv(val_paths, val_output)

        print(f"Train CSV written to: {train_output} ({len(train_paths)} episodes)")
        print(f"Val CSV written to: {val_output} ({len(val_paths)} episodes)")
    else:
        _write_single_csv(episode_paths, output_path)
        print(f"CSV written to: {output_path} ({len(episode_paths)} episodes)")


def _write_single_csv(paths: list[str], output_path: Path) -> None:
    """Write paths to a single CSV file with format: <path> <index>"""
    with open(output_path, "w") as f:
        for idx, path in enumerate(paths):
            f.write(f"{path} {idx}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate DROID dataset paths CSV file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--droid_root",
        type=str,
        required=True,
        help="Root directory of the DROID dataset (containing lab folders)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for the CSV file",
    )
    parser.add_argument(
        "--trajectory_filename",
        type=str,
        default="trajectory.h5",
        help="Name of the trajectory file to look for (default: trajectory.h5)",
    )
    parser.add_argument(
        "--train_val_split",
        type=float,
        default=None,
        help="If provided, split data into train/val with this fraction for training",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help=f"Number of parallel workers for directory scanning (default: 1, max recommended: {cpu_count()})",
    )

    args = parser.parse_args()

    episode_paths = find_droid_episodes(
        droid_root=args.droid_root,
        trajectory_filename=args.trajectory_filename,
        num_workers=args.num_workers,
    )

    if len(episode_paths) == 0:
        print(f"Warning: No episodes found in {args.droid_root}")
        print("Make sure the directory contains subdirectories with trajectory.h5 files")
        return

    print(f"Found {len(episode_paths)} episodes")

    write_paths_csv(
        episode_paths=episode_paths,
        output_path=args.output_path,
        train_val_split=args.train_val_split,
    )


if __name__ == "__main__":
    main()
