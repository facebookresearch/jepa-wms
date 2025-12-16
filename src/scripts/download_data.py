#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Download datasets from HuggingFace for JEPA-WMs experiments."""

import argparse
import os
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

REPO_ID = "facebook/jepa-wms"

DATASETS = {
    "pusht": {
        "pattern": "pusht/*",
        "post_process": "unzip_and_rename",
        "zip_file": "pusht/pusht_noise.zip",
        "extract_to": "pusht_noise",
        "cleanup": ["pusht"],
    },
    "pointmaze": {
        "pattern": "point_maze/*",
        "post_process": "unzip_in_place",
        "zip_file": "point_maze/point_maze.zip",
        "cleanup": ["point_maze/point_maze.zip"],
    },
    "wall": {
        "pattern": "wall/*",
        "post_process": "rename",
        "rename_from": "wall",
        "rename_to": "wall_single",
    },
    "metaworld": {
        "pattern": "metaworld/*",
        "post_process": "rename",
        "rename_from": "metaworld",
        "rename_to": "Metaworld",
    },
    "robocasa": {
        "pattern": "robocasa/*",
        "post_process": None,
    },
    "franka": {
        "pattern": "franka_custom/*",
        "post_process": None,
    },
}


def download_dataset(name: str, dataset_root: Path, force: bool = False) -> None:
    """Download a single dataset from HuggingFace.

    Args:
        name: Dataset name (one of DATASETS keys)
        dataset_root: Root directory to download to
        force: If True, re-download even if dataset exists
    """
    config = DATASETS[name]
    pattern = config["pattern"]

    # Check if already exists
    target_dir = dataset_root / config.get("rename_to", config.get("extract_to", pattern.split("/")[0]))
    if target_dir.exists() and not force:
        print(f"  ✓ {name} already exists at {target_dir}, skipping (use --force to re-download)")
        return

    print(f"  Downloading {name}...")

    # Download from HuggingFace
    snapshot_download(
        REPO_ID,
        allow_patterns=pattern,
        repo_type="dataset",
        local_dir=str(dataset_root),
    )

    # Post-processing
    post_process = config.get("post_process")

    if post_process == "unzip_and_rename":
        zip_path = dataset_root / config["zip_file"]
        extract_to = dataset_root / config["extract_to"]
        print(f"  Extracting {zip_path} to {extract_to}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dataset_root)
        # Cleanup
        for item in config.get("cleanup", []):
            path = dataset_root / item
            if path.exists():
                shutil.rmtree(path) if path.is_dir() else path.unlink()

    elif post_process == "unzip_in_place":
        zip_path = dataset_root / config["zip_file"]
        extract_dir = zip_path.parent
        print(f"  Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        # Cleanup
        for item in config.get("cleanup", []):
            path = dataset_root / item
            if path.exists():
                path.unlink()

    elif post_process == "rename":
        src = dataset_root / config["rename_from"]
        dst = dataset_root / config["rename_to"]
        if src.exists() and not dst.exists():
            print(f"  Renaming {src} to {dst}...")
            src.rename(dst)

    print(f"  ✓ {name} ready")


def main():
    parser = argparse.ArgumentParser(
        description="Download JEPA-WMs datasets from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_data.py                     # Download all datasets
  python download_data.py --dataset pusht     # Download only Push-T
  python download_data.py --dataset pusht pointmaze  # Download multiple
  python download_data.py --list              # List available datasets
        """,
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=list(DATASETS.keys()),
        help="Dataset(s) to download. Downloads all if not specified.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=os.environ.get("DATASET_ROOT"),
        help="Root directory for datasets (default: $DATASET_ROOT)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if dataset exists",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit",
    )

    args = parser.parse_args()

    if args.list:
        print("Available datasets:")
        for name in DATASETS:
            print(f"  - {name}")
        return

    if not args.dataset_root:
        parser.error(
            "Dataset root not specified. Set DATASET_ROOT environment variable "
            "or use --dataset-root"
        )

    dataset_root = Path(args.dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)

    datasets_to_download = args.dataset if args.dataset else list(DATASETS.keys())

    print(f"Downloading to: {dataset_root}")
    print(f"Datasets: {', '.join(datasets_to_download)}\n")

    for name in datasets_to_download:
        download_dataset(name, dataset_root, force=args.force)

    print("\n✓ All downloads complete!")


if __name__ == "__main__":
    main()
