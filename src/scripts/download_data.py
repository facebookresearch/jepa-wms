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
import sys
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

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


def download_dataset(name: str, dataset_root: Path, force: bool = False) -> bool:
    """Download a single dataset from HuggingFace.

    Args:
        name: Dataset name (one of DATASETS keys)
        dataset_root: Root directory to download to
        force: If True, re-download even if dataset exists

    Returns:
        True if download succeeded, False if it failed
    """
    config = DATASETS[name]
    pattern = config["pattern"]

    # Check if already exists
    target_dir = dataset_root / config.get("rename_to", config.get("extract_to", pattern.split("/")[0]))
    if target_dir.exists() and not force:
        print(f"  ✓ {name} already exists at {target_dir}, skipping (use --force to re-download)")
        return True

    print(f"  Downloading {name}...")

    # Download from HuggingFace
    try:
        snapshot_download(
            REPO_ID,
            allow_patterns=pattern,
            repo_type="dataset",
            local_dir=str(dataset_root),
            local_dir_use_symlinks=False,
        )
    except GatedRepoError:
        print(f"\n  ✗ ERROR: The repository '{REPO_ID}' is gated and requires access approval.")
        print("    Please visit https://huggingface.co/datasets/facebook/jepa-wms to request access,")
        print("    then authenticate with: huggingface-cli login")
        return False
    except RepositoryNotFoundError:
        print(f"\n  ✗ ERROR: Repository '{REPO_ID}' not found.")
        print("    This could mean:")
        print("    1. The repository is private/gated and you are not authenticated")
        print("    2. The repository ID is incorrect")
        print("    To authenticate, run: huggingface-cli login")
        print("    For gated repos, first request access at: https://huggingface.co/datasets/facebook/jepa-wms")
        return False
    except Exception as e:
        # Check if it's a 404 error in the exception message (fallback case)
        error_str = str(e).lower()
        if "404" in error_str or "repository not found" in error_str:
            print(f"\n  ✗ ERROR: Could not access repository '{REPO_ID}'.")
            print("    This usually means you need to authenticate with HuggingFace.")
            print("    Run: huggingface-cli login")
            print("    If the repository is gated, first request access at:")
            print("    https://huggingface.co/datasets/facebook/jepa-wms")
            return False
        raise

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
    return True


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

    failed = []
    for name in datasets_to_download:
        success = download_dataset(name, dataset_root, force=args.force)
        if not success:
            failed.append(name)

    if failed:
        print(f"\n✗ Download failed for: {', '.join(failed)}")
        print("\nTo authenticate with HuggingFace, run:")
        print("  huggingface-cli login")
        sys.exit(1)
    else:
        print("\n✓ All downloads complete!")


if __name__ == "__main__":
    main()
