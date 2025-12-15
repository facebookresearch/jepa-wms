#!/usr/bin/env python
"""
Generate Metaworld dataset paths CSV by scanning the h5folder directory.

Usage:
    python src/scripts/generate_metaworld_paths.py \
        --metaworld_root /path/to/Metaworld/h5folder \
        --output_path /path/to/output/metaworld_paths.csv \
        --episodes_per_task_seed 100
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate Metaworld dataset paths CSV")
    parser.add_argument("--metaworld_root", type=str, required=True, help="Root h5folder directory")
    parser.add_argument("--output_path", type=str, required=True, help="Output CSV path")
    parser.add_argument("--episodes_per_task_seed", type=int, default=99, help="Episodes per (task, seed)")
    args = parser.parse_args()

    root = Path(args.metaworld_root)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    episodes_by_task_seed: dict[tuple[str, int], list[tuple[int, str]]] = defaultdict(list)
    folder_pattern = re.compile(r"224_rgb_state_all_tasks_(mw-.+)_rgb_state_(\d+)")

    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        match = folder_pattern.match(folder.name)
        if not match:
            continue

        task, seed = match.group(1), int(match.group(2))
        for h5_file in folder.glob("ep*.h5"):
            ep_num = int(h5_file.stem[2:])  # "ep123" -> 123
            episodes_by_task_seed[(task, seed)].append((ep_num, str(h5_file)))

    filtered_paths = []
    for (task, seed), episodes in sorted(episodes_by_task_seed.items()):
        selected = [(ep_num, path) for ep_num, path in episodes if ep_num <= args.episodes_per_task_seed]
        selected.sort()
        filtered_paths.extend([path for _, path in selected])
        print(f"{task} (seed {seed}): {len(selected)}/{len(episodes)} episodes")

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        for idx, path in enumerate(filtered_paths):
            f.write(f"{path} {idx}\n")

    print(f"\nWritten {len(filtered_paths)} episodes to {args.output_path}")


if __name__ == "__main__":
    main()
