import argparse
import glob
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ALIASES
from app.plan_common.plot.aliases import (
    eval_setup_aliases,
    hist1_eval_setup_aliases,
    unif_eval_setup_aliases_across_tasks,
)
from src.utils.yaml_utils import expand_env_vars

CHECKPOINT_ROOT = os.environ.get("CHECKPOINT_ROOT", "~")
base_dir = "app/plan_common/plot/local/plan_joint_per_design_choice"

task_groups_mapping = {
    "droid": "DROID",
    "pt": "Push-T",
    "mz": "Maze",
    "wall": "Wall",
    "mw-reach": "MW-\nReach",
    "mw-reach-wall": "MW-\nReach-\nWall",
    "rcasa-reach": "Rc-R",
    "rcasa-pick": "Rc-P",
    "rcasa-place": "Rc-Pl",
    "rcasa-reach-pick": "Rc-RP",
    "rcasa-pick-place": "Rc-PP",
    "rcasa-reach-pick-place": "Rc-RPP",
}

# Hardcoded order for task groups in LaTeX tables
TASK_GROUP_ORDER = ["Maze", "Wall", "Push-T", "MW-\nReach", "MW-\nReach-\nWall", "Rc-R", "Rc-Pl", "DROID"]

best_eval_setup_per_task_group = {
    "Push-T": r"CEM $L_2$",
    "Maze": r"CEM rand $L_2$",
    "Wall": r"CEM rand $L_2$",
    # "MW-\nReach": r"CEM $L_2$",
    # "MW-\nReach-\nWall": r"CEM $L_2$",
    "MW-\nReach": r"NG $L_2$",
    "MW-\nReach-\nWall": r"NG $L_2$",
    # ===
    "DROID": r"CEM H3 $L_2$ max0.1",
    # "DROID": r"CEM H3 $L_2$ max0.1 ep64",
    # "DROID": r"CEM H3 $L_2$ max0.1",  # maybe comment out for the planner plot
    # "DROID": r"CEM H3 $L_2$ max0.1",  # maybe comment out for the planner plot
    # ===
    # "Rc-R": r"CEM $L_2$ ep32",
    "Rc-R": r"CEM $L_2$",
    # "Rc-P": r"CEM $L_2$",
    # "Rc-Pl": r"CEM $L_2$ ep32",
    "Rc-Pl": r"CEM $L_2$",
    # "Rc-R": r"NG $L_1$",
    # "Rc-Pl": r"NG $L_1$",
    "Rc-RP": r"CEM $L_2$",
    "Rc-PP": r"CEM $L_2$",
    "Rc-RPP": r"CEM $L_2$",
}

exclude_eval_folders = {
    "droid": [
        "droid_L2_cem_sourcedset_H6_nas6_ctxt2_gH6_r256_alpha0.0_ep32_decode",
        "droid_L2_cem_sourcedset_H1_nas1_ctxt2_gH6_r256_alpha0.0_ep16_decode",
        "droid_L2_cem_sourcedset_H1_nas1_ctxt2_gH3_r256_alpha0.0_ep16_decode",
        "droid_L2_cem_sourcedset_H1_nas1_maxnorm01_ctxt2_gH3_r256_alpha0.0_ep16_decode",
        "droid_L2_cem_sourcedset_H1_nas1_maxnorm01_momentum015_ctxt2_gH3_r256_alpha0.0_ep16_decode",
        "droid_L2_cem_sourcedset_H3_nas3_ctxt2_gH6_r256_alpha0.0_ep16_decode",
        "droid_L2_cem_sourcedset_H3_nas3_maxnorm01_momentum015_ctxt2_gH6_r256_alpha0.0_ep16_decode",
        "droid_L2_ng_sourcedset_H1_nas1_ctxt2_gH3_r256_alpha0.0_ep16_decode",
        "droid_L2_ng_sourcedset_H1_nas1_maxnorm01_momentum015_ctxt2_gH3_r256_alpha0.0_ep16_decode",
        "droid_L2_ng_sourcedset_H3_nas3_maxnorm01_ctxt2_gH6_r256_alpha0.0_ep16_decode",
        "droid_L2_ng_sourcedset_H3_nas3_maxnorm01_momentum015_ctxt2_gH6_r256_alpha0.0_ep16_decode",
    ]
}

# Define task-specific cut_eval_setup strategies
task_cut_eval_setup_mapping = {
    "droid": "ctxt",
    # "droid": "ep",
    "pt": "ctxt",
    "mz": "ctxt",
    "wall": "ctxt",
    "mw-reach": "ctxt",
    "mw-reach-wall": "ctxt",
    "rcasa-reach": "ctxt",
    # "rcasa-reach": "ep",
    "rcasa-pick": "ctxt",
    "rcasa-place": "ctxt",
    # "rcasa-place": "ep",
    "rcasa-reach-pick": "ctxt",
    "rcasa-pick-place": "ctxt",
    "rcasa-reach-pick-place": "ctxt",
}

last_n_epochs = {
    "droid": 100,
    "pt": 10,
    "mz": 10,
    "wall": 10,
    "mw-reach": 30,
    "mw-reach-wall": 30,
    # "mw-reach": 10,
    # "mw-reach-wall": 10,
    "rcasa-reach": 100,
    "rcasa-pick": 100,
    "rcasa-place": 100,
    "rcasa-reach-pick": 100,
    "rcasa-pick-place": 100,
    "rcasa-reach-pick-place": 100,
}
start_from_epoch = {
    "droid": 215,
    "pt": 40,
    "mz": 40,
    "wall": 40,
    "mw-reach": 20,
    "mw-reach-wall": 20,
    # "mw-reach": 40,
    # "mw-reach-wall": 40,
    "rcasa-reach": 215,
    "rcasa-pick": 215,
    "rcasa-place": 215,
    "rcasa-reach-pick": 215,
    "rcasa-pick-place": 215,
    "rcasa-reach-pick-place": 215,
}


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        print(f"⏱️ {self.name} took {self.end - self.start:.2f} seconds")


def clean_task_name(task_name, folder_path=None):
    """
    Clean the task name to match the correct format based on context from folder path.

    Args:
        task_name: The raw task name
        folder_path: The full folder path that provides context

    Returns:
        Properly formatted task name with environment prefix
    """
    # Common mappings regardless of environment
    task_name_mappings = {
        "reachwall": "mw-reach-wall",
        "binpicking": "mw-bin-picking",
        "buttonpresstopdownwall": "mw-button-press-topdown-wall",
    }

    # If no folder path provided, use the mappings as before
    if folder_path is None:
        return task_name_mappings.get(task_name, task_name)

    # For ambiguous task names, determine environment from folder path
    if task_name in ["reach", "reach-wall", "pick", "place", "reach-pick", "pick-place"]:
        if "droid" in folder_path.lower():
            # DROID environment tasks
            if task_name == "reach":
                return "rcasa-reach"
            elif task_name in ["pick", "place", "reach-pick", "pick-place", "reach-pick-place"]:
                return f"rcasa-{task_name}"
        elif "mw" in folder_path.lower():
            # MetaWorld environment tasks
            if task_name == "reach":
                return "mw-reach"
            elif task_name == "reach-wall":
                return "mw-reach-wall"

    # Fall back to original mappings if no environment context match
    return task_name_mappings.get(task_name, task_name)


def load_task_data(folder):
    """
    Load task data from a given folder.

    Returns:
        pd.DataFrame: DataFrame with epochs and metrics including optional metrics if available
    """
    # Initialize with required metrics
    task_data = {"epoch": [], "SR": []}

    # Track which optional metrics are available
    optional_metrics = [
        "Reward",
        "Act_err",
        "Act_err_xyz",
        "Act_err_orient",
        "Act_err_closure",
        "Total_LPIPS",
        "Total_Emb_L2",
    ]

    # Map CSV column names to our metric names
    column_mapping = {
        "episode_success": ("SR", lambda x: x * 100),  # Convert to percentage
        "episode_reward": ("Reward", lambda x: x),
        "ep_end_dist": ("Act_err", lambda x: x),
        "ep_end_dist_xyz": ("Act_err_xyz", lambda x: x),
        "ep_end_dist_orientation": ("Act_err_orient", lambda x: x),
        "ep_end_dist_closure": ("Act_err_closure", lambda x: x),
        "ep_total_lpips": ("Total_LPIPS", lambda x: x),
        "ep_total_emb_l2": ("Total_Emb_L2", lambda x: x),
    }

    # Initialize all optional metrics to empty lists
    for metric in optional_metrics:
        task_data[metric] = []

    epoch_folders = [
        f for f in glob.glob(os.path.join(folder, "epoch-*")) if os.path.basename(f).split("-")[-1].isdigit()
    ]
    epoch_folders = sorted(epoch_folders, key=lambda x: int(os.path.basename(x).split("-")[-1]))

    for epoch_folder in epoch_folders:
        epoch_name = os.path.basename(epoch_folder)
        try:
            epoch = int("".join(filter(str.isdigit, epoch_name)))
            eval_file_path = os.path.join(epoch_folder, "eval.csv")

            if os.path.exists(eval_file_path):
                try:
                    task_df = pd.read_csv(eval_file_path)
                    task_data["epoch"].append(epoch)

                    # Process each metric based on available columns
                    for csv_col, (metric_name, transform_fn) in column_mapping.items():
                        if csv_col in task_df.columns:
                            value = transform_fn(task_df[csv_col].values[-1])
                            task_data[metric_name].append(value)
                        elif metric_name in task_data:  # Fill with NaN if column doesn't exist
                            task_data[metric_name].append(np.nan)

                except Exception as e:
                    print(f"Error reading {eval_file_path}: {e}")
        except ValueError:
            print(f"Ignoring non-standard epoch folder: {epoch_name}")

    # Remove metrics that weren't found in any file
    for metric in optional_metrics:
        if all(np.isnan(x) for x in task_data[metric]) or len(task_data[metric]) == 0:
            del task_data[metric]

    return pd.DataFrame(task_data)


def collect_task_eval_data(
    model_paths,
    task_subset,
    eval_setup_aliases=None,
    collect_subfolder_seeds=True,
    cut_eval_setup="ctxt",  # not used anymore
    exclude_eval_folders=None,
    verbose=True,
    max_workers=None,
    hist1_folders=[],
):
    """
    Collect evaluation data for tasks across different models and seeds using parallel processing.

    Args:
        model_paths: List of model paths to collect data from
        task_subset: List of task names to include
        eval_setup_aliases: Optional dict to map eval setups to aliases
        collect_subfolder_seeds: Whether to look for seed subfolders
        verbose: Whether to print progress information
        max_workers: Maximum number of worker threads (defaults to min(32, os.cpu_count() + 4))
        hist1_folders: List of folder patterns to identify as hist1 folders (for special handling)

    Returns:
        dict: {(model_path, task_name, eval_setup, seed): {metric_name: DataFrame}}
            where DataFrame contains 'epoch' and metric values
    """
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 4) + 4)
    print(f"Using max_workers={max_workers} for parallel processing")

    task_eval_data = {}

    def process_eval_folder(args):
        folder_path, eval_folder, seed = args
        folder_name = os.path.basename(eval_folder)

        if exclude_eval_folders and folder_name in exclude_eval_folders:
            return None

        parts = folder_name.split("_")
        task_name = clean_task_name(parts[0], folder_path)

        # Check which model folder this eval folder belongs to for hist1 detection
        parent_model_path = folder_path
        is_hist1_folder = any(hist1 in parent_model_path for hist1 in hist1_folders)

        # Determine which cut_eval_setup strategy to use for this task
        cut_eval_setup = task_cut_eval_setup_mapping.get(task_name)

        # Extract eval setup based on criteria
        cut_idx = None
        if cut_eval_setup == "ctxt":
            cut_idx = next((i for i, part in enumerate(parts) if re.match(r"ctxt\d+", part)), None)
        elif cut_eval_setup == "bef_res":
            cut_idx = next((i for i, part in enumerate(parts) if re.match(r"^r\d+$", part)), None)
        elif cut_eval_setup == "alpha":
            cut_idx = next((i for i, part in enumerate(parts) if re.match(r"alpha\d+(\.\d+)?", part)), None)
        elif cut_eval_setup == "ep":
            cut_idx = next((i for i, part in enumerate(parts) if re.match(r"ep\d+", part)), None)
        if cut_idx is not None:
            if cut_eval_setup == "bef_res":
                cut_idx -= 1
            eval_setup = "_".join(parts[1 : cut_idx + 1])
        else:
            if verbose:
                print(f"  Skipping folder {folder_name} as it does not contain cut idx")
            return None

        # Check if this is a ctxt1 setup
        ctxt_part = parts[cut_idx] if cut_idx < len(parts) else ""
        is_ctxt1 = ctxt_part == "ctxt1"

        # Filter out eval_setups not in aliases
        if eval_setup_aliases is not None:
            # Special handling for hist1 folders
            if is_hist1_folder:
                # Use hist1-specific aliases if available
                if eval_setup in hist1_eval_setup_aliases.get(task_name, {}):
                    eval_setup = hist1_eval_setup_aliases[task_name][eval_setup]
                else:
                    if verbose:
                        print(f"  Skipping hist1 eval setup {eval_setup} as it's not in hist1 aliases")
                    return None
            else:
                # For regular folders, use task-specific aliases
                if task_name not in eval_setup_aliases or eval_setup not in eval_setup_aliases[task_name]:
                    if verbose:
                        print(f"  Skipping eval setup {eval_setup} for task {task_name} as it's not in aliases")
                    return None
                eval_setup = eval_setup_aliases[task_name][eval_setup]

        if task_name not in task_subset:
            return None

        task_df = load_task_data(eval_folder)
        if not task_df.empty:
            return (folder_path, task_name, eval_setup, seed), task_df
        elif verbose:
            print(f"No data found in {folder_name}")
        return None

    def process_model_folder(folder_path):
        if verbose:
            print(f"Processing model: {os.path.basename(folder_path)}")

        # Collect folders to process (main folder + seed folders if requested)
        folders_to_process = [(folder_path, "234")]

        if collect_subfolder_seeds:
            # Look for seed subfolders
            seed_folders = []
            for subfolder in os.listdir(folder_path):
                if "seed" in subfolder:
                    seed_path = os.path.join(folder_path, subfolder)
                    seed = subfolder.split("seed")[1]
                    if os.path.isdir(seed_path):
                        seed_folders.append((seed_path, seed))

            if seed_folders and verbose:
                print(f"  Found {len(seed_folders)} seed folders")
            folders_to_process.extend(seed_folders)

        eval_folder_args = []
        # Collect all eval folders to process
        for current_folder, seed in folders_to_process:
            eval_folders_path = os.path.join(current_folder, "simu_env_planning", "online_gc_zeroshot")
            if os.path.exists(eval_folders_path):
                eval_folders = [
                    f for f in os.listdir(eval_folders_path) if os.path.isdir(os.path.join(eval_folders_path, f))
                ]

                for folder_name in eval_folders:
                    eval_folder = os.path.join(eval_folders_path, folder_name)
                    eval_folder_args.append((current_folder, eval_folder, seed))

        return eval_folder_args

    # First, collect all eval folders from all model paths
    all_eval_folders = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for eval_folders in executor.map(process_model_folder, model_paths):
            all_eval_folders.extend(eval_folders)

    # Process all eval folders in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(filter(None, executor.map(process_eval_folder, all_eval_folders)))

    # Build the final task_eval_data dictionary
    for key, df in results:
        task_eval_data[key] = df

    return task_eval_data


def print_task_eval_data_structure(task_eval_data):
    """
    Print a concise summary of the task_eval_data structure and content.

    Args:
        task_eval_data: The data structure returned by collect_task_eval_data
    """
    print("Task Evaluation Data Structure:")
    print(f"Total entries: {len(task_eval_data)}")

    # Count models, tasks, eval setups, seeds
    models = set()
    tasks = set()
    eval_setups = set()
    seeds = set()
    metrics = set()

    for (model_path, task_name, eval_setup, seed), df in task_eval_data.items():
        models.add(model_path)
        tasks.add(task_name)
        eval_setups.add(eval_setup)
        seeds.add(seed)
        metrics.update(df.columns)

    print(f"Models: {len(models)}")
    print(f"Tasks: {len(tasks)} - {sorted(tasks)}")
    print(f"Eval setups: {len(eval_setups)} - {sorted(eval_setups)}")
    print(f"Seeds: {len(seeds)} - {sorted(seeds)}")
    print(f"Metrics: {sorted(metrics)}")

    # Print a few example entries
    print("\nExample entries:")
    count = 0
    for key, df in task_eval_data.items():
        if count >= 3:
            break
        model_path, task_name, eval_setup, seed = key
        model_name = os.path.basename(model_path)
        print(f"  Model: {model_name}, Task: {task_name}, Setup: {eval_setup}, Seed: {seed}")
        print(f"  Data shape: {df.shape}, Epochs: {min(df['epoch'])}..{max(df['epoch'])}")
        metrics_present = [col for col in df.columns if col != "epoch"]
        print(f"  Metrics: {metrics_present}")
        print(
            f"  Last epoch values: {', '.join([f'{m}={df[m].iloc[-1]:.2f}' for m in metrics_present if not pd.isna(df[m].iloc[-1])])}"
        )
        print()
        count += 1


def aggregate_task_data_by_groups(
    task_eval_data,
    design_choices,
    task_groups_mapping,
    last_n_epochs={},
    start_from_epoch={},
    use_computed_best_eval_setup=False,
    std_average_group=False,
    filter_best_eval_setup=True,
):
    """
    Aggregate task data for each task group and design choice.
    Find the best performing eval setup for each task/design choice.

    Args:
        task_eval_data: Data collected using collect_task_eval_data function
        design_choices: Dict mapping design choice labels to lists of model paths
        task_groups_mapping: Dict mapping task names to display names
        last_n_epochs: Dict mapping task names to number of last epochs to aggregate over
        start_from_epoch: Dict mapping task names to starting epoch index (inclusive).
                         If specified for a task, will filter epochs >= start_from_epoch,
                         taking precedence over last_n_epochs for that task.
        filter_best_eval_setup: If True, only use best eval setup per task group.
                                If False, keep all eval setups.

    Returns:
        If filter_best_eval_setup=True:
            dict: {task_group: {design_choice: (mean, std, count)}}
        If filter_best_eval_setup=False:
            dict: {task_group: {design_choice: {eval_setup: (mean, std, count)}}}
    """
    # Create reverse mapping from model paths to design choices
    model_to_design = {}
    for design, model_paths in design_choices.items():
        for path in model_paths:
            model_to_design[path] = design

    # First aggregation: track by (task_group, design_choice, eval_setup)
    eval_setup_data = defaultdict(dict)

    for (model_path, task_name, eval_setup, seed), df in task_eval_data.items():
        # Skip if task not in mapping
        if task_name not in task_groups_mapping:
            continue

        # For handling seed folders
        if model_path not in model_to_design:
            parent_path = os.path.dirname(model_path)
            if parent_path in model_to_design:
                model_to_design[model_path] = model_to_design[parent_path]

        design_choice = model_to_design.get(model_path)
        if design_choice is None:
            print(f"Model path {model_path} not found in design choices mapping, skipping it..")
            continue  # Skip if model doesn't match any design choice

        task_group = task_groups_mapping[task_name]
        key = (task_group, design_choice, eval_setup)

        # Determine which metric to use (use 1-Act_err_xyz for DROID tasks)
        if task_name == "droid":
            if "Act_err_xyz" in df.columns and not df["Act_err_xyz"].isnull().any():
                # For DROID, use 1-Act_err_xyz as the "success rate"
                metric_values = 800 * (0.1 - df["Act_err_xyz"])
                # metric_values = 100 * (1 - df['Act_err_xyz'])
            else:
                print(f"'Act_err_xyz' not found or NaN for {eval_setup} in {model_path}, skipping ..")
                continue
        else:
            # For other tasks, use success rate
            metric_values = df["SR"]

        # Get the last n epochs' data or filter from a starting epoch
        epochs = df["epoch"].values

        # Check if start_from_epoch is specified for this task
        if task_name in start_from_epoch:
            # Filter epochs >= start_from_epoch
            start_epoch = start_from_epoch[task_name]
            mask = epochs >= start_epoch
            filtered_values = metric_values[mask].values
        else:
            # Use the last_n_epochs logic
            task_specific_last_n = last_n_epochs.get(task_name, 10)
            if len(epochs) >= task_specific_last_n:
                filtered_values = metric_values.iloc[-task_specific_last_n:].values
            else:
                filtered_values = metric_values.values

        last_n_values = filtered_values

        # Store all values for this setup
        if key not in eval_setup_data:
            eval_setup_data[key] = last_n_values
        else:
            eval_setup_data[key] = np.concatenate([eval_setup_data[key], last_n_values])

    # Find the best eval setup for each task_group/design_choice
    results = {}
    best_setups = {}

    if not filter_best_eval_setup:
        # Don't filter - keep all eval setups
        # Structure: {task_group: {design_choice: {eval_setup: (mean, std, count)}}}
        for (task_group, design_choice, eval_setup), values in eval_setup_data.items():
            mean_perf = np.mean(values) if len(values) > 0 else 0
            std_perf = np.std(values) if len(values) > 0 else 0
            n_samples = len(values)

            if task_group not in results:
                results[task_group] = {}
            if design_choice not in results[task_group]:
                results[task_group][design_choice] = {}

            results[task_group][design_choice][eval_setup] = (mean_perf, std_perf, n_samples)
    elif use_computed_best_eval_setup:
        for (task_group, design_choice, eval_setup), values in eval_setup_data.items():
            # Calculate mean performance for this eval setup
            mean_perf = np.mean(values) if len(values) > 0 else 0
            std_perf = np.std(values) if len(values) > 0 else 0
            n_samples = len(values)

            # If this is the first or best eval setup we've seen for this group/design, store it
            if task_group not in results:
                results[task_group] = {}
                best_setups[task_group] = {}

            if design_choice not in results[task_group] or mean_perf > results[task_group][design_choice][0]:
                results[task_group][design_choice] = (mean_perf, std_perf, n_samples)
                best_setups[task_group][design_choice] = eval_setup
        print("\nBest eval setups chosen:")
        for task_group in sorted(best_setups.keys()):
            print(f"  {task_group}:")
            for design_choice in sorted(best_setups[task_group].keys()):
                eval_setup = best_setups[task_group][design_choice]
                mean, std, count = results[task_group][design_choice]
                print(f"    {design_choice}: {eval_setup} (performance: {mean:.2f} ± {std:.2f}, n={count})")
    else:
        for (task_group, design_choice, eval_setup), values in eval_setup_data.items():
            if eval_setup != best_eval_setup_per_task_group.get(task_group):
                continue
            # Calculate mean performance for this eval setup
            mean_perf = np.mean(values) if len(values) > 0 else 0
            std_perf = np.std(values) if len(values) > 0 else 0
            n_samples = len(values)
            if task_group not in results:
                results[task_group] = {}
            results[task_group][design_choice] = (mean_perf, std_perf, n_samples)
    print(f"{results=}")
    # Calculate mean across all task groups for each design choice
    mean_data = {}

    if not filter_best_eval_setup:
        # When we have all eval setups, we need to calculate average for each (design, eval_setup) pair
        # Structure: {design_choice: {eval_setup: (mean, std, count)}}
        for design in design_choices.keys():
            design_eval_setups = {}

            # Collect all eval setups for this design across all task groups
            all_setups = set()
            for task_group in results:
                if design in results[task_group]:
                    all_setups.update(results[task_group][design].keys())

            # For each eval setup, calculate average across task groups
            for eval_setup in all_setups:
                values = []
                counts = 0
                for task_group in results:
                    if design in results[task_group] and eval_setup in results[task_group][design]:
                        mean, _, count = results[task_group][design][eval_setup]
                        if count > 0:
                            values.append(mean)
                            counts += count

                if values:
                    mean_value = np.mean(values)
                    if std_average_group:
                        combined_variance = sum(
                            results[task_group][design][eval_setup][1] ** 2
                            for task_group in results
                            if design in results[task_group] and eval_setup in results[task_group][design]
                        )
                        std_value = np.sqrt(combined_variance)
                    else:
                        std_value = 0
                    design_eval_setups[eval_setup] = (mean_value, std_value, counts)

            if design_eval_setups:
                mean_data[design] = design_eval_setups
    else:
        # Original behavior when filter_best_eval_setup=True
        for design in design_choices.keys():
            values = []
            counts = 0
            for task_group in results:
                if design in results[task_group]:
                    mean, _, count = results[task_group][design]
                    if count > 0:  # Only include values where we have data
                        values.append(mean)
                        counts += count

            if values:
                mean_value = np.mean(values)
                if std_average_group:
                    # std_value = np.std(values) / np.sqrt(len(values)) if len(values) > 1 else 0
                    combined_variance = sum(results[task_group][design_choice][1] ** 2 for task_group in results)
                    std_value = np.sqrt(combined_variance)
                else:
                    std_value = 0
                mean_data[design] = (mean_value, std_value, counts)
            else:
                mean_data[design] = (0, 0, 0)

    # Add mean data to aggregated_data
    results["Avg"] = mean_data
    return results


def aggregate_task_data_by_eval_setup(
    task_eval_data,
    eval_setup_aliases,
    task_groups_mapping,
    last_n_epochs={},
    start_from_epoch={},
    std_average_group=False,
):
    """
    Aggregate task data for each task group and eval setup.

    Args:
        task_eval_data: Data collected using collect_task_eval_data function
        eval_setup_aliases: Dict mapping task names to eval setup aliases
        task_groups_mapping: Dict mapping task names to display names
        last_n_epochs: Dict mapping task names (or task groups) to number of last epochs to aggregate over
        start_from_epoch: Dict mapping task names (or task groups) to starting epoch index (inclusive).
                         If specified for a task, will filter epochs >= start_from_epoch,
                         taking precedence over last_n_epochs for that task.

    Returns:
        dict: {task_group: {eval_setup: (mean, std, count)}}
    """
    # First aggregation: track by (task_group, eval_setup)
    from collections import defaultdict

    import numpy as np

    setup_data = defaultdict(list)

    for (model_path, task_name, eval_setup, seed), df in task_eval_data.items():
        # Skip if task not in mapping
        if task_name not in task_groups_mapping:
            continue

        task_group = task_groups_mapping[task_name]
        unified_eval_setup = unif_eval_setup_aliases_across_tasks.get(task_group, {}).get(eval_setup)
        if not unified_eval_setup:
            print(f"Skipping eval setup {eval_setup} for task group {task_group} as it's not in unified aliases")
            continue

        key = (task_group, unified_eval_setup) if unified_eval_setup else (task_group, eval_setup)
        # key = (task_group, eval_setup)

        if task_name == "droid":
            if "Act_err_xyz" in df.columns and not df["Act_err_xyz"].isnull().any():
                # For DROID, use 1-Act_err_xyz as the "success rate"
                metric_values = 800 * (0.1 - df["Act_err_xyz"])
                # metric_values = 100 * (1 - df['Act_err_xyz'])
            else:
                print(f"'Act_err_xyz' not found or NaN for {eval_setup} in {model_path}, skipping ..")
                continue
        else:
            # For other tasks, use success rate
            metric_values = df["SR"]

        # Get the last n epochs' data or filter from a starting epoch
        epochs = df["epoch"].values

        # Check if start_from_epoch is specified for this task_name or task_group
        start_epoch = start_from_epoch.get(task_name) or start_from_epoch.get(task_group)

        if start_epoch is not None:
            # Filter epochs >= start_from_epoch
            mask = epochs >= start_epoch
            filtered_values = metric_values[mask]
        else:
            # Use the last_n_epochs logic
            n_epochs = last_n_epochs.get(task_group, 10) if isinstance(last_n_epochs, dict) else last_n_epochs
            filtered_values = metric_values[-n_epochs:] if len(metric_values) >= n_epochs else metric_values

        setup_data[key].extend(filtered_values)

    # Calculate mean and std for each task_group/eval_setup
    results = {}

    for (task_group, eval_setup), values in setup_data.items():
        if task_group not in results:
            results[task_group] = {}

        mean_perf = np.mean(values) if len(values) > 0 else 0
        std_perf = np.std(values) if len(values) > 1 else 0
        # / np.sqrt(len(values)) if len(values) > 1 else 0
        n_samples = len(values)
        results[task_group][eval_setup] = (mean_perf, std_perf, n_samples)

    # Calculate mean across all task groups for each eval setup
    mean_data = {}
    all_eval_setups = set()

    # Collect all eval setups across all task groups
    for task_data in results.values():
        all_eval_setups.update(task_data.keys())

    # Calculate average performance for each eval setup
    for eval_setup in all_eval_setups:
        values = []
        counts = 0
        for task_group in results:
            if eval_setup in results[task_group]:
                mean, _, count = results[task_group][eval_setup]
                if count > 0:  # Only include values where we have data
                    values.append(mean)
                    counts += count

        if values:
            mean_value = np.mean(values)
            if std_average_group:
                # std_value = np.std(values) / np.sqrt(len(values)) if len(values) > 1 else 0
                combined_variance = sum(
                    results[task_group][eval_setup][1] ** 2
                    for task_group in results
                    if eval_setup in results[task_group]
                )
                std_value = np.sqrt(combined_variance)
            else:
                std_value = 0
            mean_data[eval_setup] = (mean_value, std_value, counts)
        else:
            mean_data[eval_setup] = (0, 0, 0)

    # Add mean data to aggregated_data
    results["Avg"] = mean_data
    return results


def plot_design_choices_grouped_bar(
    task_eval_data,
    design_choices,
    task_groups_mapping,
    last_n_epochs={},
    start_from_epoch={},
    figsize=(6, 4),
    dpi=300,
    color_palette="tab10",
    save_path=None,
    design_choices_eval_setup=False,
    use_computed_best_eval_setup=False,
    std_average_group=False,
    bar_label=False,
):
    """
    Create a grouped bar chart comparing different design choices across task groups.

    Args:
        task_eval_data: Data collected using collect_task_eval_data function
        design_choices: Dict mapping design choice labels to lists of model paths
        task_groups_mapping: Dict mapping task names to display names
        last_n_epochs: Dict, key=task_group: value=Number of last epochs to aggregate over
        figsize: Figure size
        color_palette: Matplotlib color palette to use
        save_path: Optional path to save the figure
        dpi: DPI for saved figure

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    sns.set_theme()
    # Aggregate data by task groups and design choices
    if design_choices_eval_setup:
        aggregated_data = aggregate_task_data_by_eval_setup(
            task_eval_data,
            eval_setup_aliases,
            task_groups_mapping,
            last_n_epochs=last_n_epochs,
            std_average_group=std_average_group,
        )
        # Extract design choices from the aggregated data
        all_eval_setups = set()
        for task_group, eval_setups in aggregated_data.items():
            all_eval_setups.update(eval_setups.keys())
        # Filter out "Avg" if it's in there
        all_eval_setups.discard("Avg")
        design_names = sorted(list(all_eval_setups))
    else:
        aggregated_data = aggregate_task_data_by_groups(
            task_eval_data,
            design_choices,
            task_groups_mapping,
            last_n_epochs=last_n_epochs,
            start_from_epoch=start_from_epoch,
            use_computed_best_eval_setup=use_computed_best_eval_setup,
            std_average_group=std_average_group,
        )
        design_names = sorted(list(design_choices.keys()))

    task_groups = [group for group in aggregated_data.keys() if group != "Avg"]
    task_groups.append("Avg")  # Add Mean at the end

    # Set up the figure
    fig, ax = plt.subplots(figsize=figsize)

    # Set width of bars
    bar_width = 0.8 / len(design_names)

    # Set up colors
    colors = sns.color_palette(color_palette, len(design_names))

    # Plotting
    for i, design in enumerate(design_names):
        # Calculate position for this set of bars
        x_pos = np.arange(len(task_groups))

        # Extract means and stds for this design choice across all task groups
        means = []
        stds = []
        counts = []

        for task_group in task_groups:
            if design in aggregated_data[task_group]:
                mean, std, count = aggregated_data[task_group][design]
                means.append(mean)
                stds.append(std)
                counts.append(count)
            else:
                means.append(0)
                stds.append(0)
                counts.append(0)

        # Plot bars for this design choice
        offset = (i - len(design_names) / 2 + 0.5) * bar_width
        # Create a modified yerr list that hides error bars for Average group when std_average_group is False
        yerr_values = []
        for j, (s, tg) in enumerate(zip(stds, task_groups)):
            # Only include error bars for Average group if std_average_group is True
            if tg == "Avg" and not std_average_group:
                yerr_values.append(np.nan)  # No error bar for Average when std_average_group is False
            else:
                yerr_values.append(s)

        bars = ax.bar(x_pos + offset, means, bar_width, yerr=yerr_values, capsize=3, label=design, color=colors[i])

        if bar_label:
            for j, (bar, mean, count) in enumerate(zip(bars, means, counts)):
                if count > 0:  # Only label bars with data
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 2,
                        f"{mean:.1f}%\n(n={count})",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        rotation=90,
                    )

    # metric_note = "(using 1-Act_err_xyz for DROID tasks, Success Rate for others)"
    ax.set_ylabel("Performance (%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_xticks(np.arange(len(task_groups)))
    ax.set_xticklabels(task_groups, fontsize=9)

    ax.legend(fontsize=9)

    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return fig


def plot_design_choices_line(
    task_eval_data,
    design_choices,
    task_groups_mapping,
    last_n_epochs=10,
    start_from_epoch={},
    figsize=(6, 4),
    dpi=300,
    color_palette="tab10",
    save_path=None,
    connect_means=True,
    use_computed_best_eval_setup=False,
    std_average_group=False,
    offset_markers=True,
):
    """
    Create a line plot showing performance trends across ordered design choices.

    Args:
        task_eval_data: Data collected using collect_task_eval_data function
        design_choices: Dict mapping design choice labels to lists of model paths
        task_groups_mapping: Dict mapping task names to display names
        last_n_epochs: Dict, key=task_group: value=Number of last epochs to aggregate over
        figsize: Figure size
        color_palette: Matplotlib color palette to use
        save_path: Optional path to save the figure
        dpi: DPI for saved figure
        connect_means: Whether to connect mean points with a line
        sort_designs: Whether to sort design choices (if they are orderable)

    Returns:
        matplotlib.figure.Figure: The created figure
    """

    sns.set_theme()

    # Extract design choices and task groups
    if "design_choice_name" in design_choices.keys():
        design_choice_name = design_choices.pop("design_choice_name")
        design_names = sorted(list(design_choices.keys()))
    else:
        design_names = sorted(list(design_choices.keys()))
        design_choice_name = ""

    # Aggregate data by task groups and design choices
    aggregated_data = aggregate_task_data_by_groups(
        task_eval_data,
        design_choices,
        task_groups_mapping,
        last_n_epochs,
        start_from_epoch=start_from_epoch,
        use_computed_best_eval_setup=use_computed_best_eval_setup,
        std_average_group=std_average_group,
    )

    # Get task groups, ensuring Average is last
    task_groups = [group for group in aggregated_data.keys() if group != "Avg"]

    # Set up the figure
    fig, ax = plt.subplots(figsize=figsize)

    # Set up colors
    colors = sns.color_palette(color_palette, len(task_groups) + 1)  # +1 for Average
    markers = ["o", "s", "^", "v", "D", "P", "X", "*", "h", "p", "<", ">", "8"]

    # Create x-axis positions for the design choices
    x_pos = np.arange(len(design_names))

    # Calculate horizontal offsets to distribute markers
    if offset_markers:
        # Only create two offset positions regardless of group count
        offsets = [-0.05, 0.05]
        # Map each task group to either the first or second offset position
        group_to_offset_idx = {group: i % 2 for i, group in enumerate(task_groups)}
        # Add Average group if present
        if "Avg" in aggregated_data:
            group_to_offset_idx["Avg"] = 0  # Or whichever side you prefer
    else:
        offsets = [0]
        group_to_offset_idx = {group: 0 for group in task_groups + (["Avg"] if "Avg" in aggregated_data else [])}

    # Plot individual task groups
    for i, task_group in enumerate(task_groups):
        means = []
        stds = []

        for design in design_names:
            if design in aggregated_data[task_group]:
                mean, std, _ = aggregated_data[task_group][design]
                means.append(mean)
                stds.append(std)
            else:
                means.append(0)
                stds.append(0)

        marker = markers[i % len(markers)]
        x_pos_offset = x_pos + offsets[group_to_offset_idx[task_group]]

        ax.errorbar(
            x_pos_offset, means, yerr=stds, fmt=marker, capsize=3, label=task_group, color=colors[i], alpha=0.4
        )

    if "Avg" in aggregated_data:
        means = []
        stds = []

        for design in design_names:
            if design in aggregated_data["Avg"]:
                mean, std, _ = aggregated_data["Avg"][design]
                means.append(mean)
                stds.append(std)
            else:
                means.append(0)
                stds.append(0)

        ax.errorbar(
            x_pos,
            means,
            yerr=stds,
            fmt="o-" if connect_means else "o",
            capsize=3,
            label="Avg",
            color="black",
            linewidth=2.0 if connect_means else 1,
            markersize=8,
            alpha=0.6,
        )

    ax.set_ylabel("Performance (%)", fontsize=10)
    ax.set_xlabel(design_choice_name, fontsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(design_names, fontsize=9)
    ax.set_ylim(0, 100)

    ax.grid(axis="y", linestyle="--", alpha=0.7)

    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return fig


def generate_latex_table(
    task_eval_data,
    design_choices,
    task_groups_mapping,
    last_n_epochs={},
    start_from_epoch={},
    save_path=None,
    design_choices_eval_setup=False,
    use_computed_best_eval_setup=False,
    std_average_group=False,
    latex_average_column=False,
):
    """
    Generate a LaTeX table comparing different design choices across task groups.

    Args:
        task_eval_data: Data collected using collect_task_eval_data function
        design_choices: Dict mapping design choice labels to lists of model paths
        task_groups_mapping: Dict mapping task names to display names
        last_n_epochs: Dict, key=task_group: value=Number of last epochs to aggregate over
        save_path: Optional path to save the LaTeX table
        design_choices_eval_setup: Whether to use eval setups as design choices
        use_computed_best_eval_setup: Whether to use computed best eval setup
        std_average_group: Whether to compute std for the average group

    Returns:
        str: The LaTeX table
    """
    # Aggregate data similar to the plot_design_choices_grouped_bar function
    if design_choices_eval_setup:
        aggregated_data = aggregate_task_data_by_eval_setup(
            task_eval_data,
            eval_setup_aliases,
            task_groups_mapping,
            last_n_epochs=last_n_epochs,
            std_average_group=std_average_group,
        )

        all_eval_setups = set()
        for task_group, eval_setups in aggregated_data.items():
            all_eval_setups.update(eval_setups.keys())

        all_eval_setups.discard("Avg")
        design_names = list(all_eval_setups)
    else:
        aggregated_data = aggregate_task_data_by_groups(
            task_eval_data,
            design_choices,
            task_groups_mapping,
            last_n_epochs=last_n_epochs,
            start_from_epoch=start_from_epoch,
            use_computed_best_eval_setup=use_computed_best_eval_setup,
            std_average_group=std_average_group,
            filter_best_eval_setup=True,  # Keep all eval setups
        )
        design_names = list(design_choices.keys())

    # Use hardcoded order for task groups
    task_groups = [group for group in TASK_GROUP_ORDER if group in aggregated_data]
    if latex_average_column:
        task_groups.append("Average")  # Add Average at the end

    # Find the best and second-best design choices for each task group
    best_designs = {}
    second_best_designs = {}

    for task_group in task_groups:
        performances = []
        for design in design_names:
            if design in aggregated_data[task_group]:
                mean, _, count = aggregated_data[task_group][design]
                if count > 0:  # Only consider designs with data
                    performances.append((design, mean))

        # Sort by performance (highest first)
        performances.sort(key=lambda x: x[1], reverse=True)

        # Store best and second best if available
        if len(performances) > 0:
            best_designs[task_group] = performances[0][0]
            if len(performances) > 1:
                second_best_designs[task_group] = performances[1][0]

    # Clean task group names - remove line breaks
    clean_task_groups = []
    for group in task_groups:
        if group == "MW-\nReach":
            clean_task_groups.append("MW-R")
        elif group == "MW-\nReach-\nWall":
            clean_task_groups.append("MW-RW")
        else:
            clean_task_groups.append(group)

    # Start building the LaTeX table
    latex_table = "\\begin{table}[t]\n"
    latex_table += "\\centering\n"
    # latex_table += "\\vspace{-2em}\n"
    latex_table += "\\caption{Comparison of our best model to DINO-WM and V-JEPA-2-AC. MW-R and MW-RW denote the Reach and Reach-Wall tasks of Metaworld. Rc-Pl and RC-R denote the Place and Reach tasks of Robocasa. Best model is in bold.}\n"
    latex_table += "\\label{tab:final_model_comp_baselines}\n"

    # Create table header
    latex_table += "\\resizebox{\\textwidth}{!}{\n"
    latex_table += "\\begin{tabular}{l" + "c" * len(task_groups) + "}\n"
    latex_table += "\\toprule\n"
    latex_table += "Model & " + " & ".join(clean_task_groups) + " \\\\\n"
    latex_table += "\\midrule\n"

    # Add rows for each design choice
    for design in design_names:
        row = f"{design} & "

        cells = []
        for task_group in task_groups:
            if design in aggregated_data[task_group]:
                mean, std, count = aggregated_data[task_group][design]
                if count > 0:  # Only display non-zero values
                    # Format the cell with mean and std
                    if task_group == "Average":
                        cell = f"{mean:.1f}"
                    else:
                        cell = f"{mean:.1f} ({std:.1f})"

                    # Apply formatting for best and second best
                    if design == best_designs.get(task_group):
                        cell = f"\\textbf{{{cell}}}"
                else:
                    cell = "—"  # em dash for no data
            else:
                cell = "—"  # em dash for no data

            cells.append(cell)

        row += " & ".join(cells) + " \\\\\n"
        latex_table += row

    # Finish the table
    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "}\n"  # End resizebox
    latex_table += "\\vspace{-1em}\n"
    latex_table += "\\end{table}\n"

    # Save to file if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            f.write(latex_table)
        print(f"LaTeX table saved to: {save_path}")

    return latex_table


def generate_latex_table_all(
    task_eval_data,
    design_choices,
    task_groups_mapping,
    last_n_epochs={},
    start_from_epoch={},
    save_path=None,
    use_computed_best_eval_setup=False,
    std_average_group=False,
    latex_average_column=False,
    planners_to_include=None,
):
    """
    Generate a LaTeX table comparing all planners for each model across task groups.
    This creates a table grouped by planner, with rows for each model under each planner.

    Args:
        task_eval_data: Data collected using collect_task_eval_data function
        design_choices: Dict mapping design choice labels to lists of model paths
        task_groups_mapping: Dict mapping task names to display names
        last_n_epochs: Dict, key=task_group: value=Number of last epochs to aggregate over
        save_path: Optional path to save the LaTeX table
        use_computed_best_eval_setup: Whether to use computed best eval setup (unused here, kept for compatibility)
        std_average_group: Whether to compute std for the average group
        latex_average_column: Whether to include the Average column in the LaTeX table
        planners_to_include: Optional list of planner names to include (after unification). If None, use all.

    Returns:
        str: The LaTeX table
    """
    # Aggregate data without filtering by best eval setup
    aggregated_data = aggregate_task_data_by_groups(
        task_eval_data,
        design_choices,
        task_groups_mapping,
        last_n_epochs=last_n_epochs,
        start_from_epoch=start_from_epoch,
        use_computed_best_eval_setup=False,
        std_average_group=std_average_group,
        filter_best_eval_setup=False,  # Keep all eval setups
    )

    design_names = list(design_choices.keys())
    # Use hardcoded order for task groups
    task_groups = [group for group in TASK_GROUP_ORDER if group in aggregated_data]
    if latex_average_column:
        task_groups.append("Avg")  # Add Average at the end

    # Apply unified eval setup aliases to map different setups to common names
    # Only include eval setups that are in the unified aliases mapping
    unified_data = {}
    for task_group in task_groups:
        unified_data[task_group] = {}
        for design_choice in design_names:
            if design_choice in aggregated_data[task_group]:
                unified_data[task_group][design_choice] = {}
                for eval_setup, stats in aggregated_data[task_group][design_choice].items():
                    # Map to unified name - skip if not in unified aliases
                    unified_name = unif_eval_setup_aliases_across_tasks.get(task_group, {}).get(eval_setup)
                    if not unified_name:
                        print(
                            f"Skipping eval setup {eval_setup} for task group {task_group} as it's not in unified aliases"
                        )
                        continue

                    # If multiple eval setups map to the same unified name, keep the best one
                    if unified_name in unified_data[task_group][design_choice]:
                        existing_mean = unified_data[task_group][design_choice][unified_name][0]
                        new_mean = stats[0]
                        if new_mean > existing_mean:
                            unified_data[task_group][design_choice][unified_name] = stats
                    else:
                        unified_data[task_group][design_choice][unified_name] = stats

    # Collect all unique unified eval setups
    all_eval_setups = set()
    for task_group in task_groups:
        for design_choice in design_names:
            if design_choice in unified_data[task_group]:
                all_eval_setups.update(unified_data[task_group][design_choice].keys())

    # Filter to only include specified planners if provided
    if planners_to_include is not None:
        eval_setups = [es for es in sorted(list(all_eval_setups)) if es in planners_to_include]
    else:
        eval_setups = sorted(list(all_eval_setups))

    # Clean task group names - remove line breaks
    clean_task_groups = []
    for group in task_groups:
        if group == "MW-\nReach":
            clean_task_groups.append("MW-R")
        elif group == "MW-\nReach-\nWall":
            clean_task_groups.append("MW-RW")
        else:
            clean_task_groups.append(group)

    # Start building the LaTeX table
    latex_table = "\\begin{table}[t]\n"
    latex_table += "\\centering\n"
    # latex_table += "\\vspace{-2em}\n"
    latex_table += "\\caption{Comparison of different models across all planner configurations. "
    latex_table += "MW-R and MW-RW denote the Reach and Reach-Wall tasks of Metaworld. "
    latex_table += "Rc-Pl and RC-R denote the Place and Reach tasks of Robocasa.}\n"
    latex_table += "\\label{tab:all_planners_model_comp}\n"

    # Create table header
    latex_table += "\\resizebox{\\textwidth}{!}{\n"
    latex_table += "\\begin{tabular}{ll" + "c" * len(task_groups) + "}\n"
    latex_table += "\\toprule\n"
    latex_table += "Model & Planner & " + " & ".join(clean_task_groups) + " \\\\\n"
    latex_table += "\\midrule\n"

    # Track the best performance for each (planner, task_group) combination
    best_performances = {}
    for eval_setup in eval_setups:
        for task_group in task_groups:
            best_mean = -1
            for design_choice in design_names:
                if design_choice in unified_data[task_group]:
                    if eval_setup in unified_data[task_group][design_choice]:
                        mean, _, count = unified_data[task_group][design_choice][eval_setup]
                        if count > 0 and mean > best_mean:
                            best_mean = mean
            if best_mean >= 0:
                best_performances[(eval_setup, task_group)] = best_mean

    # Reorganize table: group by planner first, then show all models
    for planner_idx, eval_setup in enumerate(eval_setups):
        for model_idx, design in enumerate(design_names):
            # For the first model of each planner, include the planner name
            if model_idx == 0:
                row = f"{eval_setup} & {design} & "
            else:
                row = f" & {design} & "

            cells = []
            for task_group in task_groups:
                if design in unified_data[task_group] and eval_setup in unified_data[task_group][design]:
                    mean, std, count = unified_data[task_group][design][eval_setup]
                    if count > 0:
                        # Format the cell with mean and std
                        if task_group == "Avg":
                            cell = f"{mean:.1f}"
                        else:
                            cell = f"{mean:.1f} ({std:.1f})"

                        # Bold if this is the best performance for this (planner, task_group)
                        best_mean = best_performances.get((eval_setup, task_group), -1)
                        if abs(mean - best_mean) < 0.01:  # Close enough to be considered best
                            cell = f"\\textbf{{{cell}}}"
                    else:
                        cell = "—"
                else:
                    cell = "—"

                cells.append(cell)

            row += " & ".join(cells) + " \\\\\n"
            latex_table += row

        # Add a line separator after each planner (except the last one)
        if planner_idx < len(eval_setups) - 1:
            latex_table += "\\midrule\n"

    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "}\n"  # End resizebox
    latex_table += "\\vspace{-1em}\n"
    latex_table += "\\end{table}\n"

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            f.write(latex_table)
        print(f"LaTeX table (all planners) saved to: {save_path}")

    return latex_table


def main():
    """
    Final paper figures commands:
        python app/plan_common/local/logs_plan_joint_per_design_choice.py --design_choices_file app/plan_common/local/design_choice_yamls/W.yaml --output W_comparison --plot_line --verbose
        python app/plan_common/local/logs_plan_joint_per_design_choice.py --design_choices_file app/plan_common/local/design_choice_yamls/rollout_steps.yaml --output rollout_steps_comparison --plot_line --verbose
        python app/plan_common/local/logs_plan_joint_per_design_choice.py --design_choices_file app/plan_common/local/design_choice_yamls/model_size.yaml --output model_size_comparison --plot_line --verbose
        python app/plan_common/local/logs_plan_joint_per_design_choice.py --design_choices_file app/plan_common/local/design_choice_yamls/plan_setup.yaml --output plan_setup --verbose --design_choices_eval_setup
        python app/plan_common/local/logs_plan_joint_per_design_choice.py --design_choices_file app/plan_common/local/design_choice_yamls/enc.yaml --output enc_comparison --verbose
        python app/plan_common/local/logs_plan_joint_per_design_choice.py --design_choices_file app/plan_common/local/design_choice_yamls/pred_arch.yaml --output pred_arch_comparison --verbose
        python app/plan_common/local/logs_plan_joint_per_design_choice.py --design_choices_file app/plan_common/local/design_choice_yamls/prop.yaml --output prop_comparison --verbose
        python app/plan_common/local/logs_plan_joint_per_design_choice.py --design_choices_file app/plan_common/local/design_choice_yamls/final_baseline_comp.yaml --output final_baseline_comp --generate_latex --verbose

        Test:
        python app/plan_common/local/logs_plan_joint_per_design_choice.py --design_choices_file app/plan_common/local/design_choice_yamls/test_n_epochs.yaml --output test_n_epochs --generate_latex --verbose
        python app/plan_common/local/logs_plan_joint_per_design_choice.py --design_choices_file app/plan_common/local/design_choice_yamls/test_n_epochs.yaml --output test_n_epochs --generate_latex --verbose --cut_eval_setup ep
        Rebuttal: Comment out GD planners from unif_eval_setup_aliases_across_tasks and run below to get fixed planner table
        python app/plan_common/local/logs_plan_joint_per_design_choice.py --design_choices_file app/plan_common/local/design_choice_yamls/final_baseline_comp.yaml --output final_baseline_comp_all_planners --generate_latex_all --verbose
        Compare planners with average across DWM-S and other models without GD (as in ICLR submission): Comment out GD planners from unif_eval_setup_aliases_across_tasks and run:
        python app/plan_common/local/logs_plan_joint_per_design_choice.py --design_choices_file app/plan_common/local/design_choice_yamls/plan_setup_all.yaml --output plan_setup_all --verbose --design_choices_eval_setup
        Rebuttal: Additional compare of planners with proprio
        python app/plan_common/local/logs_plan_joint_per_design_choice.py --design_choices_file app/plan_common/local/design_choice_yamls/plan_setup_prop.yaml --output plan_setup_prop --verbose --design_choices_eval_setup
        Rebuttal: new revised final table:
        python app/plan_common/local/logs_plan_joint_per_design_choice.py --design_choices_file app/plan_common/local/design_choice_yamls/final_baseline_comp.yaml --output final_baseline_comp_revised --generate_latex --verbose
        new comparison of ftcond to seqcond:
        python app/plan_common/local/logs_plan_joint_per_design_choice.py --design_choices_file app/plan_common/local/design_choice_yamls/ft_seq_cond.yaml --output ft_seq_cond_comparison --verbose

    In the ICRL submission, we used the last_n_epochs logic. During rebuttal, we introduced the start_from_epoch logic.
    This allows to avoid having to evaluate 100 checkpoints for DROID models, but rather a subset of the epochs in [215, 315].
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Generate grouped bar chart comparing design choices across task groups"
    )
    parser.add_argument(
        "--design_choices_file", type=str, required=True, help="Path to JSON file containing design choices mapping"
    )
    parser.add_argument("--output", type=str, default=f"comparison.pdf", help="Output file path for the plot")
    parser.add_argument(
        "--design_choices_eval_setup",
        action="store_true",
        help="Use eval setups as design choices instead of model paths",
    )
    parser.add_argument("--plot_line", action="store_true", help="Plot line chart instead of grouped bar chart")
    parser.add_argument("--figsize", type=str, default="6,4", help="Figure size as width,height")
    parser.add_argument(
        "--use_computed_best_eval_setup",
        action="store_true",
        help="Use computed best eval setup instead of predefined ones",
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved figure")
    parser.add_argument("--verbose", action="store_true", help="Print verbose information")
    parser.add_argument(
        "--std_average_group", action="store_true", help="Whether to compute std for the average group"
    )
    parser.add_argument("--bar_label", action="store_true", help="Whether to add bar labels")
    parser.add_argument("--generate_latex", action="store_true", help="Generate LaTeX table instead of plot")
    parser.add_argument(
        "--latex_average_column", action="store_true", help="Whether to include the Average column in the LaTeX table"
    )
    parser.add_argument("--cut_eval_setup", type=str, default="ctxt", help="Where to cut eval_setup")
    parser.add_argument(
        "--generate_latex_all", action="store_true", help="Generate LaTeX table comparing all planners for each model"
    )
    parser.add_argument(
        "--planners_to_include",
        type=str,
        default=None,
        help='Comma-separated list of planner names to include (after unification). Example: "CEM $L_2$,NG $L_2$"',
    )
    args = parser.parse_args()

    hist1_folders = [
        "CHECKPOINT_ROOT/mw_sweep/mw_4f_fsk5_ask1_r224_pred_dino_wm_depth6_noprop_repro_1roll_hist1_save",
        "CHECKPOINT_ROOT/mz_sweep/mz_4f_fsk5_ask1_r224_pred_dino_wm_depth6_noprop_repro_1roll_hist1_save_2n",
        "CHECKPOINT_ROOT/pt_sweep/pt_4f_fsk5_ask1_r224_pred_dino_wm_depth6_noprop_repro_1roll_hist1_save",
        "CHECKPOINT_ROOT/wall_sweep/wall_4f_fsk5_ask1_r224_pred_dino_wm_depth6_noprop_repro_1roll_hist1_save_2n",
        "CHECKPOINT_ROOT/droid_final_sweep/droid_4f_fps4_r224_pred_dino_wm_depth6_noprop_repro_1roll_2fpcs_2n",
    ]

    if args.design_choices_eval_setup:
        args.use_computed_best_eval_setup = False

    # Parse figsize
    figsize = tuple(map(float, args.figsize.split(",")))

    from ruamel.yaml import YAML

    yaml = YAML(typ="safe")
    with open(args.design_choices_file, "r") as f:
        design_choices = yaml.load(f)
    design_choices = expand_env_vars(design_choices)

    save_file = Path(f"{base_dir}/{args.output}.pdf")
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    # Define task subset from task_groups_mapping
    task_subset = list(task_groups_mapping.keys())

    # Flatten model paths for data collection
    model_paths = []
    # Check if design_choices has a design_choice_name key
    design_choice_name = None
    if "design_choice_name" in design_choices:
        design_choice_name = design_choices.get("design_choice_name")
        # Only iterate through the actual design choices (not the name)
        for key, paths in design_choices.items():
            if key != "design_choice_name":
                model_paths.extend(paths)
    else:
        # Original behavior if no design_choice_name is present
        for paths in design_choices.values():
            model_paths.extend(paths)

    if args.verbose:
        print(f"Processing {len(model_paths)} model paths across {len(design_choices)} design choices")
        print(f"Looking for {len(task_subset)} task types")

    # Collect the data
    with Timer("Data Collection"):
        task_eval_data = collect_task_eval_data(
            model_paths,
            task_subset=task_subset,
            eval_setup_aliases=eval_setup_aliases,
            collect_subfolder_seeds=True,
            verbose=args.verbose,
            exclude_eval_folders=exclude_eval_folders,
            hist1_folders=hist1_folders,
            cut_eval_setup=args.cut_eval_setup,
        )

    if args.verbose:
        if design_choice_name:
            print(f"Design choice parameter: {design_choice_name}")
        print(
            f"Processing {len(model_paths)} model paths across {len(design_choices) - (1 if design_choice_name else 0)} design choices"
        )
        print_task_eval_data_structure(task_eval_data)

    # Plot the grouped bar chart
    with Timer("Plotting"):
        if args.plot_line:
            fig = plot_design_choices_line(
                task_eval_data,
                design_choices,
                task_groups_mapping,
                last_n_epochs=last_n_epochs,
                start_from_epoch=start_from_epoch,
                figsize=figsize,
                save_path=save_file,
                dpi=args.dpi,
                connect_means=True,
                use_computed_best_eval_setup=args.use_computed_best_eval_setup,
                std_average_group=args.std_average_group,
            )
        elif args.generate_latex_all:
            # Parse planners_to_include if provided
            planners_list = None
            if args.planners_to_include:
                planners_list = [p.strip() for p in args.planners_to_include.split(",")]

            latex_table = generate_latex_table_all(
                task_eval_data,
                design_choices,
                task_groups_mapping,
                last_n_epochs=last_n_epochs,
                start_from_epoch=start_from_epoch,
                save_path=save_file.with_suffix(".tex"),
                use_computed_best_eval_setup=args.use_computed_best_eval_setup,
                std_average_group=args.std_average_group,
                latex_average_column=args.latex_average_column,
                planners_to_include=planners_list,
            )
        elif args.generate_latex:
            latex_table = generate_latex_table(
                task_eval_data,
                design_choices,
                task_groups_mapping,
                last_n_epochs=last_n_epochs,
                start_from_epoch=start_from_epoch,
                save_path=save_file.with_suffix(".tex"),
                design_choices_eval_setup=args.design_choices_eval_setup,
                use_computed_best_eval_setup=args.use_computed_best_eval_setup,
                std_average_group=args.std_average_group,
                latex_average_column=args.latex_average_column,
            )
        else:
            fig = plot_design_choices_grouped_bar(
                task_eval_data,
                design_choices,
                task_groups_mapping,
                last_n_epochs=last_n_epochs,
                start_from_epoch=start_from_epoch,
                figsize=figsize,
                save_path=save_file,
                dpi=args.dpi,
                design_choices_eval_setup=args.design_choices_eval_setup,
                use_computed_best_eval_setup=args.use_computed_best_eval_setup,
                std_average_group=args.std_average_group,
                bar_label=args.bar_label,
            )


if __name__ == "__main__":
    main()
