import glob
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

# ALIASES
from app.vjepa_wm.local.aliases import (
    eval_setup_aliases,
    hist1_eval_setup_aliases,
)

# Import local configuration from macros.py (gitignored)
# Run `python setup_macros.py` to generate it from your environment variables
from macros import CHECKPOINT_ROOT, JEPA_WM_HOME

# Constants and mapping dictionaries
task_data_mapping = {
    "SR": "Success Rate (%)",
    "Reward": "Reward",
    "Act_err": "Action error",
    "Act_err_xyz": "Action error XYZ",
    "Act_err_orient": "Action error Orientation",
    "Act_err_closure": "Action error Closure",
    "Total_LPIPS": "Total LPIPS",
    "Total_Emb_L2": "Total Embedding L2",
}
metrics_map = {
    "ep_end_dist_xyz": "Act_err_xyz",
    "ep_end_dist_orientation": "Act_err_orient",
    "ep_end_dist_closure": "Act_err_closure",
    "ep_total_lpips": "Total_LPIPS",
    "ep_total_emb_l2": "Total_Emb_L2",
}

task_cut_eval_setup_mapping = {
    "droid": "ctxt",
    # "droid": "ep",
    "pt": "ctxt",
    "mz": "ctxt",
    "wall": "ctxt",
    "mw-reach": "ctxt",
    "mw-reach-wall": "ctxt",
    "rcasa-reach": "ctxt",
    "rcasa-pick": "ctxt",
    "rcasa-place": "ctxt",
    "rcasa-reach-pick": "ctxt",
    "rcasa-pick-place": "ctxt",
    "rcasa-reach-pick-place": "ctxt",
}


class Timer:
    def __init__(self, name):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        print(f"{self.name} took {elapsed:.2f} seconds")


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


def load_csv(file, **kwargs):
    lines = []
    with open(file) as f:
        lines_ = f.readlines()
        lines.append(lines_[0])
        for line in lines_[1:]:
            if line[0].isnumeric():
                lines.append(line)

    return pd.read_csv(StringIO("".join(lines)), **kwargs)


def abbreviate_number(x, pos):
    if x >= 1e6:
        return f"{x*1e-6:.1f}M"
    elif x >= 1e3:
        return f"{x*1e-3:.1f}K"
    else:
        return str(int(x))


def exponential_moving_average(data, alpha=0.3):
    """Smooth the data using an exponential moving average."""
    smoothed = np.zeros_like(data, dtype=float)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


@lru_cache(maxsize=128)
def load_task_data(folder):
    """Load task data from a folder with caching for performance."""
    task_data = {"epoch": [], "SR": [], "Reward": [], "Act_err": []}
    optional_metrics = {
        "Act_err_xyz": False,
        "Act_err_orient": False,
        "Act_err_closure": False,
        "Total_LPIPS": False,
        "Total_Emb_L2": False,
    }

    # Find all epoch folders and sort them
    epoch_folders = sorted(
        [f for f in glob.glob(os.path.join(folder, "epoch-*")) if os.path.basename(f).split("-")[-1].isdigit()],
        key=lambda x: int(os.path.basename(x).split("-")[-1]),
    )

    for epoch_folder in epoch_folders:
        try:
            epoch = int("".join(filter(str.isdigit, os.path.basename(epoch_folder))))
            eval_file_path = os.path.join(epoch_folder, "eval.csv")

            if os.path.exists(eval_file_path):
                try:
                    task_df = pd.read_csv(eval_file_path)
                    # Add required metrics
                    task_data["epoch"].append(epoch)
                    task_data["SR"].append(task_df["episode_success"].values[-1] * 100)
                    task_data["Reward"].append(task_df["episode_reward"].values[-1])
                    task_data["Act_err"].append(task_df["ep_end_dist"].values[-1])

                    for src_col, dest_col in metrics_map.items():
                        if src_col in task_df.columns:
                            if dest_col not in task_data:
                                task_data[dest_col] = []
                            task_data[dest_col].append(task_df[src_col].values[-1])
                            optional_metrics[dest_col] = True
                except Exception as e:
                    print(f"Error reading {eval_file_path}: {e}")
        except ValueError:
            print(f"Ignoring non-standard epoch folder: {os.path.basename(epoch_folder)}")

    # Fill missing values for optional metrics
    for metric, present in optional_metrics.items():
        if present and len(task_data[metric]) < len(task_data["epoch"]):
            # Fill with NaN for any missing epochs
            task_data[metric].extend([np.nan] * (len(task_data["epoch"]) - len(task_data[metric])))

    return pd.DataFrame(task_data)


def collect_task_eval_data(
    model_training_folders,
    task_subset,
    eval_setup_aliases=None,
    exclude_eval_folders=None,
    collect_subfolder_seeds=False,
    cut_eval_setup="ctxt",
    hist1_folders=[],
    verbose=True,
    max_workers=None,
):
    """
    Collect evaluation data from model training folders with parallel processing.

    Args:
        model_training_folders: List of (model_path, label) tuples to process
        task_subset: List of task names to include
        eval_setup_aliases: Dictionary mapping raw eval setup names to display names
        exclude_eval_folders: List of folder names to exclude
        collect_subfolder_seeds: Whether to look for seed subfolders
        cut_eval_setup: Where to cut eval setup
        hist1_folders: List of folder patterns identifying hist1 folders
        verbose: Whether to print detailed progress
        max_workers: Maximum number of worker threads

    Returns:
        Dictionary with (task_name, eval_setup) keys and metric data values.
        Each metric contains a list of (epochs, values, model_label, seed) tuples.
    """
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 4) + 4)

    with Timer("Collecting task evaluation data"):
        task_eval_data = {}
        all_metrics = set(["SR", "Reward", "Act_err"])  # Required metrics
        all_folders_to_process = []

        # First, collect all folders to process (main folder + seed folders if requested)
        for folder_path, label in model_training_folders:
            folders_to_process = [(folder_path, "234", label)]

            if collect_subfolder_seeds:
                # Look for seed subfolders
                seed_folders = []
                for subfolder in os.listdir(folder_path):
                    if "seed" in subfolder:
                        seed_path = os.path.join(folder_path, subfolder)
                        seed = subfolder.split("seed")[1]
                        if os.path.isdir(seed_path):
                            seed_folders.append((seed_path, seed, label))

                if seed_folders and verbose:
                    print(f"Found {len(seed_folders)} seed folders for {label}")
                folders_to_process.extend(seed_folders)

            # Get all evaluation folders for each model folder
            for current_folder, seed, current_label in folders_to_process:
                eval_folders_path = os.path.join(current_folder, "simu_env_planning", "online_gc_zeroshot")
                if os.path.exists(eval_folders_path):
                    eval_folders = [f for f in glob.glob(os.path.join(eval_folders_path, "*")) if os.path.isdir(f)]
                    for folder in eval_folders:
                        all_folders_to_process.append((folder, seed, current_label))

        # Define folder processing function
        def process_folder(args):
            folder, seed, label = args
            folder_name = os.path.basename(folder)

            if exclude_eval_folders and folder_name in exclude_eval_folders:
                return None

            parts = folder_name.split("_")
            task_name = clean_task_name(parts[0], folder)

            # Check which model folder this eval folder belongs to
            parent_model_path = None
            for model_path, _ in model_training_folders:
                if folder.startswith(model_path):
                    parent_model_path = model_path
                    break

            # Determine if this is a hist1 folder
            is_hist1_folder = any(hist1 in (parent_model_path or "") for hist1 in hist1_folders)

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

            task_df = load_task_data(folder)
            if task_df.empty:
                if verbose:
                    print(f"No data found in {folder_name}")
                return None

            return task_name, eval_setup, task_df, label, seed

        # Process all folders in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(filter(None, executor.map(process_folder, all_folders_to_process)))

            # First collect all metrics
            for task_name, eval_setup, task_df, _, _ in results:
                optional_columns = [
                    "Act_err_xyz",
                    "Act_err_orient",
                    "Act_err_closure",
                    "Total_LPIPS",
                    "Total_Emb_L2",
                ]
                all_metrics.update([col for col in optional_columns if col in task_df.columns])

            # Build the task_eval_data structure from results
            for task_name, eval_setup, task_df, label, seed in results:
                key = (task_name, eval_setup)
                if key not in task_eval_data:
                    task_eval_data[key] = {metric: [] for metric in all_metrics}

                # Add data for each metric
                for metric in all_metrics:
                    if metric in task_df.columns:
                        task_eval_data[key][metric].append((task_df["epoch"], task_df[metric], label, seed))
                    else:
                        # Add empty data for metrics not in this DataFrame
                        task_eval_data[key][metric].append(
                            (task_df["epoch"], pd.Series([np.nan] * len(task_df)), label, seed)
                        )

        return task_eval_data


def plot_data(
    task_name,
    eval_setup,
    data,
    y_value,
    smooth,
    alpha,
    show_original,
    base_dir,
    put_title=True,
    eval_setup_aliases=None,
    truncate_epoch=None,
    y_min=None,
    y_max=None,
    xtick_step=4,
    y_log_scale=None,
    x_log_scale=None,
    average_seeds=False,
):
    """Plot the data for a given task and evaluation setup."""
    with Timer(f"Plotting {task_name}/{eval_setup}/{y_value}"):
        sns.set_theme()
        plt.figure(figsize=(6, 4), dpi=300)

        if not data:
            print(f"No data to plot for {task_name}/{eval_setup}/{y_value}")
            plt.close()
            return

        trunc_data = []
        for epochs, values, label, *args in data:
            seed = args[0] if args else "234"  # Default seed if not provided

            # Convert to numpy arrays for easier handling
            epochs_arr = np.array(epochs.tolist() if isinstance(epochs, pd.Series) else epochs)
            values_arr = np.array(values.tolist() if isinstance(values, pd.Series) else values)

            # Check for valid values
            valid_mask = ~np.isnan(values_arr) & ~np.isinf(values_arr)
            if not np.any(valid_mask):
                print(f"Skipping {label} for {task_name}/{eval_setup}/{y_value} - all values are NaN/Inf")
                continue

            # Truncate data if requested
            if truncate_epoch is not None:
                mask = epochs_arr <= truncate_epoch
                trunc_epochs = epochs_arr[mask].tolist()
                trunc_values = values_arr[mask].tolist()
            else:
                trunc_epochs = epochs_arr.tolist()
                trunc_values = values_arr.tolist()

            if trunc_epochs and trunc_values:  # Make sure we have data
                trunc_data.append((trunc_epochs, trunc_values, label, seed))

        # Check if we have any data to plot after filtering
        if not trunc_data:
            print(f"No valid data to plot for {task_name}/{eval_setup}/{y_value} after filtering")
            plt.close()
            return

        max_epoch = max(max(epochs) for epochs, _, _, _ in trunc_data) if trunc_data else 0

        # Group trunc_data by label if averaging seeds
        if average_seeds:
            trunc_data_grouped = {}
            for epochs, values, label, seed in trunc_data:
                if label not in trunc_data_grouped:
                    trunc_data_grouped[label] = []
                trunc_data_grouped[label].append((epochs, values, seed))

            plot_data = []
            for label, data_list in trunc_data_grouped.items():
                # Calculate mean over seeds for each label
                # Find the common epochs across all seeds
                common_epochs = set(data_list[0][0])
                for epochs, _, _ in data_list[1:]:
                    common_epochs &= set(epochs)
                common_epochs = sorted(list(common_epochs))

                # Interpolate values at common epochs
                interpolated_values = []
                for epochs, values, _ in data_list:
                    interpolated_values.append(np.interp(common_epochs, epochs, values))

                # Calculate mean of interpolated values
                mean_values = np.mean(interpolated_values, axis=0)

                plot_data.append((common_epochs, mean_values, label))
        else:
            # Just use all data points individually
            plot_data = [(epochs, values, label) for epochs, values, label, _ in trunc_data]

        # Plot the data
        for epochs, values, label in plot_data:
            if smooth:
                smoothed_values = exponential_moving_average(values, alpha=alpha)
                sns.lineplot(x=epochs, y=smoothed_values, label=f"{label}")
                if show_original:
                    plt.fill_between(epochs, values, smoothed_values, alpha=0.15)
            else:
                sns.lineplot(x=epochs, y=values, label=label)

        # Set axis limits
        plt.xlim(0, truncate_epoch if truncate_epoch else max_epoch)

        # Handle y-axis log scale and limits
        if isinstance(y_log_scale, bool):
            y_log_scale = 10 if y_log_scale else None
        if y_log_scale is not None and y_log_scale > 0:
            plt.yscale("log", base=y_log_scale)
            ax = plt.gca()
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))  # Scientific notation formatter
            all_values = [val for _, values, _ in plot_data for val in values if np.isfinite(val)]
            if all_values:
                min_val = max(min(all_values), 1e-10)  # Avoid zero for log scale
                plt.ylim(bottom=min_val)

        # Set explicit y limits if provided
        if y_min is not None and y_max is not None:
            plt.ylim(y_min, y_max)

        # Handle x-axis log scale
        if isinstance(x_log_scale, bool):
            x_log_scale = 10 if x_log_scale else None
        if x_log_scale is not None and x_log_scale > 0:
            plt.xscale("log", base=x_log_scale)
            ax = plt.gca()
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))  # Scientific notation formatter
            current_xlim = plt.xlim()
            if current_xlim[0] < 1:
                plt.xlim(left=1)

        # Formatting
        if eval_setup_aliases is not None and isinstance(eval_setup_aliases, dict):
            eval_setup = eval_setup_aliases.get(eval_setup, eval_setup)
        if put_title:
            plt.title(f"{task_name} {eval_setup} {y_value}")
        plt.xlabel("Epoch")
        plt.ylabel(f"{y_value}")
        plt.legend(fontsize="small", framealpha=0.5)

        if not x_log_scale:
            plt.xticks(range(0, max_epoch + 2, xtick_step))

        plt.tight_layout()
        pdf_path = os.path.join(base_dir, f"{task_name}_{eval_setup}_{y_value.lower()}_evolution.pdf")
        plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
        print(f"Saved plot to {pdf_path}")
        plt.close()


def plot_task_eval_data(
    task_eval_data,
    y_values,
    smooth,
    alpha,
    show_original,
    base_dir,
    put_title=True,
    eval_setup_aliases=None,
    truncate_epoch=None,
    y_min=None,
    y_max=None,
    xtick_step=4,
    y_log_scale=None,
    x_log_scale=None,
    average_seeds=False,
):
    """Plot evaluation data for tasks."""
    with Timer(f"Plotting all tasks"):
        for (task_name, eval_setup), data in task_eval_data.items():
            for key in y_values:
                if key in data:
                    plot_data(
                        task_name,
                        eval_setup,
                        data[key],
                        task_data_mapping.get(key, key),
                        smooth,
                        alpha,
                        show_original,
                        base_dir,
                        put_title=put_title,
                        eval_setup_aliases=eval_setup_aliases,
                        truncate_epoch=truncate_epoch,
                        y_min=y_min,
                        y_max=y_max,
                        xtick_step=xtick_step,
                        y_log_scale=y_log_scale,
                        x_log_scale=x_log_scale,
                        average_seeds=average_seeds,
                    )
                else:
                    print(f"Warning: Metric {key} not found for {task_name}/{eval_setup}")


def main():
    # ==============================================
    # Paths are constructed using environment variables from macros.py
    jepa_dir = os.path.join(JEPA_WM_HOME, "jepa-wms")
    local_plan_common_dir = os.path.join(jepa_dir, "app/plan_common/local")

    base_dir = os.path.join(local_plan_common_dir, "paper-app")
    base_dir = os.path.join(base_dir, "mw_sweep")
    os.makedirs(base_dir, exist_ok=True)

    model_training_folders = [
        # Central run
        (
            os.path.join(CHECKPOINT_ROOT, "mw_sweep/mw_4f_fsk5_ask1_r224_pred_dino_wm_depth6_noprop_repro_1roll_save"),
            "WM",
        ),
        (
            os.path.join(
                CHECKPOINT_ROOT, "mw_sweep/mw_4f_fsk5_ask1_r224_pred_dino_wm_depth6_noprop_repro_1roll_save_hist7"
            ),
            r"$\text{WM}_W$",
        ),
        (
            os.path.join(CHECKPOINT_ROOT, "mw_sweep/mw_4f_fsk5_ask1_r224_pred_dino_wm_depth6_repro_1roll_save"),
            "WM-prop",
        ),
        (
            os.path.join(CHECKPOINT_ROOT, "mw_sweep/mw_4f_fsk5_ask1_r224_pred_dino_wm_depth6_noprop_repro_2roll_save"),
            "WM-2-step",
        ),
        (
            os.path.join(CHECKPOINT_ROOT, "mw_sweep/mw_4f_fsk5_ask1_r224_pred_dino_wm_depth6_noprop_repro_3roll_save"),
            "WM-3-step",
        ),
        (
            os.path.join(
                CHECKPOINT_ROOT, "mw_sweep/mw_4f_fsk5_ask1_r224_pred_dino_wm_depth6_noprop_repro_6roll_hist7_bs4_save"
            ),
            r"$\text{WM}_W$-6-step",
        ),
        (
            os.path.join(
                CHECKPOINT_ROOT, "mw_sweep/mw_4f_fsk5_ask1_r224_pred_dino_wm_dinovitb_depth6_noprop_repro_1roll_save"
            ),
            "WM-B",
        ),
        (
            os.path.join(
                CHECKPOINT_ROOT, "mw_sweep/mw_4f_fsk5_ask1_r224_pred_dino_wm_dinovitl_depth6_noprop_repro_1roll_save"
            ),
            "WM-L",
        ),
    ]

    task_eval_data = collect_task_eval_data(
        model_training_folders,
        task_subset,
        eval_setup_aliases=eval_setup_aliases,
    )

    plot_task_eval_data(
        task_eval_data,
        y_values=["SR"],
        smooth=True,
        alpha=0.2,
        show_original=True,
        base_dir=base_dir,
        put_title=True,
        eval_setup_aliases=eval_setup_aliases,
        truncate_epoch=50,
        y_min=0,
        y_max=90,
        average_seeds=True,
    )
    # ==============================================

    # ALL DATASETS
    task_subset = [
        "droid",
        "reach-pick-place",
        "reach-pick",
        "pick-place",
        "rcasa-reach",
        "pick",
        "rcasa-place",
        "pt",
        "wall",
        "mz",
    ]

    # Collect evaluation data
    task_eval_data = collect_task_eval_data(
        model_training_folders=model_training_folders,
        task_subset=task_subset,
        exclude_eval_folders=exclude_folders,
        collect_subfolder_seeds=False,
        hist1_folders=hist1_folders,
        verbose=True,
        eval_setup_aliases=eval_setup_aliases,
    )

    # DROID
    plot_task_eval_data(
        task_eval_data=task_eval_data,
        y_values=["Act_err_xyz"],
        smooth=True,
        alpha=0.2,
        show_original=True,
        base_dir=base_dir,
        truncate_epoch=316,
        y_min=0.0,
        y_max=1.0,
        xtick_step=20,
        y_log_scale=3,
        x_log_scale=False,
        average_seeds=True,
        eval_setup_aliases=eval_setup_aliases,
    )
    # OTHER DATASETS, Success Rate
    # plot_task_eval_data(
    #     task_eval_data,
    #     y_values=["SR"],
    #     smooth=True,
    #     alpha=0.02,
    #     show_original=True,
    #     base_dir=base_dir,
    #     put_title=True,
    #     truncate_epoch=315, # 50 in general
    #     y_min=0,
    #     y_max=60,
    #     xtick_step=20,
    #     average_seeds=True,
    #     eval_setup_aliases=eval_setup_aliases,
    # )

    print(f"Plots saved to {base_dir}")


if __name__ == "__main__":
    main()
