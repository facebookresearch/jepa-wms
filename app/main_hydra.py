# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# ============================================================================
# USAGE EXAMPLES:
# ============================================================================
#
# Single seed:
#   python -m app.main_hydra -c configs/vjepa_wm/droid_final_sweep/droid_8fpcs_fps4_r256_vj2acvitgL1_repro_2roll_noprop_4n_asp1_wsd_seed1.yaml \
#       account=fair_amaia_cw_explore partition=learn qos=explore
#
# Different seed:
#   python -m app.main_hydra -c configs/vjepa_wm/droid_final_sweep/droid_8fpcs_fps4_r256_vj2acvitgL1_repro_2roll_noprop_4n_asp1_wsd_seed1.yaml \
#       account=fair_amaia_cw_explore partition=learn qos=explore \
#       meta.seed=2 'folder=${CHECKPOINT_ROOT}/droid_final_sweep/droid_8fpcs_r256_vj2acvitgL1_repro_2roll_noprop_4n_asp1_wsd/seed2'
#
# Multiple seeds (multirun):
#   python -m app.main_hydra -c configs/vjepa_wm/droid_final_sweep/droid_8fpcs_fps4_r256_vj2acvitgL1_repro_2roll_noprop_4n_asp1_wsd_seed1.yaml \
#       account=fair_amaia_cw_explore partition=learn qos=explore \
#       -m meta.seed=1,2,1000 'folder=${CHECKPOINT_ROOT}/droid_final_sweep/droid_8fpcs_r256_vj2acvitgL1_repro_2roll_noprop_4n_asp1_wsd/seed${meta.seed}'
#
# ============================================================================

import atexit
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import yaml
from omegaconf import DictConfig, OmegaConf

# Short aliases for hydra.launcher options
SLURM_ALIASES = {
    "account": "hydra.launcher.account",
    "partition": "hydra.launcher.partition",
    "qos": "hydra.launcher.qos",
    "nodes": "hydra.launcher.nodes",
    "time": "hydra.launcher.timeout_min",
    "exclude": "hydra.launcher.exclude",
    "mem": "hydra.launcher.mem_gb",
}


def convert_env_vars_to_omegaconf(content: str) -> str:
    """
    Convert ${VAR_NAME} to ${oc.env:VAR_NAME} for environment variables.
    This allows configs to use the simple ${VAR} syntax while still working with OmegaConf.

    Only converts UPPERCASE variable names (which are typically env vars).
    Leaves ${lowercase} or ${nested.key} as-is for OmegaConf config interpolation.
    """
    # Pattern matches ${UPPERCASE_VAR} but not ${oc.env:VAR} (already converted)
    # and not ${lowercase} or ${nested.key} (config interpolation)
    pattern = r'\$\{([A-Z][A-Z0-9_]*)\}'

    def replace(match):
        var_name = match.group(1)
        return f'${{oc.env:{var_name}}}'

    return re.sub(pattern, replace, content)


def get_repo_root():
    """Get the repository root directory."""
    return Path(__file__).parent.parent.absolute()


def preprocess_config_file(config_path: str, cli_overrides: dict = None) -> str:
    """
    Read a config file and convert env var syntax for OmegaConf compatibility.
    Also injects Hydra searchpath, launcher defaults, and propagates top-level
    SLURM settings (nodes, tasks_per_node, etc.) to the launcher.

    Args:
        config_path: Path to the config file
        cli_overrides: Dict of CLI overrides (e.g., {"folder": "${CHECKPOINT_ROOT}/..."})

    Returns path to a temporary file with converted content.
    """
    repo_root = get_repo_root()
    hydra_config_dir = str(repo_root / "configs" / "hydra")

    with open(config_path, 'r') as f:
        content = f.read()

    converted_content = convert_env_vars_to_omegaconf(content)

    # Parse config to extract top-level SLURM settings
    parsed_config = yaml.safe_load(converted_content)
    nodes = parsed_config.get('nodes', 1)
    tasks_per_node = parsed_config.get('tasks_per_node', 8)
    cpus_per_task = parsed_config.get('cpus_per_task', 16)
    mem_per_gpu = parsed_config.get('mem_per_gpu', '210GB')

    # Get folder - CLI override takes precedence over config file
    folder_raw = parsed_config.get('folder', '')
    if cli_overrides and 'folder' in cli_overrides:
        folder_raw = cli_overrides['folder']

    # Convert ${VAR} to ${oc.env:VAR} in folder path
    folder_omegaconf = convert_env_vars_to_omegaconf(folder_raw)

    # Extract job name from folder (last component)
    job_name = folder_raw.split('/')[-1] if folder_raw else 'hydra_job'
    # If job name contains ${...}, use a placeholder
    if '${' in job_name:
        job_name = 'hydra_job'

    # Inject Hydra defaults at the top of the config if not present
    if 'defaults:' not in converted_content:
        hydra_defaults = f"""defaults:
  - override hydra/launcher: submitit_slurm

hydra:
  searchpath:
    - file://{hydra_config_dir}
  launcher:
    nodes: {nodes}
    tasks_per_node: {tasks_per_node}
    cpus_per_task: {cpus_per_task}
    gpus_per_node: {tasks_per_node}
    mem_per_gpu: {mem_per_gpu}
    submitit_folder: {folder_omegaconf}/submitit-job
  job:
    name: {job_name}

"""
        converted_content = hydra_defaults + converted_content

    # Write to temp file in the same directory (to preserve relative paths)
    config_dir = os.path.dirname(config_path) or "."
    temp_fd, temp_path = tempfile.mkstemp(suffix='.yaml', dir=config_dir)
    with os.fdopen(temp_fd, 'w') as f:
        f.write(converted_content)

    return temp_path


def preprocess_args():
    """
    Preprocess command line arguments:
    1. Parse --config/-c and convert to Hydra config_path/config_name
    2. Preprocess config file to convert ${VAR} to ${oc.env:VAR}
    3. Expand short aliases (account=, partition=, qos=) to full hydra.launcher paths
    4. Convert -m to --multirun
    5. Convert ${VAR} to ${oc.env:VAR} in CLI overrides
    """
    new_argv = [sys.argv[0]]
    config_path = None
    config_name = None
    temp_config_path = None
    config_file = None
    cli_overrides = {}

    # First pass: collect config file and CLI overrides
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in ("--config", "-c") and i + 1 < len(sys.argv):
            config_file = sys.argv[i + 1]
            i += 2
        elif "=" in arg and not arg.startswith("-"):
            key, value = arg.split("=", 1)
            cli_overrides[key] = value
            i += 1
        else:
            i += 1

    # Second pass: build new argv with preprocessing
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]

        # Handle --config / -c
        if arg in ("--config", "-c"):
            if i + 1 < len(sys.argv):
                # Preprocess the config file with CLI overrides
                temp_config_path = preprocess_config_file(config_file, cli_overrides)

                path = Path(temp_config_path)
                config_name = path.stem
                config_path = os.path.join("..", str(path.parent))
                i += 2
                continue

        # Handle -m as alias for --multirun
        elif arg == "-m":
            new_argv.append("--multirun")
            i += 1
            continue

        # Handle short aliases and convert env vars in CLI overrides
        elif "=" in arg and not arg.startswith("-"):
            key, value = arg.split("=", 1)

            # Convert ${VAR} to ${oc.env:VAR} in the value
            value = convert_env_vars_to_omegaconf(value)

            if key in SLURM_ALIASES:
                new_argv.append(f"+{SLURM_ALIASES[key]}={value}")
            else:
                new_argv.append(f"{key}={value}")
            i += 1
            continue

        new_argv.append(arg)
        i += 1

    sys.argv = new_argv
    return config_path or "../configs/hydra", config_name or "defaults", temp_config_path


# Preprocess args before importing hydra
_config_path, _config_name, _temp_config_path = preprocess_args()

import hydra
from app.scaffold import main as app_main
from src.utils.logging import get_logger, git_information

logger = get_logger(force=True)


# Clean up temp config file on exit
def cleanup_temp_config():
    if _temp_config_path and os.path.exists(_temp_config_path):
        try:
            os.remove(_temp_config_path)
        except Exception:
            pass

atexit.register(cleanup_temp_config)


def copy_code_folder(code_folder, ignore_patterns, ignore_paths):
    """Copy code folder to experiment directory for reproducibility."""
    path_to_node_folder = {}

    for path in ignore_paths:
        split_path = path.split("/")
        base_path = "/".join(split_path[:-1])
        node_folder = split_path[-1]
        path_to_node_folder[base_path] = node_folder

    def ignore_func(path, names):
        ignore_list = list(ignore_patterns)
        if path in path_to_node_folder.keys():
            ignore_list.append(path_to_node_folder[path])
        return ignore_list

    if not os.path.exists(code_folder):
        original_cwd = hydra.utils.get_original_cwd()
        shutil.copytree(original_cwd, code_folder, ignore=ignore_func)


def setup_experiment_folder(cfg: DictConfig):
    """Set up experiment folder with code copy and git info."""
    folder = cfg.folder
    Path(folder).mkdir(parents=True, exist_ok=True)

    code_folder = os.path.join(folder, "code")
    ignore_patterns = ["__pycache__", ".vscode", ".git", "core", ".venv", "local"]
    ignore_paths = [
        "./evals/ava/alphaction/data",
        "./demos",
        "./traces",
        "./configs/local",
        "./configs/*/cwtemp",
    ]
    copy_code_folder(code_folder, ignore_patterns, ignore_paths)

    params_path = os.path.join(folder, "params-pretrain.yaml")
    if not os.path.exists(params_path):
        with open(params_path, "w") as f:
            yaml.dump(OmegaConf.to_container(cfg, resolve=True), f)

    git_info_fpath = os.path.join(folder, "git-info.txt")
    with open(git_info_fpath, "w") as f:
        f.write(git_information())

    return code_folder


def is_slurm_job():
    """Check if we're running inside a SLURM job (on a compute node)."""
    return os.environ.get("SLURM_JOB_ID") is not None


@hydra.main(version_base=None, config_path=_config_path, config_name=_config_name)
def main(cfg: DictConfig, resume_preempt=False):
    """
    Hydra-based training entry point.

    When using the submitit launcher with --multirun:
    - The launcher pickles this function and submits it to SLURM
    - On the compute node, SLURM_JOB_ID is set and we run training
    - On the login node (no SLURM_JOB_ID), we should not run training

    Note: Distributed initialization is handled by the app's train.py,
    not here. This matches the pattern in main_distributed.py.
    """
    # Guard: Only run training code if we're inside a SLURM job
    # This prevents CUDA errors on login nodes where submitit is preparing the job
    if not is_slurm_job():
        logger.warning(
            "SLURM_JOB_ID not found. If you're using --multirun with submitit, "
            "the job will be submitted to SLURM. If you want to run locally, "
            "use the 'basic' launcher instead of 'submitit_slurm'."
        )
        return

    # Resolve OmegaConf interpolations
    OmegaConf.resolve(cfg)

    # Set up experiment folder (like main_distributed.py does)
    setup_experiment_folder(cfg)

    # Convert to regular dict for app_main
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Log config
    logger.info("Loaded config:")
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(cfg_dict)

    # Launch app - distributed init is handled inside app's train.py
    app_main(cfg_dict["app"], args=cfg_dict, resume_preempt=resume_preempt)


def print_submitted_job_ids(submitit_folder: str):
    """Print job IDs from submitit folder (folders named with numeric job IDs)."""
    if not os.path.exists(submitit_folder):
        return
    job_ids = [d for d in os.listdir(submitit_folder)
               if os.path.isdir(os.path.join(submitit_folder, d)) and d.isdigit()]
    if job_ids:
        for job_id in sorted(job_ids, key=int, reverse=True)[:5]:
            print(f"Submitted job: {job_id}")


if __name__ == "__main__":
    main()

    # Print job IDs if on login node (after submitit submits jobs)
    if not is_slurm_job():
        for submitit_dir in Path("multirun").glob("**/.submitit"):
            print_submitted_job_ids(str(submitit_dir))
            break  # Only check most recent
