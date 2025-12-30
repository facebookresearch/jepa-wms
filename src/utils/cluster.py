import os

import clusterscope

from src.utils.logging import get_logger

logger = get_logger("Cluster utils")

# Global environment variables for path configuration
# These should be set before running the code
# See README.md for setup instructions
DATASET_ROOT = os.environ.get("DATASET_ROOT", None)
CHECKPOINT_ROOT = os.environ.get("CHECKPOINT_ROOT", None)


def slurm_account_partition_and_qos(low_pri: bool) -> tuple:
    """
    Get SLURM account, partition, and QoS settings from current job environment.

    This function reads SLURM environment variables to inherit settings from the
    current job when launching child jobs (e.g., evaluation jobs from training).

    Any value can be None if not set by your cluster. This is normal - some clusters
    don't use accounts, some don't use QoS, etc.

    For the low-priority QoS (used for eval jobs), set the SLURM_QOS_LOW_PRIORITY
    environment variable to your cluster's low-priority QoS name.

    Args:
        low_pri: If True, use low-priority QoS (from SLURM_QOS_LOW_PRIORITY env var)

    Returns:
        Tuple of (account, partition, qos) - any can be None
    """
    account = os.environ.get("SLURM_JOB_ACCOUNT")
    partition = os.environ.get("SLURM_JOB_PARTITION")
    qos = os.environ.get("SLURM_JOB_QOS")

    if low_pri:
        # Use cluster-specific low-priority QoS from environment variable
        low_pri_qos = os.environ.get("SLURM_QOS_LOW_PRIORITY")
        if low_pri_qos is not None:
            qos = low_pri_qos

    return account, partition, qos


def _build_dataset_paths():
    """
    Build dataset paths using environment variables when available.
    Falls back to None if environment variables are not set.
    This allows users to configure their own dataset locations.
    """
    # Use environment variable or None as fallback
    dataset_root = DATASET_ROOT

    if dataset_root is None:
        logger.warning(
            "DATASET_ROOT environment variable not set. "
            "Dataset paths will need to be provided manually or set the environment variable. "
            "See README.md for setup instructions."
        )
        # Return empty dict - users must provide dataset paths manually
        return {}

    # Build dataset paths relative to the root
    # Users should organize their datasets under DATASET_ROOT
    return {
        "default": {
            # Simulated environments (used in the paper)
            "PushT": f"{dataset_root}/pusht_noise",
            "PointMaze": f"{dataset_root}/point_maze",
            "Wall": f"{dataset_root}/wall_single",
            "METAWORLD_HF": f"{dataset_root}/Metaworld/data",
            "Robocasa": f"{dataset_root}/robocasa/",
            "DROID": f"{dataset_root}/DROID/droid_paths.csv",
            "DROID_slice": f"{dataset_root}/DROID/droid_paths.csv",
            "Franka_hf": f"{dataset_root}/franka_custom",
            # Video datasets
            "K400": f"{dataset_root}/kinetics400/k400_train_paths.csv",
            "K400_val": f"{dataset_root}/kinetics400/k400_val_paths.csv",
            "K710": f"{dataset_root}/kinetics710/k710_train_paths.csv",
            "K710_val": f"{dataset_root}/kinetics710/k710_val_paths.csv",
            "SSv2": f"{dataset_root}/ssv2/ssv2_train_paths.csv",
            "SSv2_val": f"{dataset_root}/ssv2/ssv2_val_paths.csv",
            "HowTo100M": f"{dataset_root}/howto100m/howto100m_paths.csv",
            # Add other datasets as needed
        }
    }


DATASET_PATHS_BY_CLUSTER = _build_dataset_paths()


def get_dataset_path(dataset: str, cluster=None) -> str:
    """
    Get the path for a specific dataset.
    Uses 'default' cluster if environment variables are set, otherwise tries the actual cluster name.
    """
    if cluster is None:
        cluster = clusterscope.cluster()

    # Try 'default' first if DATASET_ROOT is set
    if DATASET_ROOT is not None and "default" in DATASET_PATHS_BY_CLUSTER:
        try:
            return DATASET_PATHS_BY_CLUSTER["default"][dataset]
        except KeyError:
            pass

    # Fallback to cluster-specific path (backward compatibility)
    if cluster in DATASET_PATHS_BY_CLUSTER:
        return DATASET_PATHS_BY_CLUSTER[cluster][dataset]

    raise Exception(
        f"Could not find dataset {dataset}. "
        f"Please set DATASET_ROOT environment variable or add cluster-specific paths. "
        f"See README.md for setup instructions."
    )


def get_dataset_paths(datasets: list[str], is_train: bool = True) -> list[str]:
    """
    Get paths for multiple datasets.
    """
    paths = []
    for dataset in datasets:
        if not is_train:
            dataset = f"{dataset}_val"
        try:
            path = get_dataset_path(dataset)
        except Exception:
            raise Exception(
                f"Could not find dataset {dataset}. "
                f"Please set DATASET_ROOT environment variable. "
                f"See README.md for setup instructions."
            )
        paths.append(path)
    logger.info(f"Datapaths {paths}")
    return paths


def dataset_paths() -> dict[str, str]:
    """
    Get all dataset paths for the current environment.
    Uses 'default' if DATASET_ROOT is set.
    """
    if DATASET_ROOT is not None and "default" in DATASET_PATHS_BY_CLUSTER:
        return DATASET_PATHS_BY_CLUSTER["default"]

    # Fallback to cluster-specific paths
    cluster = clusterscope.cluster()
    if cluster in DATASET_PATHS_BY_CLUSTER:
        return DATASET_PATHS_BY_CLUSTER[cluster]

    logger.warning(
        "No dataset paths configured. "
        "Please set DATASET_ROOT environment variable. "
        "See README.md for setup instructions."
    )
    return {}
