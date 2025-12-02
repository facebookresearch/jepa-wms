from src.utils.cluster import dataset_paths
from src.utils.logging import get_logger

logger = get_logger("Datasets utils")


def get_dataset_paths(datasets: list[str]):
    paths = []
    for d in datasets:
        try:
            path = dataset_paths().get(d)
        except Exception:
            raise Exception(f"Unknown dataset: {d}")
        paths.append(path)
    logger.info(f"Datapaths {paths}")
    return paths
