from pathlib import Path

from omegaconf import OmegaConf

from evals.simu_env_planning.planning.common import TASK_SET


def parse_cfg(cfg: OmegaConf) -> OmegaConf:
    """
    Parses a Hydra config. Mostly for convenience.
    """
    cfg.work_dir = Path(cfg.work_dir)
    cfg.task_specification.multitask = cfg.task_specification.task in TASK_SET.keys()
    cfg.tasks = TASK_SET.get(cfg.task_specification.task, [cfg.task_specification.task])

    return cfg
