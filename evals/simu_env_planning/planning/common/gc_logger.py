# The below code is inspired from TD-MPC2 https://github.com/nicklashansen/tdmpc2
# licensed under the MIT License

import datetime
import os

import numpy as np
import pandas as pd
from termcolor import colored

from src.utils.logging import get_logger

logger = get_logger(__name__)

CONSOLE_FORMAT = [
    ("episode_reward", "R", "float"),
    ("episode_success", "S", "float"),
    ("total_time", "T", "time"),
]


def make_dir(dir_path):
    """Create directory if it does not already exist."""
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def print_run(cfg):
    """
    Pretty-printing of current run information.
    Logger calls this method at initialization.
    """
    prefix, color, attrs = "  ", "green", ["bold"]

    def _limstr(s, maxlen=36):
        return str(s[:maxlen]) + "..." if len(str(s)) > maxlen else s

    def _pprint(k, v):
        print(
            prefix + colored(f'{k.capitalize()+":":<15}', color, attrs=attrs),
            _limstr(v),
        )

    observations = ", ".join([str(v) for v in cfg.obs_shape.values()])
    kvs = [
        ("observations", observations),
        ("actions", cfg.action_dim),
        ("experiment", cfg.logging.exp_name),
    ]
    w = np.max([len(_limstr(str(kv[1]))) for kv in kvs]) + 25
    div = "-" * w
    print(div)
    for k, v in kvs:
        _pprint(k, v)
    print(div)


class Logger:
    """Primary logging object. Logs locally."""

    def __init__(self, cfg):
        self._log_dir = make_dir(cfg.work_dir)
        self._save_csv = cfg.logging.save_csv
        if cfg.rank == 0:
            print_run(cfg)
        self.cfg = cfg

    def _format(self, key, value, ty):
        if ty == "int":
            return f'{colored(key+":", "blue")} {int(value):,}'
        elif ty == "float":
            return f'{colored(key+":", "blue")} {value:.02f}'
        elif ty == "loss":
            return f'{colored(key+":", "blue")} {value:.04f}'
        elif ty == "time":
            value = str(datetime.timedelta(seconds=int(value)))
            return f'{colored(key+":", "blue")} {value}'
        else:
            raise ValueError(f"invalid log format type: {ty}")

    def _print(self, d):
        category = colored("eval", "green")
        pieces = [f" {category:<14}"]
        for k, disp_k, ty in CONSOLE_FORMAT:
            if k in d:
                pieces.append(f"{self._format(disp_k, d[k], ty):<22}")
        print("   ".join(pieces))

    def pprint_multitask(self, d, cfg):
        """Pretty-print evaluation metrics for multi-task training."""
        print(colored(f"Evaluated agent on {len(cfg.tasks)} tasks:", "yellow", attrs=["bold"]))
        reward = []
        success = []
        for k, v in d.items():
            if "+" not in k:
                continue
            task = k.split("+")[1]
            if k.startswith("episode_reward"):
                reward.append(v)
            elif k.startswith("episode_success"):
                success.append(v)
                print(colored(f"  {task:<22}\tS: {v:.02f}", "yellow"))

    def average_task_metrics(self, d):
        reward = []
        success = []
        ep_expert_succ = []
        ep_succ_dist = []
        ep_end_dist = []
        ep_end_dist_xyz = []
        ep_end_dist_orientation = []
        ep_end_dist_closure = []
        ep_time = []
        ep_total_lpips = []
        ep_total_emb_l2 = []
        for k, v in d.items():
            if "+" not in k:
                continue
            if k.startswith("episode_reward+"):
                reward.append(v)
            elif k.startswith("episode_success+"):
                success.append(v)
            elif k.startswith("ep_expert_succ+"):
                ep_expert_succ.append(v)
            elif k.startswith("ep_succ_dist+"):
                ep_succ_dist.append(v)
            elif k.startswith("ep_end_dist+"):
                ep_end_dist.append(v)
            elif k.startswith("ep_end_dist_xyz+"):
                ep_end_dist_xyz.append(v)
            elif k.startswith("ep_end_dist_orientation+"):
                ep_end_dist_orientation.append(v)
            elif k.startswith("ep_end_dist_closure+"):
                ep_end_dist_closure.append(v)
            elif k.startswith("ep_time+"):
                ep_time.append(v)
            elif k.startswith("ep_total_lpips+"):
                ep_total_lpips.append(v)
            elif k.startswith("ep_total_emb_l2+"):
                ep_total_emb_l2.append(v)
        d["episode_reward"] = np.nanmean(reward)
        d["episode_success"] = np.nanmean(success)
        if ep_expert_succ:
            d["ep_expert_succ"] = np.nanmean(ep_expert_succ)
        if ep_succ_dist:
            d["ep_succ_dist"] = np.nanmean(ep_succ_dist)
        if ep_end_dist:
            d["ep_end_dist"] = np.nanmean(ep_end_dist)
        if ep_end_dist_xyz:
            d["ep_end_dist_xyz"] = np.nanmean(ep_end_dist_xyz)
        if ep_end_dist_orientation:
            d["ep_end_dist_orientation"] = np.nanmean(ep_end_dist_orientation)
        if ep_end_dist_closure:
            d["ep_end_dist_closure"] = np.nanmean(ep_end_dist_closure)
        if ep_time:
            d["ep_time"] = np.nanmean(ep_time)
        if ep_total_lpips:
            d["ep_total_lpips"] = np.nanmean(ep_total_lpips)
        if ep_total_emb_l2:
            d["ep_total_emb_l2"] = np.nanmean(ep_total_emb_l2)
        return d

    def log(self, d, multitask=False):
        d = self.average_task_metrics(d)
        if self._save_csv:
            general_possible_keys = [
                "total_time",
                "episode_reward",
                "episode_success",
                "ep_expert_succ",
                "ep_succ_dist",
                "ep_end_dist",
                "ep_end_dist_xyz",
                "ep_end_dist_orientation",
                "ep_end_dist_closure",
                "ep_time",
                "ep_total_lpips",
                "ep_total_emb_l2",
            ]
            general_keys = [elt for elt in general_possible_keys if elt in d.keys()]
            general_data = [d[elt] for elt in general_keys]
            general_file_path = self._log_dir / "eval.csv"
            file_exists = os.path.isfile(general_file_path)
            pd.DataFrame([general_data]).to_csv(
                general_file_path, mode="a", header=general_keys if not file_exists else False, index=None
            )
            if multitask:
                task_keys = [key for key in d.keys() if key not in general_keys]
                for task in set(key.split("+")[1] for key in task_keys):
                    # filter out task_keys for this task
                    task_specific_keys = [k for k in task_keys if task == k.split("+")[1]]
                    task_data = [d[key] for key in task_specific_keys]
                    task_file_path = self._log_dir / f"eval_{task}.csv"
                    file_exists = os.path.isfile(task_file_path)
                    pd.DataFrame([task_data]).to_csv(
                        task_file_path, mode="a", header=task_specific_keys if not file_exists else False, index=None
                    )
        self._print(d)
