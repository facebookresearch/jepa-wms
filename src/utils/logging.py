# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import os
import subprocess
import sys
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Hashable

import torch

logger = logging.getLogger(__name__)


def gpu_timer(closure, log_timings=True):
    """Helper to time gpu-time to execute closure()"""
    log_timings = log_timings and torch.cuda.is_available()

    elapsed_time = -1.0
    if log_timings:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    result = closure()

    if log_timings:
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)

    return result, elapsed_time


LOG_FORMAT = "[%(levelname)-8s][%(asctime)s][%(name)-20s][%(funcName)-25s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name=None, force=False):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT, force=force)
    return logging.getLogger(name=name)


class MetricsLogger(ABC):
    @abstractmethod
    def log(self) -> None: ...  # noqa: E704


class CSVLogger(MetricsLogger):

    def __init__(self, fname, *argv, **kwargs):
        self.fname = fname
        self.fields = []
        self.types = []
        self.delim = kwargs.get("delim", ",")
        for i, v in enumerate(argv, 1):
            self.types.append(v[0])
            self.fields.append(v[1])
        # -- print header
        header = self.delim.join(self.fields)
        with open(self.fname, "+a") as f:
            print(header, file=f)

    def log(self, *argv, **kwargs):
        # kwargs enable use as a Chained logger (along with Otel, wandb, etc.)
        if len(kwargs) > 0:
            if len(argv) > 0:
                raise ValueError("Send either a list of values or a 'metrics' dict, not both!")
            # Populate the CSV fields from the kwargs. 'epoch' and 'iteration' are input separately
            metrics = kwargs["metrics"]
            metrics["epoch"] = kwargs["epoch"]
            metrics["iteration"] = kwargs["iteration"]
            metrics["itr"] = kwargs["iteration"]  # TODO: replace 'itr' with 'iteration' in all calls
            data = [metrics[k] for k in self.fields]
        else:
            # argv is the simple, legacy interface
            data = argv
        line = self.delim.join([formatter % value for formatter, value in zip(self.types, data)])
        with open(self.fname, "+a") as f:
            print(line, file=f)


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = float("-inf")
        self.min = float("inf")
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


@torch.no_grad
def grad_logger(named_params):
    stats = AverageMeter()
    stats.first_layer = None
    stats.last_layer = None
    selected_params = []
    first_layer_id = last_layer_id = None
    i = 0
    for n, p in named_params:
        if (p.grad is not None) and not (n.endswith(".bias") or len(p.shape) == 1):
            selected_params.append(p.grad)
            if "qkv" in n:
                last_layer_id = i
                if first_layer_id is None:
                    first_layer_id = i
            i += 1
    norms = torch.stack(torch._foreach_norm(selected_params, 2.0))
    mean_norm = norms.mean()
    stats.update(mean_norm.item())  # TODO: investigate how to remove this sync point
    if last_layer_id is None or first_layer_id is None:
        stats.first_layer = stats.last_layer = 0.0
    else:
        stats.first_layer = norms[first_layer_id].item()
        stats.last_layer = norms[last_layer_id].item()
    return stats


class ModelGradLogger:
    def __init__(self, model):
        selected_params = []
        first_layer_id = last_layer_id = None
        i = 0
        for n, p in model.named_parameters():
            # if (p.grad is not None) and not (n.endswith(".bias") or len(p.shape) == 1):
            if not (n.endswith(".bias") or len(p.shape) == 1):
                selected_params.append(p)
                if "qkv" in n:
                    last_layer_id = i
                    if first_layer_id is None:
                        first_layer_id = i
                i += 1
        self.selected_params = selected_params
        self.first_layer_id = first_layer_id
        self.last_layer_id = last_layer_id

    def get_grad_stats(self):
        selected_params = []
        for p in self.selected_params:
            if p.grad is not None:
                selected_params.append(p.grad)

        norms = torch.stack(torch._foreach_norm(selected_params, 2.0))
        mean_norm = norms.mean()

        first_layer_id = last_layer_id = 0  # TODO: fix it

        stats = AverageMeter()
        stats.first_layer = None
        stats.last_layer = None
        stats.update(mean_norm.item())  # TODO: fix
        if last_layer_id is None or first_layer_id is None:
            stats.first_layer = stats.last_layer = 0.0
        else:
            stats.first_layer = norms[first_layer_id].item()
            stats.last_layer = norms[last_layer_id].item()
        return stats


def adamw_logger(optimizer):
    """logging magintude of first and second momentum buffers in adamw"""
    # TODO: assert that optimizer is instance of torch.optim.AdamW
    state = optimizer.state_dict().get("state")
    exp_avg_stats = AverageMeter()
    exp_avg_sq_stats = AverageMeter()
    exp_avg_t = []
    exp_avg_sq_t = []
    numels = []
    for key in state:
        s = state.get(key)
        exp_avg_t.append(s.get("exp_avg"))
        exp_avg_sq_t.append(s.get("exp_avg_sq"))
        numels.append(s.get("exp_avg").numel())
    n1 = torch.stack(torch._foreach_norm(exp_avg_t, 1.0))
    n2 = torch.stack(torch._foreach_norm(exp_avg_sq_t, 1.0))

    # sync point! maybe consider different way by doing global avg instead
    numels = torch.as_tensor(numels, dtype=n1.dtype, device=n1.device)
    n1 /= numels
    n2 /= numels
    exp_avg_stats.update(n1.mean().item())  # TODO: fix
    exp_avg_sq_stats.update(n2.mean().item())  # TODO: fix
    return {"exp_avg": exp_avg_stats, "exp_avg_sq": exp_avg_sq_stats}


def jepa_rootpath():
    this_file = os.path.abspath(__file__)
    return "/".join(this_file.split("/")[:-3])


def git_information():
    jepa_root = jepa_rootpath()
    try:
        resp = (
            subprocess.check_output(["git", "-C", jepa_root, "rev-parse", "HEAD", "--abbrev-ref", "HEAD"])
            .decode("ascii")
            .strip()
        )
        commit, branch = resp.split("\n")
        return f"branch: {branch}\ncommit: {commit}\n"
    except Exception:
        return "unknown"
