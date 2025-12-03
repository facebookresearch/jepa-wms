# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gc
import logging
import sys
from abc import ABC, abstractmethod

import torch
from torch import nn

from src.utils.adamw import AdamW as RAdamW
from src.utils.logging import AverageMeter, CSVLogger, adamw_logger, get_logger, gpu_timer, grad_logger
from src.utils.schedulers import CosineWDSchedule, WarmupCosineSchedule

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def clean_state_dict(state_dict):
    """Remove 'module.' prefix from state_dict keys."""
    out = {k.replace("module.", ""): v for k, v in state_dict.items()}
    return out


class TrainableModel(ABC):
    def __init__(self, model, train_config=None, device="cpu"):
        """
        model: nn.Module
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.optimizer = None
        self.scaler = None
        self.scheduler = None
        self.wd_scheduler = None
        self.mixed_precision = None
        self.clip_grad = None
        self.use_radamw = None
        self.train_config = train_config

    def init_opt(
        self,
        use_radamw=False,
        betas=(0.9, 0.999),
        eps=1e-8,
        ipe_scale=1.25,
        weight_decay=1e-6,
        final_weight_decay=1e-6,
        final_lr=0.0,
        start_lr=0.0,
        ref_lr=1e-3,
        warmup=2,
        num_epochs=90,
        iterations_per_epoch=1000,
        mixed_precision=True,
        clip_grad=1.0,
    ):
        self.mixed_precision = mixed_precision
        self.clip_grad = clip_grad
        self.use_radamw = use_radamw
        param_groups = []
        param_groups += [
            {"params": (p for n, p in self.model.named_parameters() if ("bias" not in n) and (len(p.shape) != 1))},
        ]
        param_groups += [
            {
                "params": (p for n, p in self.model.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
                "WD_exclude": True,
                "weight_decay": 0,
            },
        ]

        if use_radamw:
            logger.info("Using Rescaled-AdamW")
            optimizer = RAdamW(param_groups, betas=betas, eps=eps)
        else:
            logger.info("Using AdamW")
            optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=float(eps))
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=int(warmup * iterations_per_epoch),
            start_lr=start_lr,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )
        wd_scheduler = CosineWDSchedule(
            optimizer,
            ref_wd=weight_decay,
            final_wd=final_weight_decay,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )
        scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
        self.optimizer = optimizer
        self.scaler = scaler
        self.scheduler = scheduler
        self.wd_scheduler = wd_scheduler
        return optimizer, scaler, scheduler, wd_scheduler

    def load_checkpoint(self, path, load_optimizer=True):
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        epoch = checkpoint["epoch"]

        # -- loading module
        pretrained_dict = clean_state_dict(checkpoint["model"])
        # Filter out decoder_pos_embed to allow loading with different resolutions
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k != "decoder_pos_embed"}
        msg = self.model.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained trainable module from epoch {epoch} with msg: {msg}")
        # -- loading optimizer
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["opt"])
            if self.scaler is not None:
                self.scaler.load_state_dict(checkpoint["scaler"])
            logger.info(f"loaded optimizers from epoch {epoch}")
            logger.info(f"read-path: {path}")

            del checkpoint

        return epoch

    def save_checkpoint(self, epoch, path, rank=0):
        if rank != 0:
            return
        save_dict = {
            "model": self.model.state_dict(),
            "opt": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scaler": None if self.scaler is None else self.scaler.state_dict(),
            "epoch": epoch,
        }
        try:
            torch.save(save_dict, path)
        except Exception as e:
            logger.info(f"Encountered exception when saving checkpoint: {e}")

    def delete_opt(self):
        del self.optimizer
        del self.scaler
        del self.scheduler
        del self.wd_scheduler
        torch.cuda.empty_cache()
        gc.collect()

    @abstractmethod
    def compute_loss(self, *args, **kwargs):
        pass

    def backward(self, loss):
        if self.mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def train(self):
        return self.model.train()

    def eval(self):
        return self.model.eval()

    def optimization_step(self):
        self.scaler.unscale_(self.optimizer)
        if self.clip_grad > 0:
            _grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            if self.use_radamw and (_grad_norm > self.clip_grad):
                logger.info(f"Gradient spike... skipping update {_grad_norm=}")
                self.optimizer.skip_step()
        if self.mixed_precision:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        if self.clip_grad > 0:
            grad_stats = grad_logger(self.model.named_parameters())
            grad_stats.global_norm = float(_grad_norm)
        else:
            grad_stats = None
        self.optimizer.zero_grad()
        optim_stats = adamw_logger(self.optimizer)
        return grad_stats, optim_stats
