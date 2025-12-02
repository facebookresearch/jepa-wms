from typing import Union

import torch
from tensordict import TensorDict

# TODO: Add the classical estimated discounted reward objective.

# #######################
# OBJECTIVES TO MINIMIZE
# #######################


def cos(a, b):
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    return (a * b).sum(-1)


class BaseMPCObjective:
    """Base class for MPC objective.
    This is a callable that takes encodings and returns a tensor -
    objective to be optimized.
    """

    def __call__(self, encodings: torch.Tensor, actions: torch.Tensor, keepdims: bool = False) -> torch.Tensor:
        pass


class ReprTargetCosMPCObjective(BaseMPCObjective):
    """Objective to minimize minus the cosine similarity to the target representation."""

    def __init__(
        self,
        cfg: dict,
        target_enc: torch.Tensor,
        sum_all_diffs: bool = False,
        alpha: float = 1.0,  # weight for proprioceptive loss
        **kwargs,
    ):
        self.cfg = cfg
        self.target_enc = target_enc
        self.sum_all_diffs = sum_all_diffs
        self.alpha = alpha

    def __call__(
        self, encodings: Union[torch.Tensor, TensorDict], actions: torch.Tensor, keepdims: bool = False
    ) -> torch.Tensor:
        """
        Args:
            encodings: tensor or TensorDict,
                if tensor: (T x B x ... x D) for visual, (T x B x ... x P) for proprio
                if TensorDict: {'visual': (T x B x ... x D), 'proprio': (T x B x ... x P)}
                in general: D = P and ... = N or ... = V, H, W
            target_enc: tensor or TensorDict,
                if tensor: (1 x ... x D) for visual or (1 x ... x P) for proprio
                if TensorDict: {'visual': (1 x ... x D), 'proprio': (1 x ... x P)}
                in general: D = P and ... = N or ... = V, H, W
            actions: tensor, (T x B x A)
        Returns:
            loss: tensor, (T x B) or (B) if not keepdims
        """
        if isinstance(encodings, TensorDict) and isinstance(self.target_enc, TensorDict):
            sims_visual = cos(
                self.target_enc["visual"].reshape(1, -1),
                encodings["visual"].reshape(encodings["visual"].shape[0], encodings["visual"].shape[1], -1),
            )
            sims_proprio = cos(
                self.target_enc["proprio"].reshape(1, -1),
                encodings["proprio"].reshape(encodings["proprio"].shape[0], encodings["proprio"].shape[1], -1),
            )
            sims = sims_visual + self.alpha * sims_proprio
        elif isinstance(encodings, torch.Tensor) and isinstance(self.target_enc, torch.Tensor):
            sims = cos(
                self.target_enc.reshape(1, -1),
                encodings.reshape(encodings.shape[0], encodings.shape[1], -1),
            )
        else:
            raise ValueError("Input type mismatch")
        if not keepdims:
            if self.sum_all_diffs:
                sims = sims.sum(0)
            else:
                sims = sims[-1]
        elif self.sum_all_diffs:
            sims = sims.cumsum(0).flip(0)
        return -1 * sims


class ReprTargetDistMPCObjective(BaseMPCObjective):
    """Objective to minimize distance to the target representation."""

    def __init__(
        self,
        cfg: dict,
        target_enc: Union[torch.Tensor, TensorDict],
        sum_all_diffs: bool = False,
        alpha: float = 1.0,  # weight for proprioceptive loss
        **kwargs,
    ):
        self.cfg = cfg
        self.target_enc = target_enc
        self.sum_all_diffs = sum_all_diffs
        self.alpha = alpha

    def __call__(
        self, encodings: Union[torch.Tensor, TensorDict], actions: torch.Tensor, keepdims: bool = False
    ) -> torch.Tensor:
        """
        Args:
            encodings: tensor or TensorDict,
                if tensor: (T x B x ... x D) for visual, (T x B x ... x P) for proprio
                if TensorDict: {'visual': (T x B x ... x D), 'proprio': (T x B x ... x P)}
                in general: D = P and ... = N or ... = V, H, W
            target_enc: tensor or TensorDict,
                if tensor: (1 x ... x D) for visual or (1 x ... x P) for proprio
                if TensorDict: {'visual': (1 x ... x D), 'proprio': (1 x ... x P)}
                in general: D = P and ... = N or ... = V, H, W
            actions: tensor, (T x B x A)
        Returns:
            loss: tensor, (T x B) or (B) if not keepdims
        """
        if isinstance(encodings, TensorDict) and isinstance(self.target_enc, TensorDict):
            diff_visual = (
                (self.target_enc["visual"] - encodings["visual"])
                .pow(2)
                .mean(dim=tuple(range(2, encodings["visual"].ndim)))
            )
            diff_proprio = (
                (self.target_enc["proprio"] - encodings["proprio"])
                .pow(2)
                .mean(dim=tuple(range(2, encodings["proprio"].ndim)))
            )
            diff = diff_visual + self.alpha * diff_proprio
        elif isinstance(encodings, torch.Tensor) and isinstance(self.target_enc, torch.Tensor):
            diff = (self.target_enc - encodings).pow(2).mean(dim=tuple(range(2, encodings.ndim)))
        else:
            raise ValueError("Input type mismatch")
        if not keepdims:
            if self.sum_all_diffs:
                diff = diff.sum(0)
            else:
                diff = diff[-1]
        elif self.sum_all_diffs:
            diff = diff.cumsum(0).flip(0)
        return diff


class ReprTargetDistL1MPCObjective(BaseMPCObjective):
    """Objective to minimize L1 distance to the target representation."""

    def __init__(
        self,
        cfg: dict,
        target_enc: Union[torch.Tensor, TensorDict],
        sum_all_diffs: bool = False,
        alpha: float = 1.0,  # weight for proprioceptive loss
        **kwargs,
    ):
        self.cfg = cfg
        self.target_enc = target_enc
        self.sum_all_diffs = sum_all_diffs
        self.alpha = alpha

    def __call__(
        self, encodings: Union[torch.Tensor, TensorDict], actions: torch.Tensor, keepdims: bool = False
    ) -> torch.Tensor:
        """
        Args:
            encodings: tensor or TensorDict,
                if tensor: (T x B x ... x D) for visual, (T x B x ... x P) for proprio
                if TensorDict: {'visual': (T x B x ... x D), 'proprio': (T x B x ... x P)}
                in general: D = P and ... = N or ... = V, H, W
            target_enc: tensor or TensorDict,
                if tensor: (1 x ... x D) for visual or (1 x ... x P) for proprio
                if TensorDict: {'visual': (1 x ... x D), 'proprio': (1 x ... x P)}
                in general: D = P and ... = N or ... = V, H, W
            actions: tensor, (T x B x A)
        Returns:
            loss: tensor, (T x B) or (B) if not keepdims
        """
        if isinstance(encodings, TensorDict) and isinstance(self.target_enc, TensorDict):
            diff_visual = torch.abs(self.target_enc["visual"] - encodings["visual"]).mean(
                dim=tuple(range(2, encodings["visual"].ndim))
            )
            diff_proprio = torch.abs(self.target_enc["proprio"] - encodings["proprio"]).mean(
                dim=tuple(range(2, encodings["proprio"].ndim))
            )
            diff = diff_visual + self.alpha * diff_proprio
        elif isinstance(encodings, torch.Tensor) and isinstance(self.target_enc, torch.Tensor):
            diff = torch.abs(self.target_enc - encodings).mean(dim=tuple(range(2, encodings.ndim)))
        else:
            raise ValueError("Input type mismatch")
        if not keepdims:
            if self.sum_all_diffs:
                diff = diff.sum(0)
            else:
                diff = diff[-1]
        elif self.sum_all_diffs:
            diff = diff.cumsum(0).flip(0)
        return diff
