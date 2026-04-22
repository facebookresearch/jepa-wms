# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABC, abstractmethod
from typing import Callable, List, NamedTuple

import nevergrad as ng
import numpy as np
import torch
import torch.distributed as dist
from tensordict import TensorDict

from evals.simu_env_planning.planning.planning import objectives
from src.utils.logging import get_logger

logger = get_logger(__name__)

########### PLANNERS IN LATENT SPACE ###############


class PlanningResult(NamedTuple):
    actions: torch.Tensor
    # locations that the model has planned to achieve
    losses: torch.Tensor = None
    prev_elite_losses_mean: torch.Tensor = None
    prev_elite_losses_std: torch.Tensor = None
    info: dict = None
    plan_metrics: dict = None
    pred_frames_over_iterations: List = None
    predicted_best_encs_over_iterations: List = None


class Planner(ABC):
    def __init__(self, unroll: Callable):
        self.objective = None
        self.unroll = unroll

    def set_objective(self, objective: objectives.BaseMPCObjective):
        self.objective = objective

    @abstractmethod
    def plan(self, obs: torch.Tensor, steps_left: int):
        pass

    def cost_function(self, actions: torch.Tensor, z_init: torch.Tensor) -> torch.Tensor:
        predicted_encs = self.unroll(z_init, actions)
        return self.objective(predicted_encs, actions)


class NevergradPlanner(Planner):
    def __init__(
        self,
        unroll: Callable,
        action_dim: int,
        iterations: int,
        var_scale: float = 1,
        max_norms: List[float] = None,
        max_norm_dims: List[List[int]] = [[0, 1, 2], [6]],
        num_samples: int = 1,
        horizon: int = None,
        num_act_stepped: int = None,
        decode_each_iteration: bool = False,
        decode_unroll: Callable = None,
        num_elites: int = 10,
        optimizer_name: str = "NgIohTuned",
        **kwargs,
    ):
        super().__init__(unroll)
        self.action_dim = action_dim
        self.iterations = iterations
        self.var_scale = var_scale
        self.max_norms = max_norms
        self.max_norm_dims = max_norm_dims
        self.num_samples = num_samples
        self.horizon = horizon
        self.num_act_stepped = num_act_stepped
        self.decode_each_iteration = decode_each_iteration
        self.decode_unroll = decode_unroll
        self.num_elites = num_elites  # just for logging
        self.optimizer_name = optimizer_name
        self.optimizer_map = {
            "NgIohTuned": ng.optimizers.NgIohTuned,
            "NGOpt": ng.optimizers.NGOpt,
            # CMA-ES variants - numerically stable, good for continuous optimization
            "CMA": ng.optimizers.CMA,
            "ParametrizedCMA": ng.optimizers.ParametrizedCMA,
            "DiagonalCMA": ng.optimizers.DiagonalCMA,
            # Other stable alternatives
            "PSO": ng.optimizers.PSO,
            "DE": ng.optimizers.DE,
            "OnePlusOne": ng.optimizers.OnePlusOne,
            "TwoPointsDE": ng.optimizers.TwoPointsDE,
        }

    def build_optimizer(self, optimizer_name, **kwargs):
        """Build an optimizer by name."""
        if optimizer_name in self.optimizer_map:
            return self.optimizer_map[optimizer_name](**kwargs)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _get_optimizer(self, plan_length: int):
        parametrization = ng.p.Array(shape=(self.horizon, self.action_dim))
        if self.max_norms is not None:
            lower_bounds = -np.ones((plan_length, self.action_dim))
            upper_bounds = np.ones((plan_length, self.action_dim))

            for max_norm_group, dims in zip(self.max_norms, self.max_norm_dims):
                for d in dims:
                    lower_bounds[:, d] = -max_norm_group
                    upper_bounds[:, d] = max_norm_group

            parametrization.set_bounds(lower=lower_bounds, upper=upper_bounds)
        optimizer = self.build_optimizer(
            self.optimizer_name,
            parametrization=parametrization,
            budget=self.iterations * self.num_samples,
            num_workers=self.num_samples,
        )
        logger.info(f"⚙️  Optimizer: {optimizer}")
        logger.info(f"   Optimizer info: {optimizer._info()}")

        # Check if NGOpt selected MetaModel - it causes numerical instability
        # due to polynomial regression overflow when loss variance is low.
        # In this case, replace with DiagonalCMA which is what NGOpt typically
        # selects in other configurations and is more numerically stable.
        if hasattr(optimizer, "optim") and optimizer.optim.name == "MetaModel":
            logger.warning(
                "NGOpt selected MetaModel optimizer which can cause numerical instability. "
                "Switching to DiagonalCMA for better numerical stability."
            )
            optimizer = self.build_optimizer(
                "DiagonalCMA",
                parametrization=parametrization,
                budget=self.iterations * self.num_samples,
                num_workers=self.num_samples,
            )
            logger.info(f"⚙️  Replacement optimizer: {optimizer}")

        if hasattr(optimizer, "optim"):
            if optimizer.optim.name in ["MetaModel", "CMApara"]:
                if hasattr(optimizer.optim, "_optim"):
                    if hasattr(optimizer.optim._optim, "_es") and optimizer.optim._optim._es is not None:
                        logger.info(f"{optimizer.optim._optim._es.inopts=}")
                    else:
                        logger.info("No _es in optimizer")
        return optimizer

    @torch.no_grad()
    def plan(
        self,
        z_init: torch.Tensor,
        steps_left: int = None,
    ) -> PlanningResult:
        if steps_left is not None:
            plan_length = min(self.horizon, steps_left)
        else:
            plan_length = self.horizon
        optimizer = self._get_optimizer(plan_length)
        costs = []
        prev_elite_losses_mean = []
        prev_elite_losses_std = []
        pred_frames_over_iterations = []
        predicted_best_encs_over_iterations = []

        for itr in range(self.iterations):
            candidates = [optimizer.ask() for _ in range(self.num_samples)]
            candidate_values = torch.from_numpy(np.array([c.value for c in candidates])).to(
                device=z_init.device, dtype=torch.float32
            )
            loss = self.cost_function(candidate_values.permute(1, 0, 2), z_init)

            # Check for NaN or Inf values in loss
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                logger.warning(f"NaN or Inf detected in loss at iteration {itr}. Replacing with large values.")
                loss = torch.nan_to_num(loss, nan=1e6, posinf=1e6, neginf=-1e6)

            # for logging
            elite_losses = torch.topk(loss, k=self.num_elites, largest=False).values
            prev_elite_losses_mean.append(elite_losses.mean().item())
            prev_elite_losses_std.append(elite_losses.std().item())

            for i, c in enumerate(candidates):
                optimizer.tell(c, loss[i].item())
            costs.append(loss.min().item())

            best_solution = optimizer.provide_recommendation().value
            actions = torch.tensor(best_solution, device=z_init.device, dtype=torch.float32).unsqueeze(1)
            predicted_best_encs = self.unroll(z_init, act_suffix=actions)
            predicted_best_encs_over_iterations.append(predicted_best_encs)
            if self.decode_each_iteration and self.decode_unroll is not None:
                pred_frames = self.decode_unroll(predicted_best_encs)
                pred_frames_over_iterations.append(pred_frames)

        best_solution = optimizer.provide_recommendation().value
        actions = torch.tensor(best_solution, device=z_init.device)
        result = PlanningResult(
            actions=actions[: self.num_act_stepped],
            losses=torch.tensor(costs).detach().unsqueeze(-1),
            prev_elite_losses_mean=torch.tensor(prev_elite_losses_mean).unsqueeze(-1),
            prev_elite_losses_std=torch.tensor(prev_elite_losses_std).unsqueeze(-1),
            pred_frames_over_iterations=pred_frames_over_iterations if self.decode_each_iteration else None,
            predicted_best_encs_over_iterations=predicted_best_encs_over_iterations,
        )
        return result


class CEMPlanner(Planner):
    def __init__(
        self,
        unroll: Callable,
        iterations: int = 6,
        num_samples: int = 512,
        horizon: int = 32,
        action_dim: int = 4,
        var_scale: float = 1,
        num_elites: int = 64,
        momentum_mean: float = 0.0,
        momentum_std: float = 0.0,
        max_norms: List[float] = None,
        max_norm_dims: List[List[int]] = [[0, 1, 2], [6]],
        distribute_planner: bool = False,
        local_generator: torch.Generator = None,
        num_act_stepped: int = None,
        decode_each_iteration: bool = False,
        decode_unroll: Callable = None,
        **kwargs,
    ):
        super().__init__(unroll)
        self.iterations = iterations
        self.num_samples = num_samples
        self.horizon = horizon
        self.action_dim = action_dim
        self.device = torch.device("cuda")
        self.var_scale = var_scale
        self.num_elites = num_elites
        self.momentum_mean = momentum_mean
        self.momentum_std = momentum_std
        self.max_norms = max_norms
        self.max_norm_dims = max_norm_dims
        self._prev_mean = None
        self.distribute_planner = distribute_planner
        self.local_generator = local_generator
        self.num_act_stepped = num_act_stepped
        self.decode_each_iteration = decode_each_iteration
        self.decode_unroll = decode_unroll

    @torch.no_grad()
    def plan(
        self,
        z_init,
        steps_left=None,
    ):
        """
        Same as MPPIPlanner but without a policy network.
        Plan a sequence of actions using the learned world model.
        This planner assumes independence between temporal dimensions: we sample actions according
        to a diagonal Gaussian

        Args:
                z_init (torch.Tensor): Latent state from which to plan.
                t0 (bool): Whether this is the first observation in the episode.
                eval_mode (bool): Whether to use the mean of the action distribution.
                task (Torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
                torch.Tensor: Action to take in the environment.
        """
        if steps_left is None:
            plan_length = self.horizon
        else:
            plan_length = min(self.horizon, steps_left)
        mean = torch.zeros(plan_length, self.action_dim, device=self.device)
        std = self.var_scale * torch.ones(plan_length, self.action_dim, device=self.device)
        actions = torch.empty(
            plan_length,
            self.num_samples,
            self.action_dim,
            device=self.device,
        )
        losses, elite_means, elite_stds = [], [], []
        predicted_best_encs_over_iterations = []
        if self.decode_each_iteration:
            pred_frames_over_iterations = []
        # Iterate CEM
        for itr in range(self.iterations):
            actions[:, :] = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                plan_length, self.num_samples, self.action_dim, device=std.device, generator=self.local_generator
            )
            # Mean sample inclusion trick to never loose best previous action
            actions[:, 0, :] = mean
            # Apply clipping if max_norms is specified
            if self.max_norms is not None:
                for h in range(plan_length):
                    # Loop through each group of dimensions to clip
                    for i, (dims, maxnorm) in enumerate(zip(self.max_norm_dims, self.max_norms)):
                        # Clip the specified dimensions to [-maxnorm, maxnorm]
                        actions[h, :, dims] = torch.clip(actions[h, :, dims], min=-maxnorm, max=maxnorm)
            # Compute elite actions
            cost = self.cost_function(actions, z_init).unsqueeze(1)
            losses.append(cost.min().item())
            # Gather all values
            if self.distribute_planner:
                cost = torch.cat(FullGatherLayer.apply(cost), dim=0)
                all_actions = torch.cat(FullGatherLayer.apply(actions), dim=1)
            else:
                all_actions = actions
            elite_idxs = torch.topk(-cost.squeeze(1), self.num_elites, dim=0).indices
            elite_loss, elite_actions = cost[elite_idxs], all_actions[:, elite_idxs]  # [EL,1] , [H,EL,A]
            # Log the mean and std of the elite values
            elite_means.append(elite_loss.mean().item())
            elite_stds.append(elite_loss.std().item())
            # Update parameters with momentum
            new_mean = torch.mean(elite_actions, dim=1)
            new_std = torch.std(elite_actions, dim=1)
            # Apply momentum to mean and std updates
            mean = new_mean * (1 - self.momentum_mean) + mean * self.momentum_mean
            std = new_std * (1 - self.momentum_std) + std * self.momentum_std
            # Decoding logic
            predicted_best_encs = self.unroll(z_init, act_suffix=mean.unsqueeze(1))
            predicted_best_encs_over_iterations.append(predicted_best_encs)
            if self.decode_each_iteration and self.decode_unroll is not None:
                pred_frames = self.decode_unroll(
                    predicted_best_encs,
                )
                pred_frames_over_iterations.append(pred_frames)
                # [T H W 3]: uint 8 in [0, 255]

        self._prev_mean = mean
        a = mean[: self.num_act_stepped]
        if self.distribute_planner:
            dist.broadcast(a, src=0)
        result = PlanningResult(
            actions=a,
            losses=torch.tensor(losses).detach().unsqueeze(-1),
            prev_elite_losses_mean=torch.tensor(elite_means).unsqueeze(-1),
            prev_elite_losses_std=torch.tensor(elite_stds).unsqueeze(-1),
            pred_frames_over_iterations=pred_frames_over_iterations if self.decode_each_iteration else None,
            predicted_best_encs_over_iterations=predicted_best_encs_over_iterations,
        )
        return result

class GRASPlanner(Planner):
    def __init__(
        self,
        unroll: Callable,
        action_dim: int,
        horizon: int = 32,
        iterations: int = 100,
        state_lr: float = 0.01,
        action_lr: float = 0.01,
        gamma: float = 1.0,
        K_sync: int = 10,
        J_sync: int = 1,
        sync_lr: float = 0.01,
        state_sigma: float = 0.01,
        epsilon: float = 0.01,
        var_scale: float = 1.0,
        max_norms: List[float] = None,
        max_norm_dims: List[List[int]] = [[0, 1, 2], [6]],
        num_act_stepped: int = None,
        decode_each_iteration: bool = False,
        decode_unroll: Callable = None,
        enc_pred_wm: torch.nn.Module = None,
        **kwargs,
    ):
        super().__init__(unroll)
        self.action_dim = action_dim
        self.horizon = horizon
        self.iterations = iterations
        self.state_lr = state_lr
        self.action_lr = action_lr
        self.gamma = gamma
        self.K_sync = K_sync
        self.J_sync = J_sync
        self.sync_lr = sync_lr
        self.state_sigma = state_sigma
        self.epsilon = epsilon
        self.var_scale = var_scale
        self.max_norms = max_norms
        self.max_norm_dims = max_norm_dims
        self.num_act_stepped = num_act_stepped
        self.decode_each_iteration = decode_each_iteration
        self.decode_unroll = decode_unroll
        self.enc_pred_wm = enc_pred_wm
        self.device = torch.device("cuda")

    def _clip_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if self.max_norms is not None:
            for dims, maxnorm in zip(self.max_norm_dims, self.max_norms):
                actions[:, :, dims] = actions[:, :, dims].clamp(-maxnorm, maxnorm)
        return actions

    def plan(
        self,
        z_init,
        steps_left=None,
    ) -> PlanningResult:
        plan_length = min(self.horizon, steps_left) if steps_left is not None else self.horizon

        z_goal = self.objective.target_enc
        is_tensordict = isinstance(z_init, TensorDict)
        if is_tensordict:
            z_init_visual = z_init["visual"]
            z_init_proprio = z_init["proprio"]
            z_goal_visual = z_goal["visual"]
        else:
            z_init_visual = z_init
            z_init_proprio = None
            z_goal_visual = z_goal

        B = z_init_visual.shape[0]
        tau = z_init_visual.shape[1]
        state_shape = z_init_visual.shape[2:]

        s_0 = z_init_visual[:, -1:, ...]
        proprio_const = z_init_proprio[:, -1:, ...] if z_init_proprio is not None else None
        if proprio_const is not None:
            proprio_traj = proprio_const.expand(-1, plan_length, *proprio_const.shape[2:]).clone()
        else:
            proprio_traj = None
        z_goal_flat = z_goal_visual[:, -1:, ...]

        self.enc_pred_wm.model.eval()
        for p in self.enc_pred_wm.model.parameters():
            p.requires_grad_(False)

        actions = torch.zeros(plan_length, B, self.action_dim, device=self.device)
        actions = actions.detach().requires_grad_(True)

        weights = torch.linspace(1.0 / plan_length, 1.0, plan_length, device=self.device)
        w = weights.view(plan_length, *([1] * (len(state_shape) + 1)))
        s_0_squeezed = s_0.squeeze(1)
        g_squeezed = z_goal_flat.squeeze(1)
        states_interp = (1 - w) * s_0_squeezed.unsqueeze(0) + w * g_squeezed.unsqueeze(0)
        states = (states_interp + self.epsilon * torch.randn_like(states_interp)).detach()
        states[-1] = g_squeezed.clone()
        states = states.requires_grad_(True)

        losses = []
        predicted_best_encs_over_iterations = []
        pred_frames_over_iterations = [] if self.decode_each_iteration else None

        if plan_length > 1:
            win_idx = torch.arange(plan_length - 1, device=self.device).unsqueeze(1) + torch.arange(2, device=self.device).unsqueeze(0)

        for k in range(self.iterations):
            states_B = states.permute(1, 0, *range(2, states.ndim))
            full_states = torch.cat([s_0, states_B], dim=1)

            actions_BTA = actions.permute(1, 0, 2)
            act_feats = self.enc_pred_wm.model.encode_act(actions_BTA)

            vid_ctx_0 = s_0.detach()
            act_ctx_0 = act_feats[:, 0:1]
            prop_ctx_0 = proprio_const if z_init_proprio is not None else None
            pred_0_full, _, _ = self.enc_pred_wm.model.forward_pred(vid_ctx_0, act_ctx_0, prop_ctx_0)
            pred_0 = pred_0_full[:, -1]

            if plan_length > 1:
                vid_batch = full_states[:, win_idx].reshape(
                    B * (plan_length - 1), 2, *state_shape
                ).detach()
                act_batch = act_feats[:, win_idx].reshape(
                    B * (plan_length - 1), 2, self.action_dim
                )
                if z_init_proprio is not None:
                    full_prop = torch.cat([proprio_const, proprio_traj], dim=1)
                    prop_batch = full_prop[:, win_idx].reshape(
                        B * (plan_length - 1), 2, *proprio_const.shape[2:]
                    )
                else:
                    prop_batch = None

                batch_preds_full, _, _ = self.enc_pred_wm.model.forward_pred(
                    vid_batch, act_batch, prop_batch
                )
                batch_preds = batch_preds_full[:, -1]
                batch_preds = batch_preds.view(plan_length - 1, B, *batch_preds.shape[1:])
                all_preds = torch.cat([pred_0.unsqueeze(0), batch_preds], dim=0)
            else:
                all_preds = pred_0.unsqueeze(0)

            states_target = full_states[:, 1:].permute(
                1, 0, *range(2, full_states.ndim)
            )
            dyn_loss = (all_preds - states_target).pow(2).sum(0).mean()

            pred_stack = all_preds
            if is_tensordict:
                proprio_expanded = proprio_traj.permute(
                    1, 0, *range(2, proprio_traj.ndim)
                )
                pred_for_obj = TensorDict({"visual": pred_stack, "proprio": proprio_expanded})
            else:
                pred_for_obj = pred_stack
            goal_loss = self.objective(pred_for_obj, actions_BTA, keepdims=True)
            goal_loss = goal_loss.sum(0).mean()

            total_loss = dyn_loss + self.gamma * goal_loss
            losses.append(total_loss.item())

            grad_actions, grad_states = torch.autograd.grad(total_loss, [actions, states])

            grad_states[-1] = 0
            noise = self.state_sigma * torch.randn_like(states)
            noise[-1] = 0
            actions = self._clip_actions(actions - self.action_lr * grad_actions).detach().requires_grad_(True)
            states = (states - self.state_lr * grad_states + noise).detach()
            states[-1] = g_squeezed.clone()
            states = states.requires_grad_(True)

            if (k + 1) % self.K_sync == 0:
                for _ in range(self.J_sync):
                    actions_for_sync = actions.detach().clone().requires_grad_(True)
                    sync_rollout = self.unroll(z_init, act_suffix=actions_for_sync)
                    if is_tensordict:
                        final_state = sync_rollout["visual"][-1:]
                    else:
                        final_state = sync_rollout[-1:]
                    sync_goal_loss = (final_state - z_goal_flat).pow(2).mean()
                    grad_sync = torch.autograd.grad(sync_goal_loss, actions_for_sync)[0]
                    actions = self._clip_actions(actions - self.sync_lr * grad_sync).detach().requires_grad_(True)

                with torch.no_grad():
                    sync_rollout = self.unroll(z_init, act_suffix=actions.detach())
                    if is_tensordict:
                        new_states = sync_rollout["visual"][tau:]
                        new_proprio = sync_rollout["proprio"][tau:]
                        proprio_traj = new_proprio.permute(1, 0, *range(2, new_proprio.ndim)).detach()
                    else:
                        new_states = sync_rollout[tau:]
                    states_data = new_states.detach()
                    states_data[-1] = g_squeezed.clone()
                    states = states_data.requires_grad_(True)

            if self.decode_each_iteration and self.decode_unroll is not None:
                with torch.no_grad():
                    pred_encs = self.unroll(z_init, act_suffix=actions.detach())
                    predicted_best_encs_over_iterations.append(pred_encs)
                    pred_frames_over_iterations.append(self.decode_unroll(pred_encs))

        with torch.no_grad():
            final_pred_encs = self.unroll(z_init, act_suffix=actions.detach())
            predicted_best_encs_over_iterations.append(final_pred_encs)

        final_actions = actions.detach().squeeze(1)
        result = PlanningResult(
            actions=final_actions[: self.num_act_stepped] if self.num_act_stepped else final_actions,
            losses=torch.tensor(losses).detach().unsqueeze(-1),
            prev_elite_losses_mean=torch.tensor(losses).detach().unsqueeze(-1),
            prev_elite_losses_std=torch.zeros(len(losses)).unsqueeze(-1),
            pred_frames_over_iterations=pred_frames_over_iterations,
            predicted_best_encs_over_iterations=predicted_best_encs_over_iterations,
        )
        return result


class CEMGDPlanner(Planner):
    """CEM-GD Planner: Cross-Entropy Method with Gradient Descent refinement.

    Combines CEM sampling with gradient-based optimization:
    1. Sample N action sequences from a Gaussian, iterate CEM to refine the distribution.
    2. Select top-k elite sequences from the final CEM distribution.
    3. Refine each elite with G gradient descent steps, trying J geometrically-decaying
       step sizes (eta_init * rho^j for j=0..J-1) and keeping the best.
    4. Pick the overall best refined sequence and execute the first action(s).
    """

    def __init__(
        self,
        unroll: Callable,
        action_dim: int,
        horizon: int = 32,
        # CEM parameters
        num_samples: int = 512,
        var_scale: float = 1.0,
        num_elites: int = 64,
        iterations: int = 6,
        momentum_mean: float = 0.0,
        momentum_std: float = 0.0,
        # GD refinement parameters
        top_k: int = 1,
        gd_steps: int = 10,
        gd_lr_init: float = 0.01,
        gd_lr_decay: float = 0.67,
        gd_num_lrs: int = 8,
        # Action clipping
        max_norms: List[float] = None,
        max_norm_dims: List[List[int]] = [[0, 1, 2], [6]],
        # Output
        num_act_stepped: int = None,
        decode_each_iteration: bool = False,
        decode_unroll: Callable = None,
        # Distribution
        distribute_planner: bool = False,
        local_generator: torch.Generator = None,
        **kwargs,
    ):
        super().__init__(unroll)
        self.action_dim = action_dim
        self.horizon = horizon
        self.device = torch.device("cuda")
        self.num_samples = num_samples
        self.var_scale = var_scale
        self.num_elites = num_elites
        self.iterations = iterations
        self.momentum_mean = momentum_mean
        self.momentum_std = momentum_std
        self.top_k = top_k
        self.gd_steps = gd_steps
        self.gd_lr_init = gd_lr_init
        self.gd_lr_decay = gd_lr_decay
        self.gd_num_lrs = gd_num_lrs
        self.max_norms = max_norms
        self.max_norm_dims = max_norm_dims
        self.num_act_stepped = num_act_stepped
        self.decode_each_iteration = decode_each_iteration
        self.decode_unroll = decode_unroll
        self.distribute_planner = distribute_planner
        self.local_generator = local_generator

    def _clip_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if self.max_norms is not None:
            for dims, maxnorm in zip(self.max_norm_dims, self.max_norms):
                actions[..., dims] = torch.clip(actions[..., dims], min=-maxnorm, max=maxnorm)
        return actions

    def _gd_refine(self, actions_init: torch.Tensor, z_init: torch.Tensor) -> torch.Tensor:
        actions = actions_init.clone().unsqueeze(0).detach().requires_grad_(True)
        decay = self.gd_lr_decay if 0.0 < self.gd_lr_decay < 1.0 else 0.5

        for _ in range(self.gd_steps):
            if actions.grad is not None:
                actions.grad.zero_()

            actions_for_unroll = actions.squeeze(0).unsqueeze(1)
            predicted_encs = self.unroll(z_init, act_suffix=actions_for_unroll)
            loss = self.objective(predicted_encs, actions_for_unroll)
            current_cost = loss.mean()
            current_cost.backward()

            if actions.grad is None:
                continue

            with torch.no_grad():
                grad = actions.grad.detach().clone()
                if not torch.isfinite(grad).all():
                    actions.grad.zero_()
                    continue

                base_actions = actions.detach().clone()
                base_cost = current_cost.detach()
                lr = self.gd_lr_init

                for _ in range(self.gd_num_lrs):
                    candidate = self._clip_actions(base_actions - lr * grad)
                    candidate_for_unroll = candidate.squeeze(0).unsqueeze(1)
                    candidate_cost = self.cost_function(candidate_for_unroll, z_init).mean()

                    if torch.isfinite(candidate_cost) and candidate_cost < base_cost:
                        actions.copy_(candidate)
                        break

                    lr *= decay

            actions.grad.zero_()

        return actions.squeeze(0).detach().clone()

    def plan(
        self,
        z_init: torch.Tensor,
        steps_left: int = None,
    ) -> PlanningResult:
        if steps_left is not None:
            plan_length = min(self.horizon, steps_left)
        else:
            plan_length = self.horizon

        mean = torch.zeros(plan_length, self.action_dim, device=self.device)
        std = self.var_scale * torch.ones(plan_length, self.action_dim, device=self.device)
        actions = torch.empty(
            plan_length, self.num_samples, self.action_dim, device=self.device,
        )
        losses = []
        elite_means, elite_stds = [], []
        predicted_best_encs_over_iterations = []
        pred_frames_over_iterations = [] if self.decode_each_iteration else None

        with torch.no_grad():
            for itr in range(self.iterations):
                actions[:, :] = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                    plan_length, self.num_samples, self.action_dim,
                    device=self.device, generator=self.local_generator,
                )
                actions[:, 0, :] = mean
                if self.max_norms is not None:
                    for h in range(plan_length):
                        for dims, maxnorm in zip(self.max_norm_dims, self.max_norms):
                            actions[h, :, dims] = torch.clip(actions[h, :, dims], min=-maxnorm, max=maxnorm)
                cost = self.cost_function(actions, z_init).unsqueeze(1)
                losses.append(cost.min().item())
                if self.distribute_planner:
                    cost = torch.cat(FullGatherLayer.apply(cost), dim=0)
                    all_actions = torch.cat(FullGatherLayer.apply(actions), dim=1)
                else:
                    all_actions = actions
                elite_idxs = torch.topk(-cost.squeeze(1), self.num_elites, dim=0).indices
                elite_loss, elite_actions = cost[elite_idxs], all_actions[:, elite_idxs]
                elite_means.append(elite_loss.mean().item())
                elite_stds.append(elite_loss.std().item())
                new_mean = torch.mean(elite_actions, dim=1)
                new_std = torch.std(elite_actions, dim=1)
                mean = new_mean * (1 - self.momentum_mean) + mean * self.momentum_mean
                std = new_std * (1 - self.momentum_std) + std * self.momentum_std

            top_k_idxs = torch.topk(-cost.squeeze(1), self.top_k, dim=0).indices
            top_k_actions = all_actions[:, top_k_idxs, :]

        best_overall_cost = float("inf")
        best_overall_actions = None

        for i in range(self.top_k):
            candidate = top_k_actions[:, i, :]
            refined = self._gd_refine(candidate, z_init)

            with torch.no_grad():
                refined_for_unroll = refined.unsqueeze(1)
                refined_cost = self.cost_function(refined_for_unroll, z_init).item()

            if refined_cost < best_overall_cost:
                best_overall_cost = refined_cost
                best_overall_actions = refined

        losses.append(best_overall_cost)

        with torch.no_grad():
            final_for_unroll = best_overall_actions.unsqueeze(1)
            predicted_best_encs = self.unroll(z_init, act_suffix=final_for_unroll)
            predicted_best_encs_over_iterations.append(predicted_best_encs)
            if self.decode_each_iteration and self.decode_unroll is not None:
                pred_frames = self.decode_unroll(predicted_best_encs)
                pred_frames_over_iterations.append(pred_frames)

        a = best_overall_actions[: self.num_act_stepped] if self.num_act_stepped else best_overall_actions
        if self.distribute_planner:
            dist.broadcast(a, src=0)

        losses_tensor = torch.tensor(losses).detach().unsqueeze(-1)
        result = PlanningResult(
            actions=a,
            losses=losses_tensor,
            prev_elite_losses_mean=torch.tensor(elite_means).unsqueeze(-1),
            prev_elite_losses_std=torch.tensor(elite_stds).unsqueeze(-1),
            pred_frames_over_iterations=pred_frames_over_iterations if self.decode_each_iteration else None,
            predicted_best_encs_over_iterations=predicted_best_encs_over_iterations,
        )
        return result


class MPPIPlanner(Planner):
    def __init__(
        self,
        unroll: Callable,
        iterations: int = 6,
        num_samples: int = 512,
        horizon: int = 32,
        action_dim: int = 4,
        max_std: float = 2,
        min_std: float = 0.05,
        num_elites: int = 64,
        temperature: float = 0.5,
        distribute_planner: bool = False,
        local_generator: torch.Generator = None,
        num_act_stepped: int = None,
        decode_each_iteration: bool = False,
        decode_unroll: Callable = None,
        **kwargs,
    ):
        super().__init__(unroll)
        self.iterations = iterations
        self.num_samples = num_samples
        self.horizon = horizon
        self.action_dim = action_dim
        self.device = torch.device("cuda")
        self.max_std = max_std
        self.min_std = min_std
        self.num_elites = num_elites
        self.temperature = temperature
        self._prev_mean = None
        self.distribute_planner = distribute_planner
        self.local_generator = local_generator
        self.num_act_stepped = num_act_stepped
        self.decode_each_iteration = decode_each_iteration
        self.decode_unroll = decode_unroll

    @torch.no_grad()
    def plan(self, z_init, eval_mode=False, task=None, steps_left=None):
        """
        MPPIPlanner without a policy network.
        Plan a sequence of actions using the learned world model.

        Args:
                z_init (torch.Tensor): Latent state from which to plan.
                t0 (bool): Whether this is the first observation in the episode.
                eval_mode (bool): Whether to use the mean of the action distribution.
                task (Torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
                torch.Tensor: Action to take in the environment.
        """
        if steps_left is None:
            plan_length = self.horizon
        else:
            plan_length = min(self.horizon, steps_left)

        # Initialize state and parameters
        mean = torch.zeros(plan_length, self.action_dim, device=self.device)
        std = self.max_std * torch.ones(plan_length, self.action_dim, device=self.device)
        actions = torch.empty(
            plan_length,
            self.num_samples,
            self.action_dim,
            device=self.device,
        )

        losses, elite_means, elite_stds = [], [], []
        predicted_best_encs_over_iterations = []
        if self.decode_each_iteration:
            pred_frames_over_iterations = []
        # Iterate MPPI
        for _ in range(self.iterations):
            # Sample actions
            actions[:, :] = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                plan_length,
                self.num_samples,
                self.action_dim,
                device=std.device,
                generator=self.local_generator,
            )
            # Compute costs
            cost = self.cost_function(actions, z_init).unsqueeze(1)
            losses.append(cost.min().item())
            # Get elite actions
            elite_idxs = torch.topk(-cost.squeeze(1), self.num_elites, dim=0).indices
            elite_loss, elite_actions = cost[elite_idxs], actions[:, elite_idxs]
            # Record statistics
            elite_means.append(elite_loss.mean().item())
            elite_stds.append(elite_loss.std().item())
            # Update parameters
            min_cost = cost.min(0)[0]
            score = torch.exp(self.temperature * (min_cost - elite_loss[:, 0]))  # increasing with elite_value
            score /= score.sum(0)
            mean = torch.sum(score.unsqueeze(0).unsqueeze(2) * elite_actions, dim=1) / (score.sum(0) + 1e-9)  # T B A
            std = torch.sqrt(
                torch.sum(
                    score.unsqueeze(0).unsqueeze(2) * (elite_actions - mean.unsqueeze(1)) ** 2,
                    dim=1,  # T B A
                )
                / (score.sum(0) + 1e-9)
            )
            # Decoding logic
            predicted_best_encs = self.unroll(z_init, act_suffix=mean.unsqueeze(1))
            predicted_best_encs_over_iterations.append(predicted_best_encs)
            if self.decode_each_iteration and self.decode_unroll is not None:
                pred_frames = self.decode_unroll(
                    predicted_best_encs,
                )
                pred_frames_over_iterations.append(pred_frames)
                # [T H W 3]: uint 8 in [0, 255]
        # Select action
        score = score.cpu().numpy()  # [EL,]
        # actions: [H, A]
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]  # [H,A]
        self._prev_mean = mean
        a, std = actions[: self.num_act_stepped], std[: self.num_act_stepped]  # [N, A], [N, A]
        if not eval_mode:
            a += std * torch.randn(self.action_dim, device=std.device, generator=self.local_generator)
        # to make sure each GPU outputs same action
        if self.distribute_planner:
            dist.broadcast(a, src=0)

        result = PlanningResult(
            actions=a,
            losses=torch.tensor(losses).detach().unsqueeze(-1),
            prev_elite_losses_mean=torch.tensor(elite_means).unsqueeze(-1),
            prev_elite_losses_std=torch.tensor(elite_stds).unsqueeze(-1),
            pred_frames_over_iterations=pred_frames_over_iterations if self.decode_each_iteration else None,
            predicted_best_encs_over_iterations=predicted_best_encs_over_iterations,
        )
        return result


class GradientDescentPlanner(Planner):
    def __init__(
        self,
        unroll: Callable,
        action_dim: int,
        horizon: int,
        iterations: int = 500,
        lr: float = 1,
        action_noise: float = 0.003,
        sample_type: str = "randn",
        var_scale: float = 1,
        max_norms: List[float] = None,
        max_norm_dims: List[List[int]] = [[0, 1, 2], [6]],
        num_act_stepped: int = None,
        decode_each_iteration: bool = False,
        decode_unroll: Callable = None,
        optimizer_type: str = "sgd",
        adam_betas: tuple = (0.9, 0.995),
        adam_eps: float = 1e-8,
        **kwargs,
    ):
        """
        Gradient Descent Planner for action optimization in latent space.

        Args:
            unroll: Function to unroll the world model
            action_dim: Dimension of the action space
            horizon: Planning horizon (number of timesteps)
            iterations: Number of optimization iterations
            lr: Learning rate for gradient descent
            action_noise: Standard deviation of Gaussian noise to add after each gradient step
            sample_type: Type of action initialization ("randn" or "zero")
            max_norms: List of maximum norm values for each group of dimensions (None to disable clipping)
            max_norm_dims: List of dimension groups to clip (e.g., [[0, 1, 2], [6]])
            num_act_stepped: Number of actions to execute (default: all)
            decode_each_iteration: Whether to decode predictions at each iteration
            decode_unroll: Function to decode latent predictions to frames
            optimizer_type: Type of optimizer to use ("sgd" or "adam")
            adam_betas: Betas for Adam optimizer (default: (0.9, 0.995))
            adam_eps: Epsilon for Adam optimizer (default: 1e-8)
        """
        super().__init__(unroll)
        self.action_dim = action_dim
        self.horizon = horizon
        self.iterations = iterations
        self.lr = lr
        self.action_noise = action_noise
        self.var_scale = var_scale
        self.sample_type = sample_type
        self.max_norms = max_norms
        self.max_norm_dims = max_norm_dims
        self.num_act_stepped = num_act_stepped
        self.decode_each_iteration = decode_each_iteration
        self.decode_unroll = decode_unroll
        self.optimizer_type = optimizer_type.lower()
        self.adam_betas = adam_betas
        self.adam_eps = adam_eps
        self.device = torch.device("cuda")

    def init_actions(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Initialize actions for planning.

        Args:
            device: Device to place actions on

        Returns:
            actions: (1, horizon, action_dim) initialized actions
        """
        if self.sample_type == "randn":
            actions = torch.randn(1, self.horizon, self.action_dim, device=device) * self.var_scale
        elif self.sample_type == "zero":
            actions = torch.zeros(1, self.horizon, self.action_dim, device=device)
        else:
            raise ValueError(f"Unknown sample_type: {self.sample_type}")
        return actions

    def plan(
        self,
        z_init: torch.Tensor,
        steps_left: int = None,
    ) -> PlanningResult:
        """
        Plan a sequence of actions using gradient descent optimization.

        Args:
            z_init: Initial latent state
            steps_left: Number of steps left in episode (optional)

        Returns:
            PlanningResult with optimized actions and planning metrics
        """
        if steps_left is not None:
            plan_length = min(self.horizon, steps_left)
        else:
            plan_length = self.horizon

        # Initialize actions: (batch_size, plan_length, action_dim)
        actions = self.init_actions(1, self.device)[:, :plan_length, :]
        actions.requires_grad = True

        # Setup optimizer based on optimizer_type
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam([actions], lr=self.lr, betas=self.adam_betas, eps=self.adam_eps)
        else:
            optimizer = torch.optim.SGD([actions], lr=self.lr)

        losses = []
        predicted_best_encs_over_iterations = []
        if self.decode_each_iteration:
            pred_frames_over_iterations = []

        # Optimization loop
        for itr in range(self.iterations):
            optimizer.zero_grad()

            # Unroll world model with current actions
            # actions shape: (1, plan_length, action_dim)
            # Need to transpose to (plan_length, 1, action_dim) for unroll
            actions_transposed = actions.transpose(0, 1)

            predicted_encs = self.unroll(z_init, act_suffix=actions_transposed)
            loss = self.objective(predicted_encs, actions_transposed)  # (1,)

            total_loss = loss.mean()
            total_loss.backward()

            # Manual gradient descent update with noise
            with torch.no_grad():
                actions_new = actions - self.lr * actions.grad

                # Add Gaussian noise if specified
                if self.action_noise > 0:
                    actions_new += torch.randn_like(actions_new) * self.action_noise

                # Apply clipping if max_norms is specified (similar to CEM)
                if self.max_norms is not None:
                    for dims, maxnorm in zip(self.max_norm_dims, self.max_norms):
                        actions_new[:, :, dims] = torch.clip(actions_new[:, :, dims], min=-maxnorm, max=maxnorm)

                actions.copy_(actions_new)

            # Reset gradients after manual update
            actions.grad.zero_()

            losses.append(total_loss.item())

            # Store predictions for this iteration
            with torch.no_grad():
                predicted_best_encs = self.unroll(z_init, act_suffix=actions.transpose(0, 1))
                predicted_best_encs_over_iterations.append(predicted_best_encs)

                if self.decode_each_iteration and self.decode_unroll is not None:
                    pred_frames = self.decode_unroll(predicted_best_encs)
                    pred_frames_over_iterations.append(pred_frames)

        # Return the optimized actions
        final_actions = actions.squeeze(0).detach()
        losses = torch.tensor(losses).detach().unsqueeze(-1)

        result = PlanningResult(
            actions=final_actions[: self.num_act_stepped] if self.num_act_stepped else final_actions,
            losses=losses,
            prev_elite_losses_mean=losses,
            prev_elite_losses_std=torch.zeros_like(losses),
            pred_frames_over_iterations=pred_frames_over_iterations if self.decode_each_iteration else None,
            predicted_best_encs_over_iterations=predicted_best_encs_over_iterations,
        )
        return result


class AdamPlanner(GradientDescentPlanner):
    """Adam optimizer-based planner for action optimization in latent space.

    This is a convenience wrapper around GradientDescentPlanner with optimizer_type="adam".
    """

    def __init__(
        self,
        unroll: Callable,
        action_dim: int,
        horizon: int,
        iterations: int = 500,
        lr: float = 1,
        action_noise: float = 0.003,
        sample_type: str = "randn",
        var_scale: float = 1,
        max_norms: List[float] = None,
        max_norm_dims: List[List[int]] = [[0, 1, 2], [6]],
        num_act_stepped: int = None,
        decode_each_iteration: bool = False,
        decode_unroll: Callable = None,
        adam_betas: tuple = (0.9, 0.995),
        adam_eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__(
            unroll=unroll,
            action_dim=action_dim,
            horizon=horizon,
            iterations=iterations,
            lr=lr,
            action_noise=action_noise,
            sample_type=sample_type,
            var_scale=var_scale,
            max_norms=max_norms,
            max_norm_dims=max_norm_dims,
            num_act_stepped=num_act_stepped,
            decode_each_iteration=decode_each_iteration,
            decode_unroll=decode_unroll,
            optimizer_type="adam",
            adam_betas=adam_betas,
            adam_eps=adam_eps,
            **kwargs,
        )


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]
