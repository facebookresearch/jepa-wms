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


class GRASPPlanner(Planner):
    """GRASP: Gradient RelAxed Stochastic Planner.

    Jointly optimizes actions AND virtual states using:
    - Stop-gradient dynamics loss (detach state inputs to world model)
    - Goal shaping loss (pred_next vs goal at every timestep)
    - Langevin-style state noise
    - Periodic full-rollout sync (GD on terminal loss)

    Works with both Tensor and TensorDict latent representations.
    For TensorDict (visual+proprio), virtual states and optimization
    operate on the visual component only; proprio is predicted by the
    world model during rollouts.
    """

    def __init__(
        self,
        unroll: Callable,
        action_dim: int,
        horizon: int,
        lr_s: float = 0.1,
        lr_a: float = 0.001,
        opt_steps: int = 1000,
        state_noise_scale: float = 0.5,
        gd_interval: int = 100,
        gd_opt_steps: int = 25,
        gd_lr: float = 0.1,
        sync_mode: str = "gd",
        cem_sync_samples: int = 64,
        cem_sync_topk: int = 8,
        cem_sync_var_scale: float = 1.0,
        cem_sync_var_min: float = 0.01,
        schedule_decay: bool = False,
        init_noise_scale: float = 0.1,
        min_noise_scale: float = 0.0,
        init_goal_weight: float = 1.0,
        min_goal_weight: float = 0.0,
        num_act_stepped: int = None,
        decode_each_iteration: bool = False,
        decode_unroll: Callable = None,
        local_generator: torch.Generator = None,
        action_noise: float = 0.0,
        sample_type: str = "zero",
        var_scale: float = 1.0,
        max_norms: List[float] = None,
        max_norm_dims: List[List[int]] = [[0, 1, 2], [6]],
        **kwargs,
    ):
        super().__init__(unroll)
        self.action_dim = action_dim
        self.horizon = horizon
        self.lr_s = lr_s
        self.lr_a = lr_a
        self.opt_steps = opt_steps
        self.state_noise_scale = state_noise_scale
        self.gd_interval = gd_interval
        self.gd_opt_steps = gd_opt_steps
        self.gd_lr = gd_lr
        self.sync_mode = sync_mode
        self.cem_sync_samples = cem_sync_samples
        self.cem_sync_topk = cem_sync_topk
        self.cem_sync_var_scale = cem_sync_var_scale
        self.cem_sync_var_min = cem_sync_var_min
        self.schedule_decay = schedule_decay
        self.init_noise_scale = init_noise_scale
        self.min_noise_scale = min_noise_scale
        self.init_goal_weight = init_goal_weight
        self.min_goal_weight = min_goal_weight
        self.num_act_stepped = num_act_stepped
        self.decode_each_iteration = decode_each_iteration
        self.decode_unroll = decode_unroll
        self.local_generator = local_generator
        self.action_noise = action_noise
        self.sample_type = sample_type
        self.var_scale = var_scale
        self.max_norms = max_norms
        self.max_norm_dims = max_norm_dims
        self.device = torch.device("cuda")

    def _is_tensordict(self, x):
        """Check if x is a TensorDict or dict with visual/proprio keys."""
        if isinstance(x, dict):
            return True
        return hasattr(x, '__class__') and 'TensorDict' in x.__class__.__name__

    def _extract_visual(self, z):
        """Extract visual tensor from z (TensorDict or Tensor)."""
        if self._is_tensordict(z):
            return z["visual"]
        return z

    def _make_context(self, visual_flat, B, vis_shape, z_init):
        """Build a context state for the unroll function from a flat visual representation.

        For TensorDict models, constructs a TensorDict with visual and proprio keys.
        The proprio is taken from the initial context (z_init) since we only optimize visual states.

        Args:
            visual_flat: (B, D_vis) flattened visual state
            B: batch size
            vis_shape: shape of a single visual frame (V, H, W, D_emb)
            z_init: original z_init to extract proprio from

        Returns:
            context suitable for self.unroll()
        """
        vis_spatial = visual_flat.view(B, 1, *vis_shape)  # [B, 1, V, H, W, D]
        if self._is_tensordict(z_init):
            from tensordict import TensorDict as TD
            # Use the last frame's proprio from the original context
            proprio = z_init["proprio"][:, -1:].detach()  # [B, 1, Np, D_p]
            proprio = proprio.expand(B, -1, -1, -1)
            return TD({"visual": vis_spatial, "proprio": proprio}, device=self.device)
        return vis_spatial

    def _single_step_rollout_flat(self, s_t_flat, a_t, B, vis_shape, z_init):
        """Single-step world model rollout returning flattened visual prediction.

        Args:
            s_t_flat: (B, D_vis) flattened visual state, should be DETACHED
            a_t: (B, 1, action_dim) action
            B: batch size
            vis_shape: (V, H, W, D_emb)
            z_init: original context for proprio extraction

        Returns:
            pred_next_flat: (B, D_vis) predicted next visual state
        """
        ctx = self._make_context(s_t_flat, B, vis_shape, z_init)
        a_t_transposed = a_t.transpose(0, 1)  # [1, B, action_dim]
        pred = self.unroll(ctx, act_suffix=a_t_transposed)
        # pred: TensorDict or Tensor with shape [2, B, ...]
        pred_vis = self._extract_visual(pred)
        pred_next = pred_vis[1]  # [B, V, H, W, D]
        return pred_next.reshape(B, -1)

    def _compute_grasp_loss(self, s_flat, a, g_flat, s_0_flat, B, vis_shape, z_init, goal_weight=1.0):
        """Compute GRASP loss with stop-gradient dynamics + goal shaping.

        Args:
            s_flat: (B, T-1, D_vis) virtual states
            a: (B, T, action_dim) actions
            g_flat: (B, D_vis) goal visual representation
            s_0_flat: (B, D_vis) start visual state (fixed)
            B, vis_shape, z_init: for context construction
            goal_weight: weight for the goal shaping loss (1.0 = full, 0.0 = no goal shaping)
        """
        T_minus_1 = s_flat.shape[1]
        T = T_minus_1 + 1

        # Full state sequence: s_0, s_1, ..., s_{T-1}, g
        s_full = torch.cat([s_0_flat.unsqueeze(1), s_flat, g_flat.unsqueeze(1)], dim=1)  # (B, T+1, D)

        losses_dyn = []
        losses_goal = []

        for t in range(T):
            s_t = s_full[:, t]  # (B, D)
            a_t = a[:, t:t+1, :]  # (B, 1, action_dim)

            # Stop-gradient on state input
            pred_next_flat = self._single_step_rollout_flat(
                s_t.detach(), a_t, B, vis_shape, z_init
            )

            s_next = s_full[:, t + 1]

            losses_dyn.append(((pred_next_flat - s_next) ** 2).mean())
            losses_goal.append(((pred_next_flat - g_flat) ** 2).mean())

        return sum(losses_dyn) + goal_weight * sum(losses_goal)

    def _sync_step(self, s_0_flat, g_flat, a, B, vis_shape, z_init):
        """Periodic sync: GD on terminal loss |s_T - g|^2, actions only."""
        a_opt = a.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([a_opt], lr=self.gd_lr)

        ctx = self._make_context(s_0_flat.detach(), B, vis_shape, z_init)

        for _ in range(self.gd_opt_steps):
            optimizer.zero_grad()
            a_transposed = a_opt.transpose(0, 1)  # (T, B, A)
            traj = self.unroll(ctx, act_suffix=a_transposed)
            traj_vis = self._extract_visual(traj)
            s_T = traj_vis[-1]  # [B, V, H, W, D]
            s_T_flat = s_T.reshape(B, -1)
            loss = ((s_T_flat - g_flat) ** 2).mean()
            loss.backward()
            optimizer.step()

        return a_opt.detach()

    def _cem_sync_step(self, s_0_flat, g_flat, a, s_virtual, B, vis_shape, z_init):
        """CEM-based sync: sample around current actions with adaptive variance.

        Variance is set from the deviation between virtual states and actual
        rollout states. A minimum floor ensures all actions get some exploration.
        """
        with torch.no_grad():
            a_mean = a.detach().clone()  # (B, T, act_dim)
            T = a_mean.shape[1]
            ctx = self._make_context(s_0_flat.detach(), B, vis_shape, z_init)

            # Compute per-timestep variance from virtual state deviation
            a_transposed = a_mean.transpose(0, 1)
            traj = self.unroll(ctx, act_suffix=a_transposed)
            traj_vis = self._extract_visual(traj)
            # traj_vis: (T+1, B, V, H, W, D)
            rollout_flat = traj_vis[1:-1].reshape(T - 1, B, -1).permute(1, 0, 2)
            # rollout_flat: (B, T-1, D_vis)
            virtual_flat = s_virtual.detach()  # (B, T-1, D_vis)

            if virtual_flat.shape[1] > 0:
                deviation = ((rollout_flat - virtual_flat) ** 2).mean(dim=-1)  # (B, T-1)
                mean_dev = deviation.mean(dim=-1, keepdim=True)  # (B, 1)
                deviation_full = torch.cat([mean_dev, deviation], dim=1)  # (B, T)
                var_t = self.cem_sync_var_scale * deviation_full + self.cem_sync_var_min
            else:
                s_T_flat = traj_vis[-1].reshape(B, -1)
                terminal_gap = ((s_T_flat - g_flat) ** 2).mean(dim=-1, keepdim=True)
                var_t = self.cem_sync_var_scale * terminal_gap + self.cem_sync_var_min

            var_t = var_t.unsqueeze(-1)  # (B, T, 1)

            for _ in range(self.gd_opt_steps):
                noise = torch.randn(
                    B, self.cem_sync_samples, T, self.action_dim,
                    device=a_mean.device,
                )
                samples = a_mean.unsqueeze(1) + noise * var_t.unsqueeze(1).sqrt()

                costs = []
                for i in range(self.cem_sync_samples):
                    sample_ctx = self._make_context(s_0_flat.detach(), B, vis_shape, z_init)
                    sample_a = samples[:, i].transpose(0, 1)
                    sample_traj = self.unroll(sample_ctx, act_suffix=sample_a)
                    sample_vis = self._extract_visual(sample_traj)
                    s_T_flat = sample_vis[-1].reshape(B, -1)
                    cost = ((s_T_flat - g_flat) ** 2).sum(dim=-1)
                    costs.append(cost)
                costs = torch.stack(costs, dim=1)  # (B, N)

                _, elite_idx = torch.topk(costs, self.cem_sync_topk, dim=1, largest=False)
                elite_actions = torch.gather(
                    samples, 1,
                    elite_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, T, self.action_dim)
                )

                a_mean = elite_actions.mean(dim=1)
                var_t = elite_actions.var(dim=1).mean(dim=-1, keepdim=True) + self.cem_sync_var_min

        return a_mean

    def plan(
        self,
        z_init,
        steps_left: int = None,
    ) -> PlanningResult:
        """Plan using GRASP: jointly optimize actions and virtual states."""
        if steps_left is not None:
            plan_length = min(self.horizon, steps_left)
        else:
            plan_length = self.horizon

        T = plan_length
        num_virtual_states = T - 1

        # Extract visual representation
        z_visual = self._extract_visual(z_init)
        B = z_visual.shape[0]
        vis_shape = z_visual.shape[2:]  # (V, H, W, D_emb)
        D_vis = 1
        for d in vis_shape:
            D_vis *= d

        # Use last frame of context as s_0
        s_0_flat = z_visual[:, -1].reshape(B, -1).detach()

        # Get goal visual representation
        target = self.objective.target_enc
        g_visual = self._extract_visual(target)
        if g_visual.dim() == z_visual.dim():
            g_visual = g_visual[:, -1]
        g_flat = g_visual.reshape(1, -1).expand(B, -1).detach()

        # Initialize virtual states by linear interpolation
        if num_virtual_states > 0:
            t_interp = torch.linspace(0, 1, num_virtual_states + 2, device=self.device)
            t_interp = t_interp[1:-1].view(1, -1, 1)
            s = (s_0_flat.unsqueeze(1) + t_interp * (g_flat.unsqueeze(1) - s_0_flat.unsqueeze(1)))
            s = s.clone().detach().requires_grad_(True)
        else:
            s = torch.zeros(B, 0, D_vis, device=self.device)

        # Initialize actions
        a = torch.zeros(B, T, self.action_dim, device=self.device, requires_grad=True)

        # Optimizer for both states and actions
        params = [{"params": a, "lr": self.lr_a}]
        if num_virtual_states > 0:
            params.append({"params": s, "lr": self.lr_s})
        optimizer = torch.optim.Adam(params)

        losses = []
        predicted_best_encs_over_iterations = []

        for k in range(self.opt_steps):
            # Compute scheduled decay within current GD phase
            if self.schedule_decay and self.gd_interval > 0:
                phase_step = k % self.gd_interval
                decay_frac = phase_step / max(self.gd_interval - 1, 1)  # 0 -> 1 over phase
                cur_noise = self.init_noise_scale + (self.min_noise_scale - self.init_noise_scale) * decay_frac
                cur_goal_weight = self.init_goal_weight + (self.min_goal_weight - self.init_goal_weight) * decay_frac
            else:
                cur_noise = self.state_noise_scale
                cur_goal_weight = 1.0

            optimizer.zero_grad()
            loss = self._compute_grasp_loss(s, a, g_flat, s_0_flat, B, vis_shape, z_init, goal_weight=cur_goal_weight)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # Langevin-style state noise (scheduled or constant)
            if num_virtual_states > 0 and cur_noise > 0:
                with torch.no_grad():
                    s.data += cur_noise * torch.randn_like(s)

            # Periodic sync
            if self.gd_interval > 0 and (k + 1) % self.gd_interval == 0 and k > 0:
                if self.sync_mode == "cem":
                    a_synced = self._cem_sync_step(s_0_flat, g_flat, a, s, B, vis_shape, z_init)
                else:
                    a_synced = self._sync_step(s_0_flat, g_flat, a, B, vis_shape, z_init)
                a = a_synced.requires_grad_(True)
                params = [{"params": a, "lr": self.lr_a}]
                if num_virtual_states > 0:
                    params.append({"params": s, "lr": self.lr_s})
                optimizer = torch.optim.Adam(params)

        # Return optimized actions
        final_actions = a.squeeze(0).detach()  # (T, action_dim)
        losses_tensor = torch.tensor(losses).detach().unsqueeze(-1)

        # Get final predicted encodings
        with torch.no_grad():
            ctx = self._make_context(s_0_flat, B, vis_shape, z_init)
            a_transposed = a.detach().transpose(0, 1)
            predicted_best_encs = self.unroll(ctx, act_suffix=a_transposed)
            predicted_best_encs_over_iterations.append(predicted_best_encs)

        result = PlanningResult(
            actions=final_actions[: self.num_act_stepped] if self.num_act_stepped else final_actions,
            losses=losses_tensor,
            prev_elite_losses_mean=losses_tensor,
            prev_elite_losses_std=torch.zeros_like(losses_tensor),
            pred_frames_over_iterations=None,
            predicted_best_encs_over_iterations=predicted_best_encs_over_iterations,
        )
        return result


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
