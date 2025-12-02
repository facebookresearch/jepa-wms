# The below code is inspired from TD-MPC2 https://github.com/nicklashansen/tdmpc2
# licensed under the MIT License

import torch

from evals.simu_env_planning.planning.planning.objectives import (
    ReprTargetCosMPCObjective,
    ReprTargetDistL1MPCObjective,
    ReprTargetDistMPCObjective,
)
from evals.simu_env_planning.planning.planning.planner import (
    CEMPlanner,
    MPPIPlanner,
    NevergradPlanner,
    GradientDescentPlanner,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class GC_Agent:
    """A modification of TDMPC2 that
    allows for pre-training offline in self-supervised manner.
    Allows for not using the reward information during pre-training.
    """

    def __init__(self, cfg, model, dset=None, preprocessor=None):
        self.cfg = cfg
        self.device = torch.device("cuda", index=0)
        logger.info("starting WorldModel construction")
        self.model = model
        self.dset = dset
        if hasattr(self.dset, "frames_per_clip") and cfg.task_specification.goal_source == "dset":
            if self.dset.frames_per_clip < cfg.frameskip * cfg.task_specification.goal_H + 1:
                self.dset.frames_per_clip = cfg.frameskip * cfg.task_specification.goal_H + 1
        self.preprocessor = preprocessor
        self.local_generator = torch.Generator(device="cpu")
        self.local_gpu_generator = torch.Generator(device="cuda:0")
        self.local_generator.manual_seed(cfg.local_seed)
        self.local_gpu_generator.manual_seed(cfg.local_seed)

        self.model.eval()
        self.goal_state = None
        self._prev_losses = None
        # --------
        if self.cfg.planner.planner_name == "nevergrad":
            self.planner = NevergradPlanner(
                unroll=self.model.unroll,
                action_dim=self.model.action_dim,
                decode_unroll=self.model.decode_unroll,
                **self.cfg.planner,
            )
        elif self.cfg.planner.planner_name == "cem":
            self.planner = CEMPlanner(
                unroll=self.model.unroll,
                action_dim=self.model.action_dim,
                action_masks=None,
                local_generator=self.local_gpu_generator,
                decode_unroll=self.model.decode_unroll,
                **self.cfg.planner,
            )
        elif self.cfg.planner.planner_name == "mppi":
            self.planner = MPPIPlanner(
                unroll=self.model.unroll,
                action_dim=self.model.action_dim,
                action_masks=None,
                local_generator=self.local_gpu_generator,
                decode_unroll=self.model.decode_unroll,
                **self.cfg.planner,
            )
        elif self.cfg.planner.planner_name == "gd":
            self.planner = GradientDescentPlanner(
                unroll=self.model.unroll,
                action_dim=self.model.action_dim,
                action_masks=None,
                local_generator=self.local_gpu_generator,
                decode_unroll=self.model.decode_unroll,
                **self.cfg.planner,
            )
        else:
            raise ValueError(f"Unknown planner: {self.cfg.planner}")

    @torch.no_grad()
    def set_goal(self, goal_state):
        """
        goal_state:
        - rgb : [T, C, H, W] if not cfg.task_specification.obs_concat_channels with T >= tubelet_size_enc
        """
        assert goal_state is not None, "Goal state must be set first for gc agent."
        self.goal_state = goal_state.unsqueeze(0)
        self.goal_state_enc = self.model.encode(self.goal_state, act=False).detach()
        if self.cfg.planner.planning_objective.objective_type == "repr_dist":
            # Careful: unsqueeze applies to TensorDicts
            self.objective = ReprTargetDistMPCObjective(
                self.cfg, target_enc=self.goal_state_enc, **self.cfg.planner.planning_objective
            )
        elif self.cfg.planner.planning_objective.objective_type == "repr_l1":
            self.objective = ReprTargetDistL1MPCObjective(
                self.cfg, target_enc=self.goal_state_enc, **self.cfg.planner.planning_objective
            )
        elif self.cfg.planner.planning_objective.objective_type == "repr_sim":
            self.objective = ReprTargetCosMPCObjective(
                self.cfg, target_enc=self.goal_state_enc, **self.cfg.planner.planning_objective
            )
        else:
            raise ValueError(f"Unknown objective type: {self.cfg.planner.planning_objective.objective_type}")

        if self.planner is not None:
            self.planner.set_objective(self.objective)

    def plan(
        self,
        z,
        steps_left=None,
    ):
        """
        args:
            z: initial context state
            steps_left: number of steps left in the episode, used to adjust planning horizon.
        Stored variables:
            self.prev_elite_losses_mean: mean loss of the elite action sequence from the previous planning optimizer iteration.
            self.prev_elite_losses_std: std of the loss of the elite action sequence from the previous planning optimizer iteration.
        Returns:
            a: action to take, shape (1, action_dim)
        """
        planning_result = self.planner.plan(
            z,
            steps_left=steps_left,
        )
        self._prev_losses = planning_result.losses
        self._prev_elite_losses_mean = planning_result.prev_elite_losses_mean
        self._prev_elite_losses_std = planning_result.prev_elite_losses_std
        self._prev_pred_frames_over_iterations = planning_result.pred_frames_over_iterations
        self._predicted_best_encs_over_iterations = planning_result.predicted_best_encs_over_iterations
        return planning_result.actions

    def act(
        self,
        obs,
        steps_left=None,
    ):
        """
        Select an action by planning in the latent space of the world model.

        Args:
                obs (torch.Tensor): Observation from the environment.
                t0 (bool): Whether this is the first observation in the episode.
                eval_mode (bool): Whether to use the mean of the action distribution.
                task (int): Task index (only used for multi-task experiments).

        Make sure the models's encode() and unroll() function match: they interact only here:
        unroll() is used in self.plan()

        Returns:
                torch.Tensor: Action to take in the environment.
        """
        if self.cfg.task_specification.obs == "rgb":
            obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
        elif self.cfg.task_specification.obs == "rgb_state":
            obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
        z = self.model.encode(obs, act=True)
        a = self.plan(
            z,
            steps_left=steps_left,
        )
        return a.cpu()
