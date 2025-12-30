"""
DINO-WM-style plan evaluator that follows the original DINO-WM codebase logic exactly.

This module provides evaluation functionality that mimics the behavior of:
- evals/dino_wm_plan/plan.py (PlanWorkspace)
- evals/dino_wm_plan/planning/evaluator.py (PlanEvaluator)
- evals/dino_wm_plan/planning/cem.py (CEMPlanner)

To use this instead of PlanEvaluator, set `planner.use_dino_wm_planner: true`
in your config file.

Options:
- `model_kwargs.dino_wm_format: true`: Load checkpoint in DINO-WM format (with weights_only=False)
- `model_kwargs.load_dino_wm_modules: true`: Load entire module objects from checkpoint and
  build VWorldModel directly (instead of using VWorldModelAdapter). This provides exact
  compatibility with DINO-WM trained checkpoints.
"""

import os
import gym
import torch
import numpy as np
import hydra
from pathlib import Path
from einops import rearrange, repeat
from omegaconf import OmegaConf
from time import time

from evals.dino_wm_plan.env.venv import SubprocVectorEnv
from evals.dino_wm_plan.env.serial_vector_env import SerialVectorEnv
from evals.dino_wm_plan.preprocessor import Preprocessor
from evals.dino_wm_plan.planning.evaluator import PlanEvaluator as DinoWMPlanEvaluator
from evals.dino_wm_plan.planning.cem import CEMPlanner as DinoWMCEMPlanner
from evals.dino_wm_plan.planning.objectives import create_objective_fn
from evals.dino_wm_plan.utils import move_to_device

from src.utils.logging import get_logger

log = get_logger(__name__)


def load_dino_wm_dataset(cfg):
    """
    Load DINO-WM style dataset for a given task.

    This function loads the exact same dataset classes used in the DINO-WM codebase
    to ensure identical trajectory sampling behavior.

    Uses DWM_DATASET_DIR environment variable to find dataset paths, matching
    the DINO-WM codebase behavior.

    Args:
        cfg: Configuration with task_specification

    Returns:
        traj_dset: The trajectory dataset (validation split)
    """
    from evals.dino_wm_plan.datasets.img_transforms import default_transform

    # Get dataset directory from environment variable (same as DINO-WM codebase)
    dwm_dataset_dir = os.environ.get("DWM_DATASET_DIR", "/data/datasets")

    task = cfg.task_specification.task
    transform = default_transform(img_size=224)

    if task.startswith("pusht-"):
        from evals.dino_wm_plan.datasets.pusht_dset import load_pusht_slice_train_val

        data_path = os.path.join(dwm_dataset_dir, "pusht_noise")
        _, traj_dset = load_pusht_slice_train_val(
            transform=transform,
            n_rollout=None,
            data_path=data_path,
            normalize_action=True,
            split_ratio=0.9,
            num_hist=2,
            num_pred=1,
            frameskip=cfg.frameskip,
            with_velocity=cfg.task_specification.env.get("with_velocity", True),
        )
        log.info(f"Loaded DINO-WM PushT dataset from {data_path}")

    elif task.startswith("wall-"):
        from evals.dino_wm_plan.datasets.wall_dset import load_wall_slice_train_val

        data_path = os.path.join(dwm_dataset_dir, "wall_single")
        _, traj_dset = load_wall_slice_train_val(
            transform=transform,
            n_rollout=None,
            data_path=data_path,
            normalize_action=True,
            split_ratio=0.9,
            split_mode="random",
            num_hist=2,
            num_pred=1,
            frameskip=cfg.frameskip,
        )
        log.info(f"Loaded DINO-WM Wall dataset from {data_path}")

    elif task.startswith("maze-"):
        from evals.dino_wm_plan.datasets.point_maze_dset import load_point_maze_slice_train_val

        data_path = os.path.join(dwm_dataset_dir, "point_maze")
        _, traj_dset = load_point_maze_slice_train_val(
            transform=transform,
            n_rollout=None,
            data_path=data_path,
            normalize_action=True,
            split_ratio=0.9,
            num_hist=2,
            num_pred=1,
            frameskip=cfg.frameskip,
            traj_subset=True,
        )
        log.info(f"Loaded DINO-WM PointMaze dataset from {data_path}")

    else:
        raise ValueError(f"Unknown task for DINO-WM dataset: {task}")

    return traj_dset["valid"]


class VWorldModelAdapter(torch.nn.Module):
    """
    Adapter that provides a DINO-WM VWorldModel compatible interface but uses
    our model's actual forward logic internally.

    This is necessary because our model's predictor architecture is different from
    DINO-WM's and cannot be directly plugged into VWorldModel. Instead, we implement
    the same interface (rollout, encode_obs, etc.) but delegate to our model's methods.
    """

    def __init__(self, enc_pred_wm, preprocessor, tubelet_size=1, grid_size=14):
        """
        Args:
            enc_pred_wm: EncPredWM instance from the current codebase
            preprocessor: Preprocessor with transform, denormalize methods
            tubelet_size: Tubelet size for the model
            grid_size: Grid size for visual embeddings
        """
        super().__init__()
        self.enc_pred_wm = enc_pred_wm
        self.preprocessor = preprocessor
        self.tubelet_size = tubelet_size
        self.grid_size = grid_size
        self.device = enc_pred_wm.device

        # DINO-WM planning components expect these attributes
        self.decoder = enc_pred_wm.heads.get("image_head", None) if hasattr(enc_pred_wm, "heads") else None

    def to(self, device):
        self.device = device
        return super().to(device)

    def encode_obs(self, trans_obs):
        """
        Encode transformed observations into latent space.

        This mimics DINO-WM's VWorldModel.encode_obs() interface.

        Args:
            trans_obs: dict with 'visual' (B, T, C, H, W) and 'proprio' (B, T, D)
                      Already preprocessed/transformed observations (visual in [0, 1])

        Returns:
            z_obs: dict with 'visual' (B, T, ...) and 'proprio' (B, T, ...)
        """
        from tensordict.tensordict import TensorDict

        b, t, c, h, w = trans_obs["visual"].shape
        visual = trans_obs["visual"].to(self.device, dtype=torch.float32)
        proprio = trans_obs["proprio"].to(self.device, dtype=torch.float32)

        # Denormalize proprio (DINO-WM preprocessor normalizes proprio)
        proprio_denorm = self.preprocessor.denormalize_proprios(proprio.cpu())
        proprio_denorm = proprio_denorm.to(self.device)

        # Our encode() expects visual in [0, 255] uint8 and raw proprio
        z_obs_td = self.enc_pred_wm.encode(
            TensorDict(
                {
                    "visual": (visual * 255).to(torch.uint8),
                    "proprio": proprio_denorm,
                },
                device=self.device,
            ),
            act=False,
        )

        z_obs = {
            "visual": z_obs_td["visual"],
            "proprio": z_obs_td["proprio"],
        }
        return z_obs

    def rollout(self, obs_0, act, step_size=1):
        """
        Rollout the world model from initial observations with given actions.

        This mimics DINO-WM's VWorldModel.rollout() interface.

        Args:
            obs_0: dict with 'visual' (B, T_init, C, H, W) and 'proprio' (B, T_init, D)
                  Already preprocessed/transformed initial observations
            act: (B, T, action_dim) normalized actions to rollout
            step_size: Step size for rollout (typically tubelet_size)

        Returns:
            z_obses: dict with 'visual' (B, T_total, ...) and 'proprio' (B, T_total, ...)
            z: combined embeddings (for compatibility, same as z_obses)
        """
        from tensordict.tensordict import TensorDict

        # Encode initial observations
        z_init = self.encode_obs(obs_0)

        z_init_td = TensorDict(
            {
                "visual": z_init["visual"],
                "proprio": z_init["proprio"],
            },
            device=self.device,
        )

        # Convert actions from (B, T, A) to (T, B, A) for our unroll method
        action_for_unroll = rearrange(act, "b t a -> t b a")

        # Use our model's unroll method
        pred_z_td = self.enc_pred_wm.unroll(z_init_td, act_suffix=action_for_unroll)

        # Convert back to (B, T, ...) format
        z_obses = {
            "visual": rearrange(pred_z_td["visual"], "t b ... -> b t ..."),
            "proprio": rearrange(pred_z_td["proprio"], "t b ... -> b t ..."),
        }

        return z_obses, z_obses

    def decode_obs(self, z_obs):
        """
        Decode latent observations back to visual space.

        Args:
            z_obs: dict with 'visual' (B, T, ...) and 'proprio' (B, T, ...)

        Returns:
            obs: dict with 'visual' (B, T, C, H, W)
            diff: placeholder for compatibility
        """
        from tensordict.tensordict import TensorDict

        z_obs_td = TensorDict(
            {
                "visual": rearrange(z_obs["visual"], "b t ... -> t b ..."),
                "proprio": rearrange(z_obs["proprio"], "b t ... -> t b ..."),
            },
            device=self.device,
        )

        if hasattr(self.enc_pred_wm, "decode_unroll"):
            pred_frames = self.enc_pred_wm.decode_unroll(z_obs_td, batch=True)
            obs = {"visual": torch.from_numpy(pred_frames).to(self.device)}
        else:
            obs = {"visual": None}

        return obs, 0.0


def make_dino_wm_env(cfg, n_envs=1):
    """
    Create a DINO-WM style environment exactly as in evals/dino_wm_plan/plan.py.

    This function mimics the environment creation logic in plan.py lines 567-586.
    """
    # Import to trigger gym.envs.registration.register() calls
    import evals.dino_wm_plan.env  # noqa: F401

    # Determine env name and kwargs from task
    task = cfg.task_specification.task
    if task.startswith("pusht-"):
        env_name = "pusht"
        env_args = []
        env_kwargs = {
            "with_velocity": cfg.task_specification.env.get("with_velocity", True),
            "with_target": cfg.task_specification.env.get("with_target", True),
        }
    elif task.startswith("wall-"):
        env_name = "wall"
        env_args = []
        env_kwargs = {}
    elif task.startswith("maze-"):
        env_name = "point_maze"
        env_args = []
        env_kwargs = {}
    else:
        raise ValueError(f"Unknown task for DINO-WM env: {task}")

    # Use SerialVectorEnv for wall/deformable_env, SubprocVectorEnv for others
    # (same logic as in evals/dino_wm_plan/plan.py lines 567-586)
    if env_name == "wall" or env_name == "deformable_env":
        env = SerialVectorEnv(
            [
                gym.make(env_name, *env_args, **env_kwargs)
                for _ in range(n_envs)
            ]
        )
    else:
        env = SubprocVectorEnv(
            [
                lambda: gym.make(env_name, *env_args, **env_kwargs)
                for _ in range(n_envs)
            ]
        )

    return env


class DummyWandbRun:
    """Dummy wandb run for when wandb logging is disabled."""

    def __init__(self):
        self.mode = "disabled"

    def log(self, *args, **kwargs):
        pass

    def watch(self, *args, **kwargs):
        pass

    def config(self, *args, **kwargs):
        pass

    def finish(self):
        pass


class DinoWMPlanWorkspace:
    """
    Planning workspace that follows the exact logic of evals/dino_wm_plan/plan.py PlanWorkspace.

    This class wraps the DINO-WM planning components to provide a similar interface
    to our codebase's evaluation system while using the exact DINO-WM planning logic.
    """

    def __init__(
        self,
        cfg,
        agent,
        env,
        device,
        wandb_run=None,
    ):
        """
        Initialize the planning workspace.

        Args:
            cfg: Config with task_specification, planner settings, etc.
            agent: The agent containing the model and preprocessor
            env: DINO-WM style vectorized environment
            device: Device to run on
            wandb_run: Optional wandb run for logging
        """
        self.cfg = cfg
        self.agent = agent
        self.env = env
        self.device = device
        self.wandb_run = wandb_run if wandb_run is not None else DummyWandbRun()

        # Number of evaluations (batch size for vectorized env)
        self.n_evals = 1  # We run one episode at a time
        self.frameskip = cfg.frameskip
        self.goal_H = cfg.task_specification.goal_H
        self.goal_source = cfg.task_specification.goal_source
        self.action_dim = cfg.action_dim * self.frameskip  # DINO-WM uses frameskip-expanded action dim

        # Build VWorldModel from agent's model components (exactly as in DINO-WM codebase)
        self.wm = self._build_vworld_model(agent, cfg, device)

        # Create DINO-WM preprocessor from agent's preprocessor
        # Use the agent's preprocessor stats
        self.data_preprocessor = Preprocessor(
            action_mean=agent.preprocessor.action_mean,
            action_std=agent.preprocessor.action_std,
            state_mean=getattr(agent.preprocessor, 'state_mean', torch.zeros(1)),
            state_std=getattr(agent.preprocessor, 'state_std', torch.ones(1)),
            proprio_mean=agent.preprocessor.proprio_mean,
            proprio_std=agent.preprocessor.proprio_std,
            transform=agent.preprocessor.transform,
        )

        # Initialize objective function and logging
        self._init_objective_and_logging()

    def _build_vworld_model(self, agent, cfg, device):
        """
        Get or build a VWorldModel-compatible model from the agent.

        There are two modes:
        1. If `load_dino_wm_modules: true` in the config, init_module() has already loaded
           the VWorldModel using evals/dino_wm_plan/plan.py:load_model().
           We simply return the agent's model directly.
        2. Otherwise, we use VWorldModelAdapter which provides the same interface but
           delegates to our model's actual forward logic.

        Args:
            agent: Agent containing the model (either VWorldModel or EncPredWM wrapper around VideoWM)
            cfg: Configuration
            device: Device to run on

        Returns:
            VWorldModel or VWorldModelAdapter: DINO-WM compatible world model
        """
        from evals.dino_wm_plan.visual_world_model import VWorldModel

        # Check if the agent's model is already a VWorldModel (loaded via load_dino_wm_modules in init_module)
        if isinstance(agent.model, VWorldModel):
            log.info("Using VWorldModel directly from init_module (load_dino_wm_modules=True)")
            return agent.model

        # Otherwise, use VWorldModelAdapter (original behavior)
        enc_pred_wm = agent.model

        # Get model dimensions and config
        tubelet_size = enc_pred_wm.tubelet_size_enc
        grid_size = getattr(enc_pred_wm, 'grid_size', 14)

        # Build VWorldModelAdapter that wraps our model
        wm = VWorldModelAdapter(
            enc_pred_wm=enc_pred_wm,
            preprocessor=agent.preprocessor,
            tubelet_size=tubelet_size,
            grid_size=grid_size,
        )

        wm.to(device)

        log.info(f"Built VWorldModelAdapter from agent's model:")
        log.info(f"  tubelet_size={tubelet_size}, grid_size={grid_size}")
        log.info(f"  Using VWorldModelAdapter to delegate to model's actual forward logic")

        return wm

    def _init_objective_and_logging(self):
        """Initialize objective function and logging settings."""
        # Objective function (same as DINO-WM)
        self.objective_fn = create_objective_fn(
            alpha=self.cfg.planner.planning_objective.get("alpha", 0.1),
            mode="last",
        )

        self.log_filename = "logs.json"

    def prepare_targets_from_states(self, seeds, init_states, goal_states):
        """
        Prepare initial and goal observations from states.

        This mimics the prepare_targets logic in plan.py for random_state goal_source.

        Args:
            seeds: List of seeds for environment reset
            init_states: Initial states array (n_evals, state_dim)
            goal_states: Goal states array (n_evals, state_dim)
        """
        # Prepare initial observations
        obs_0, state_0 = self.env.prepare(seeds, init_states)
        obs_g, state_g = self.env.prepare(seeds, goal_states)

        # Add time dimension for tubelet_size (same as plan.py lines 249-253)
        tubelet_size = self.wm.tubelet_size
        for k in obs_0.keys():
            obs_0[k] = np.stack([obs_0[k]] * tubelet_size, axis=1)
            obs_g[k] = np.stack([obs_g[k]] * tubelet_size, axis=1)

        self.obs_0 = obs_0
        self.obs_g = obs_g
        self.state_0 = init_states
        self.state_g = goal_states
        self.gt_actions = None

    def prepare_targets_from_rollout(self, seeds, init_states, actions, env_info=None):
        """
        Prepare initial and goal observations by replaying actions.

        This mimics the prepare_targets logic in plan.py for dset/random_action goal_source.

        Args:
            seeds: List of seeds
            init_states: Initial states array (n_evals, state_dim)
            actions: Actions to replay (n_evals, T, action_dim)
            env_info: Optional environment info for updating env
        """
        if env_info is not None:
            self.env.update_env(env_info)

        # Denormalize actions for execution
        exec_actions = self.data_preprocessor.denormalize_actions(actions).numpy()

        # Rollout in environment (same as plan.py lines 292-304)
        rollout_obses, rollout_states = self.env.rollout(seeds, init_states, exec_actions)

        tubelet_size = self.wm.tubelet_size
        self.obs_0 = {
            key: arr[:, :tubelet_size]
            for key, arr in rollout_obses.items()
        }
        self.obs_g = {
            key: arr[:, -tubelet_size:]
            for key, arr in rollout_obses.items()
        }
        self.state_0 = init_states
        self.state_g = rollout_states[:, -1]

        # Convert actions to world model format (frameskip expansion)
        wm_actions = rearrange(actions, "b (t f) d -> b t (f d)", f=self.frameskip)
        self.gt_actions = wm_actions

    def create_evaluator(self):
        """Create the DINO-WM PlanEvaluator."""
        # Seeds for evaluation
        eval_seeds = [self.cfg.local_seed * n + 1 for n in range(self.n_evals)]

        self.evaluator = DinoWMPlanEvaluator(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            state_0=self.state_0,
            state_g=self.state_g,
            env=self.env,
            wm=self.wm,
            frameskip=self.frameskip,
            seed=eval_seeds,
            preprocessor=self.data_preprocessor,
            n_plot_samples=self.cfg.planner.get("n_plot_samples", 1),
        )
        return self.evaluator

    def create_planner(self):
        """Create the DINO-WM CEMPlanner."""
        # Create planner config dict to match hydra.utils.instantiate format
        planner_cfg = {
            "horizon": self.goal_H,
            "topk": self.cfg.planner.num_elites,
            "num_samples": self.cfg.planner.num_samples,
            "var_scale": self.cfg.planner.var_scale,
            "opt_steps": self.cfg.planner.iterations,
            "eval_every": self.cfg.planner.get("eval_every", 10),
        }

        self.planner = DinoWMCEMPlanner(
            horizon=planner_cfg["horizon"],
            topk=planner_cfg["topk"],
            num_samples=planner_cfg["num_samples"],
            var_scale=planner_cfg["var_scale"],
            opt_steps=planner_cfg["opt_steps"],
            eval_every=planner_cfg["eval_every"],
            wm=self.wm,
            env=self.env,
            action_dim=self.action_dim,
            objective_fn=self.objective_fn,
            preprocessor=self.data_preprocessor,
            evaluator=self.evaluator,
            wandb_run=self.wandb_run,
            log_filename=self.log_filename,
        )
        return self.planner

    def perform_planning(self):
        """
        Perform planning and evaluation.

        This mimics the perform_planning method in plan.py.

        Returns:
            logs: Dictionary of evaluation metrics
        """
        # Plan actions
        actions, action_len = self.planner.plan(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            actions=None,  # No initialization
        )

        # Evaluate planned actions
        logs, successes, _, _ = self.evaluator.eval_actions(
            actions.detach(),
            action_len,
            save_video=True,
            filename="output_final",
        )

        logs = {f"final_eval/{k}": v for k, v in logs.items()}
        self.wandb_run.log(logs)

        return logs, successes


class DinoWMPlanEvaluatorWrapper:
    """
    Wrapper that provides the same interface as our PlanEvaluator but uses
    the exact DINO-WM planning logic internally.

    This is the main entry point for DINO-WM style evaluation.
    """

    def __init__(self, cfg, agent, dino_wm_env=None, dino_wm_dset=None):
        """
        Initialize the evaluator.

        Args:
            cfg: Configuration with planner settings, task_specification, etc.
            agent: Agent containing the model and preprocessor
            dino_wm_env: Pre-created DINO-WM vectorized environment
            dino_wm_dset: Pre-loaded DINO-WM dataset (validation split)
        """
        self.cfg = cfg
        self.agent = agent
        self.dino_wm_env = dino_wm_env
        self.dino_wm_dset = dino_wm_dset
        # Get device from model parameters (VWorldModel doesn't have .device attribute)
        self.device = next(agent.model.parameters()).device

    def eval(self, cfg, agent, env, task_idx=-1, ep=0):
        """
        Evaluate a single episode using DINO-WM planning logic.

        This method follows the exact pattern from plan.py but adapts it
        to work with our evaluation interface.

        Args:
            cfg: Configuration
            agent: Agent with model and preprocessor
            env: Our codebase's environment (not used when dino_wm_env is available)
            task_idx: Task index for multi-task settings
            ep: Episode number

        Returns:
            Tuple of evaluation metrics matching PlanEvaluator.eval() return signature
        """
        work_dir = cfg.work_dir / cfg.tasks[task_idx] / f"ep_{ep}"
        os.makedirs(work_dir, exist_ok=True)

        # Change to work directory for saving outputs
        original_dir = os.getcwd()
        os.chdir(work_dir)

        try:
            # Episode seed
            ep_seed = (cfg.local_seed * cfg.local_seed + ep * cfg.local_seed) % (2**32 - 2)
            seeds = [ep_seed]

            # Create or use DINO-WM environment
            if self.dino_wm_env is None:
                dino_wm_env = make_dino_wm_env(cfg, n_envs=1)
            else:
                dino_wm_env = self.dino_wm_env

            # Create planning workspace
            workspace = DinoWMPlanWorkspace(
                cfg=cfg,
                agent=agent,
                env=dino_wm_env,
                device=self.device,
                wandb_run=None,
            )

            # Prepare targets based on goal source
            if cfg.task_specification.goal_source == "random_state":
                # Sample random init and goal states from environment
                init_states, goal_states = dino_wm_env.sample_random_init_goal_states(seeds)
                workspace.prepare_targets_from_states(seeds, init_states, goal_states)

            elif cfg.task_specification.goal_source in ["dset", "random_action"]:
                # Sample trajectory from dataset (similar to plan.py)
                traj_len = cfg.frameskip * cfg.task_specification.goal_H + workspace.wm.tubelet_size
                observations, states, actions, env_info = self._sample_traj_segment_from_dset(
                    cfg, agent, traj_len
                )

                init_states = np.array([states[0]])
                if cfg.task_specification.goal_source == "random_action":
                    actions = torch.randn_like(actions)

                workspace.prepare_targets_from_rollout(
                    seeds,
                    init_states,
                    actions.unsqueeze(0),  # Add batch dim
                    env_info=[env_info],
                )

            else:
                raise ValueError(f"Unknown goal source: {cfg.task_specification.goal_source}")

            # Create evaluator and planner
            workspace.create_evaluator()
            workspace.create_planner()

            # Perform planning
            log.info("Starting DINO-WM style planning...")
            plan_start_time = time()
            logs, successes = workspace.perform_planning()
            plan_end_time = time()
            log.info(f"Planning completed in {plan_end_time - plan_start_time:.2f} seconds")

            # Extract metrics
            success = successes[0] if len(successes) > 0 else False
            success_rate = logs.get("final_eval/success_rate", 0.0)
            state_dist = logs.get("final_eval/mean_state_dist", -1.0)

            log.info(f"Episode {ep}: success={success}, success_rate={success_rate}")

            # Return metrics in the same format as PlanEvaluator.eval()
            return (
                1,  # expert_success (always 1 for non-expert goal sources)
                success,
                0.0,  # ep_reward (not tracked in DINO-WM style)
                float(success),  # success_dist
                state_dist,  # end_distance
                -1.0,  # end_distance_xyz
                -1.0,  # end_distance_orientation
                -1.0,  # end_distance_closure
                state_dist,  # state_dist
                -1.0,  # total_lpips
                -1.0,  # total_emb_l2
            )

        finally:
            os.chdir(original_dir)

    def _sample_traj_segment_from_dset(self, cfg, agent, traj_len):
        """
        Sample a trajectory segment from the dataset.

        This mimics sample_traj_segment_from_dset in plan.py.

        Args:
            cfg: Configuration
            agent: Agent with dataset
            traj_len: Required trajectory length

        Returns:
            observations, states, actions, env_info
        """
        # Check if we should use DINO-WM datasets (pre-loaded in __init__)
        use_dino_wm_dsets = cfg.task_specification.get("use_dino_wm_dsets", False)

        if use_dino_wm_dsets and self.dino_wm_dset is not None:
            # Use DINO-WM dataset (pre-loaded at initialization)
            dset = self.dino_wm_dset

            # Check if any trajectory is long enough
            valid_traj_indices = [
                i for i in range(len(dset))
                if dset.get_seq_length(i) >= traj_len
            ]
            if len(valid_traj_indices) == 0:
                raise ValueError("No trajectory in the DINO-WM dataset is long enough.")

            # Sample a valid trajectory (using random sampling like plan.py)
            import random
            max_offset = -1
            while max_offset < 0:
                traj_id = random.randint(0, len(dset) - 1)
                # DINO-WM dataset returns (obs, act, state, env_info)
                obs, act, state, e_info = dset[traj_id]
                max_offset = obs["visual"].shape[0] - traj_len

            if isinstance(state, torch.Tensor):
                state = state.numpy()
            offset = random.randint(0, max_offset)

            log.info(f"Sampled traj (DINO-WM dset): traj id: {traj_id}  Offset: {offset}")

            obs = {
                key: arr[offset : offset + traj_len]
                for key, arr in obs.items()
            }
            state = state[offset : offset + traj_len]
            act = act[offset : offset + cfg.frameskip * cfg.task_specification.goal_H]

            return obs, state, act, e_info
        else:
            # Use agent's dataset (original behavior)
            dset = agent.dset

            # Check if any trajectory is long enough
            valid_traj_indices = [
                i for i in range(len(dset))
                if dset.get_seq_length(i) >= traj_len
            ]
            if len(valid_traj_indices) == 0:
                raise ValueError("No trajectory in the dataset is long enough.")

            # Sample a valid trajectory
            max_offset = -1
            while max_offset < 0:
                traj_id = torch.randint(
                    low=0, high=len(dset), size=(1,), generator=agent.local_generator
                ).item()
                obs, act, state, reward, e_info = dset[traj_id]
                max_offset = obs["visual"].shape[0] - traj_len

            state = state.numpy()
            offset = torch.randint(
                low=0, high=max_offset + 1, size=(1,), generator=agent.local_generator
            ).item()

            log.info(f"Sampled traj: traj id: {traj_id}  Offset: {offset}")

            obs = {
                key: arr[offset : offset + traj_len]
                for key, arr in obs.items()
            }
            state = state[offset : offset + traj_len]
            act = act[offset : offset + cfg.frameskip * cfg.task_specification.goal_H]

            return obs, state, act, e_info
