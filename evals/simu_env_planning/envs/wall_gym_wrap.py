import gym
import numpy as np
from torchvision import transforms

from evals.simu_env_planning.envs.wrappers.time_limit import TimeLimit

from .wall_env.data.wall_utils import generate_wall_layouts
from .wall_env.envs.wall import DotWall, WallDatasetConfig

ENV_ACTION_DIM = 2
STATE_RANGES = np.array([[16.6840, 46.9885], [4.0083, 25.2532]])
DEFAULT_CFG = WallDatasetConfig(
    action_angle_noise=0.2,
    action_step_mean=1.0,
    action_step_std=0.4,
    action_lower_bd=0.2,
    action_upper_bd=1.8,
    batch_size=64,
    device="cuda",
    dot_std=1.7,
    border_wall_loc=5,
    fix_wall_batch_k=None,
    fix_wall=True,
    fix_door_location=30,
    fix_wall_location=32,
    exclude_wall_train="",
    exclude_door_train="",
    only_wall_val="",
    only_door_val="",
    wall_padding=20,
    door_padding=10,
    wall_width=6,
    door_space=4,
    num_train_layouts=-1,
    img_size=65,
    max_step=1,
    n_steps=17,
    n_steps_reduce_factor=1,
    size=20000,
    val_size=10000,
    train=True,
    repeat_actions=1,
)
resize_transform = transforms.Resize((224, 224))
TRANSFORM = resize_transform


class WallEnvWrapper(gym.Wrapper):
    def __init__(self, env, cfg=None):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self.action_dim = ENV_ACTION_DIM
        self.transform = TRANSFORM

    def eval_state(self, goal_state, cur_state):
        success = np.linalg.norm(goal_state[:2] - cur_state[:2]) < 4.5
        state_dist = np.linalg.norm(goal_state - cur_state)
        return {
            "success": success,
            "state_dist": state_dist,
        }

    def sample_random_init_goal_states(self, seed):
        return self.env.generate_random_state(seed)

    def update_env(self, env_info):
        self.env.wall_config.fix_door_location = env_info["fix_door_location"].item()
        self.env.wall_config.fix_wall_location = env_info["fix_wall_location"].item()
        layouts, other_layouts = generate_wall_layouts(self.env.wall_config)
        self.env.layouts = layouts
        self.env.wall_x, self.env.hole_y = self.env._generate_wall()

    def prepare(self, seed, init_state, env_info=None):
        self.env.seed(seed)
        self.env.set_init_state(init_state)
        obs, info = self.reset()
        return obs, info

    def reset(self, seed=0, **kwargs):
        # init_state, _ = self.sample_random_init_goal_states(seed)
        # self.env.set_init_state(init_state)
        obs, state = self.env.reset(**kwargs)
        obs["visual"] = self.transform(obs["visual"])
        obs["visual"] = obs["visual"].permute(1, 2, 0)
        info = {"state": state, "proprio": obs["proprio"]}
        return obs, info

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs["visual"] = self.transform(obs["visual"])
        obs["visual"] = obs["visual"].permute(1, 2, 0)
        info["state"] = info["state"].float()
        info["proprio"] = info["state"]
        return obs, reward, None, done, info

    def render(self, *args, **kwargs):
        visual = self.env.render(**kwargs)
        visual = self.transform(visual).permute(1, 2, 0).numpy()  # H, W, C
        return visual


def make_env(
    cfg,
    env_cls=None,
    rng=42,
    wall_config=DEFAULT_CFG,
    fix_wall=True,
    cross_wall=False,
    fix_wall_location=32,
    fix_door_location=10,
    device="cpu",
    **kwargs,
):
    if not cfg.task_specification.task.startswith("wall-"):
        raise ValueError("Unknown task:", cfg.task_specification.task)
    # kwargs from dino-wm codebase, env/__init__.py
    env = DotWall(
        rng,
        wall_config,
        fix_wall,
        cross_wall,
        fix_wall_location=fix_wall_location,
        fix_door_location=fix_door_location,
        device=device,
        **kwargs,
    )
    # rng=cfg.rng, wall_config=cfg.wall_config, fix_wall=cfg.fix_wall,
    #   cross_wall=cfg.cross_wall, fix_wall_location=cfg.fix_wall_location,
    #   fix_door_location=cfg.fix_door_location, device=cfg.device)
    env = WallEnvWrapper(env, cfg)
    env = TimeLimit(env, max_episode_steps=cfg.task_specification.max_episode_steps)
    env.max_episode_steps = env._max_episode_steps
    return env
