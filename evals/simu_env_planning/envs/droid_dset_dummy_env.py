import os

import gym
import numpy as np

from evals.simu_env_planning.envs.wrappers.time_limit import TimeLimit


class DummyEnv:
    def __init__(self):
        self.spec = None
        self.env = None
        self._seed = 1

    def seed(self, seed=None):
        if seed is not None:
            self._seed = seed
        return self._seed

    @property
    def unwrapped(self):
        return self  # Return self as it's the base environment


class DroidDummyWrapper(gym.Wrapper):
    def __init__(self, env, cfg=None):
        super().__init__(env)
        # create dummy env class with None spec attr
        self.env = DummyEnv()
        self.cfg = cfg
        self.env.width = cfg.task_specification.img_size
        self.env.height = cfg.task_specification.img_size
        self.action_dim = 7
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)

    def reset(self, **kwargs):
        info = {}
        info["proprio"] = np.zeros(shape=(1, 1))
        info["state"] = np.zeros(shape=(1, 1))
        return np.zeros(shape=(1, 1)), info

    def eval_state(self, goal_state, cur_state):
        return {
            "success": 1.0,
            "state_dist": 0.0,
        }

    def prepare(self, seed, init_state, env_info=None):
        """
        Reset with controlled init_state
        obs: (H W C)
        state: (state_dim)
        """
        return self.reset()

    def step(self, action):
        info = {}
        info["proprio"] = np.zeros(shape=(1, 1))
        info["state"] = np.zeros(shape=(1, 1))
        return np.zeros(shape=(1, 1)), np.zeros(shape=(1, 1)), False, False, info

    def update_env(self, env_info):
        pass

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, *args, **kwargs):
        return np.random.normal(size=(self.env.height, self.env.width, 3))  # flip vertically


def make_env(cfg, env_cls=None):
    """
    Make Meta-World environment.
    """
    if not cfg.task_specification.task.startswith("droid-"):
        raise ValueError("Unknown task:", cfg.task_specification.task)
    env = None
    env = DroidDummyWrapper(env, cfg)
    env = TimeLimit(env, max_episode_steps=cfg.task_specification.max_episode_steps)
    env.max_episode_steps = env._max_episode_steps
    return env
