import gym
import numpy as np

from evals.simu_env_planning.envs.pointmaze_env.maze_model import MazeEnv
from evals.simu_env_planning.envs.wrappers.time_limit import TimeLimit

STATE_RANGES = np.array(
    [
        [0.39318362, 3.2198412],
        [0.62660956, 3.2187355],
        [-5.2262554, 5.2262554],
        [-5.2262554, 5.2262554],
    ]
)


class PointMazeWrapper(gym.Wrapper):
    def __init__(self, env, cfg=None):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self.action_dim = self.env.action_space.shape[0]

    def sample_random_init_goal_states(self, seed):
        rs = np.random.RandomState(seed)

        def generate_state():
            valid = False
            while not valid:
                x = rs.uniform(0.5, 3.1)
                y = rs.uniform(0.5, 3.1)
                valid = ((0.5 <= x <= 1.1 or 2.5 <= x <= 3.1) and (0.5 <= y <= 3.1)) or (
                    (1.1 < x < 2.5) and (2.5 <= y <= 3.1)
                )
            state = np.array(
                [
                    x,
                    y,
                    rs.uniform(low=STATE_RANGES[2][0], high=STATE_RANGES[2][1]),
                    rs.uniform(low=STATE_RANGES[3][0], high=STATE_RANGES[3][1]),
                ]
            )
            return state

        init_state = generate_state()
        goal_state = generate_state()
        return init_state, goal_state

    def update_env(self, env_info):
        pass

    def eval_state(self, goal_state, cur_state):
        success = np.linalg.norm(goal_state[:2] - cur_state[:2]) < 0.5
        state_dist = np.linalg.norm(goal_state - cur_state)
        return {
            "success": success,
            "state_dist": state_dist,
        }

    def prepare(self, seed, init_state, env_info=None):
        self.env.prepare_for_render()
        self.env.seed(seed)
        self.env.set_init_state(init_state)
        obs, info = self.reset()
        return obs, info

    def reset(self, **kwargs):
        """
        obs: (H W C)
        state: (state_dim)
        """
        obs, state = self.env.reset()
        info = {"state": state, "proprio": obs["proprio"]}
        return obs, info

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info["proprio"] = info["state"]
        return obs, reward, None, done, info

    def render(self, *args, **kwargs):
        # IMPORTANT: Call _render_frame of the MazeModel, not render of some Mujoco underlying wrapper
        obs = self.env._render_frame(**kwargs)
        return obs


def make_env(cfg, env_cls=None):
    if not cfg.task_specification.task.startswith("maze-"):
        raise ValueError("Unknown task:", cfg.task_specification.task)
    # kwargs from dino-wm codebase, env/__init__.py
    kwargs = {
        "reward_type": "sparse",
        "reset_target": False,
        "ref_min_score": 23.85,
        "ref_max_score": 161.86,
        "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5",
    }
    env = MazeEnv(**kwargs)
    env = PointMazeWrapper(env, cfg)
    env = TimeLimit(env, max_episode_steps=cfg.task_specification.max_episode_steps)
    env.max_episode_steps = env._max_episode_steps
    return env
