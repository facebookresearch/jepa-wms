import gym
import numpy as np

from evals.simu_env_planning.envs.pusht_env.pusht_env import PushTEnv
from evals.simu_env_planning.envs.wrappers.time_limit import TimeLimit


class PushTWrapper(gym.Wrapper):
    def __init__(self, env, cfg=None):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self.action_dim = self.env.action_space.shape[0]

    def sample_random_init_goal_states(self, seed):
        """
        Return two random states: one as the initial state and one as the goal state.
        """
        rs = np.random.RandomState(seed)

        def generate_state():
            if self.env.with_velocity:
                return np.array(
                    [
                        rs.randint(50, 450),
                        rs.randint(50, 450),
                        rs.randint(100, 400),
                        rs.randint(100, 400),
                        rs.randn() * 2 * np.pi - np.pi,
                        0,
                        0,  # agent velocities default 0
                    ]
                )
            else:
                return np.array(
                    [
                        rs.randint(50, 450),
                        rs.randint(50, 450),
                        rs.randint(100, 400),
                        rs.randint(100, 400),
                        rs.randn() * 2 * np.pi - np.pi,
                    ]
                )

        init_state = generate_state()
        goal_state = generate_state()

        return init_state, goal_state

    def update_env(self, env_info):
        self.env.shape = env_info["shape"]

    def eval_state(self, goal_state, cur_state):
        """
        Return True if the goal is reached
        [agent_x, agent_y, T_x, T_y, angle, agent_vx, agent_vy]
        """
        # if position difference is < 20, and angle difference < np.pi/9, then success
        pos_diff = np.linalg.norm(goal_state[:4] - cur_state[:4])
        angle_diff = np.abs(goal_state[4] - cur_state[4])
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        success = pos_diff < 20 and angle_diff < np.pi / 9
        state_dist = np.linalg.norm(goal_state - cur_state)
        return {
            "success": success,
            "state_dist": state_dist,
        }

    def prepare(self, seed, init_state, env_info=None):
        """
        Reset with controlled init_state
        obs: (H W C)
        state: (state_dim)
        """
        self.env.seed(seed)
        self.env.reset_to_state = init_state
        obs, info = self.reset()
        return obs, info

    def reset(self, **kwargs):
        obs, state = self.env.reset(**kwargs)
        info = {"state": state, "proprio": obs["proprio"]}
        # discard the obs['visual'], let the PixelWrapper manage this
        return state, info

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # discard the obs['visual'], let the PixelWrapper manage this
        state = self.env._get_obs()
        info["proprio"] = obs["proprio"]
        # trunc is None
        return state, reward, None, done, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)


def make_env(cfg, env_cls=None):
    if not cfg.task_specification.task.startswith("pusht-"):
        raise ValueError("Unknown task:", cfg.task_specification.task)
    env = PushTEnv(
        with_velocity=cfg.task_specification.env.with_velocity,
        with_target=cfg.task_specification.env.with_target,
        render_size=cfg.task_specification.img_size,
    )
    # kwargs from dino-wm codebase, env/__init__.py
    env = PushTWrapper(env, cfg)
    env = TimeLimit(env, max_episode_steps=cfg.task_specification.max_episode_steps)
    env.max_episode_steps = env._max_episode_steps
    return env
