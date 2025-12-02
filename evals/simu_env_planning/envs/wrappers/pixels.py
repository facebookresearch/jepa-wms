# The below code is inspired from TD-MPC2 https://github.com/nicklashansen/tdmpc2
# licensed under the MIT License

from collections import deque

import gym
import numpy as np
import torch


class PixelWrapper(gym.Wrapper):
    """
    Wrapper for pixel observations. Compatible with DMControl environments.
    """

    def __init__(self, cfg, env):
        super().__init__(env)
        self.cfg = cfg
        self.env = env
        self.proprio_env = env
        self.num_frames = cfg.task_specification.num_frames
        self.min_frames = self.num_frames
        self.min_proprios = cfg.task_specification.num_proprios
        self.obs_concat_channels = self.cfg.task_specification.obs_concat_channels
        render_size = cfg.task_specification.img_size
        shape = (
            (self.num_frames * 3, render_size, render_size)
            if cfg.task_specification.obs_concat_channels
            else (self.num_frames, 3, render_size, render_size)
        )
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=shape,
            dtype=np.uint8,
        )
        self._frames = deque([], maxlen=self.num_frames)
        self._proprios = deque([], maxlen=self.min_proprios)
        self._render_size = render_size

    def _get_obs(self):
        frame = self.env.render(mode="rgb_array", width=self._render_size, height=self._render_size)  # H, W, C
        frame = frame.transpose(2, 0, 1)  # H, W, C -> C, H, W
        self._frames.append(frame)
        if self.cfg.task_specification.obs_concat_channels:
            return torch.from_numpy(np.concatenate(self._frames))
        else:
            return torch.stack([torch.from_numpy(f.copy()) for f in self._frames])

    def get_proprios(self):
        return torch.stack([torch.tensor(proprio) for proprio in self._proprios])

    def reset(self, task_idx=None, *args, **kwargs):
        """
        same frame repeated self._frames.maxlen times
        """
        obs, info = self.env.reset(*args, **kwargs)
        for _ in range(self.min_frames):
            obs = self._get_obs()
        for _ in range(self.min_proprios):
            self._proprios.append(info["proprio"])
        info["proprio"] = self.get_proprios()
        return obs, info

    def reset_warmup(self, *args, **kwargs):
        """
        Reset environment and take self.min_frames step with zero action.
        """
        obs, info = self.reset(*args, **kwargs)
        zero_action = torch.zeros(self.env.action_space.shape)
        for _ in range(self.min_frames):
            obs, reward, done, truncated, new_info = self.step(zero_action)
        return obs, new_info

    def step(self, action, debug=False):
        _, reward, done, truncated, info = self.env.step(action)
        self._proprios.append(info["proprio"])
        info["proprio"] = self.get_proprios()
        return self._get_obs(), reward, done, truncated, info

    def prepare(self, seed, init_state, env_info=None):
        """
        info["state"] and info["proprio"] should have the same scale. The preprocessor
        only acts in the EncPredWM.obs() function.
        Here, we do not call self.reset() as it would undo what the underlying self.env.prepare()
        does, which is to set the environment to a specific state.
        """
        obs, info = self.env.prepare(seed, init_state, env_info=env_info)
        self.env.set_elapsed_steps(0)
        for _ in range(self.min_frames):
            obs = self._get_obs()
        for _ in range(self.min_proprios):
            self._proprios.append(info["proprio"])
        info["proprio"] = self.get_proprios()
        return obs, info

    def step_multiple(self, actions):
        obs_list = []
        rewards = []
        dones = []
        infos = []
        for action in actions:
            obs, reward, done, truncated, info = self.step(action)
            obs_list.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            if done:
                break
        return obs_list, rewards, dones, infos

    def rollout(self, seed, init_state, actions, env_info=None):
        """
        only returns np arrays of observations and states
        seed: int
        init_state: (state_dim, )
        actions: (T, action_dim)
        obses: dict (T, H, W, C)
        states: (T, D)
        """
        obs, info = self.prepare(seed, init_state, env_info=env_info)
        obses, rewards, dones, infos = self.step_multiple(actions)
        obses = torch.cat([obs.unsqueeze(0), torch.stack(obses)], dim=0)
        infos = [info] + infos
        return obses, infos
