import os
from pathlib import Path
from typing import Callable, List, Optional

import h5py
import numpy as np
import pandas as pd
import torch
from einops import rearrange

from .traj_dset import TrajDataset, get_train_val_sliced


class MetaworldDataset(TrajDataset):
    def __init__(
        self,
        # data_path: str = "data/metaworld",
        data_paths: List[str],
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
        action_scale=1.0,
        filter_first_episodes=100,
        filter_tasks=None,
        with_reward=True,
    ):
        # self.data_path = Path(data_path)
        self.data_paths = [Path(data_path) for data_path in data_paths]
        self.transform = transform
        self.normalize_action = normalize_action
        self.with_reward = with_reward
        samples = []
        for data_path in self.data_paths:
            samples.extend(list(pd.read_csv(data_path, header=None, delimiter=" ").values[:, 0]))
        # samples = list(pd.read_csv(data_path, header=None, delimiter=' ').values[:, 0])
        if filter_tasks is not None:
            filtered_paths = []
            print(f"Filtering for tasks {filter_tasks}...")
            for ep_path in samples:
                ep_dir = Path(ep_path).parent.name
                ep_task = ep_dir.split("mw-")[1].split("_")[0]
                if ep_task in filter_tasks:
                    filtered_paths.append(ep_path)
            samples = filtered_paths
        if filter_first_episodes is not None:
            filtered_paths = []
            print(f"Filtering for first {filter_first_episodes} episodes of each task...")
            for ep_path in samples:
                ep_nb = ep_path.split("/")[-1]
                ep_nb = float(ep_nb.removeprefix("ep").removesuffix(".h5"))
                if ep_nb < filter_first_episodes:
                    filtered_paths.append(ep_path)
            samples = filtered_paths
        if n_rollout:
            n = n_rollout
        else:
            n = len(samples)

        self.samples = samples[:n]
        print(f"Loaded {n} rollouts")

        states = []
        actions = []
        proprio_states = []
        seq_lengths = []
        rewards = []

        for path in self.samples:
            trajectory = h5py.File(path)
            state = np.array(trajectory["state"])
            action = np.array(trajectory["action"])
            proprio_state = np.array(trajectory["state"])[:, :4]
            if self.with_reward:
                reward = np.array(trajectory["reward"])
            if "data/Metaworld/h5folder/" in path:
                # discard first state and action because not same env seed as rest of the traj
                # discard last state resulting from last action explicitly, although it was discarded
                # implicitly by get_seq_length in __getitem__
                states.append(torch.tensor(state)[1:-1])
                actions.append(torch.tensor(action)[1:])
                proprio_states.append(torch.tensor(proprio_state)[1:-1])
                if self.with_reward:
                    rewards.append(torch.tensor(reward)[1:])
                seq_lengths.append(len(action) - 1)
            else:
                states.append(torch.tensor(state))
                actions.append(torch.tensor(action))
                proprio_states.append(torch.tensor(proprio_state))
                if self.with_reward:
                    rewards.append(torch.tensor(reward))
                seq_lengths.append(len(action))

        self.states = torch.stack(states)
        self.actions = torch.stack(actions)
        self.proprios = torch.stack(proprio_states)
        self.seq_lengths = torch.tensor(seq_lengths)
        if self.with_reward:
            self.rewards = torch.stack(rewards)
        else:
            self.rewards = None

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]

        if normalize_action:
            self.action_mean, self.action_std = self.get_data_mean_std(self.actions, self.seq_lengths)
            self.state_mean, self.state_std = self.get_data_mean_std(self.states, self.seq_lengths)
            self.proprio_mean, self.proprio_std = self.get_data_mean_std(self.proprios, self.seq_lengths)
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std

    def get_data_mean_std(self, data, traj_lengths):
        all_data = []
        for traj in range(len(traj_lengths)):
            traj_len = traj_lengths[traj]
            traj_data = data[traj, :traj_len]
            all_data.append(traj_data)
        all_data = torch.vstack(all_data)
        data_mean = torch.mean(all_data, dim=0)
        data_std = torch.std(all_data, dim=0)
        return data_mean, data_std

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_frames(self, idx, frames):
        """
        Metaworld env.reset() function is not sufficient to take into account the new rand_vec
        that is randomly reinitialized. We need to step a first action in the env to make the new goal and init position
        happen. Hence we remove the first observation and action of the dataset here with a shift of 1.
        """
        path = self.samples[idx]
        trajectory = h5py.File(path)
        # frame_data = torch.tensor(trajectory['obs'][frames], dtype=torch.float32)
        if "data/Metaworld/h5folder/" in path:
            frame_data = torch.tensor(trajectory["obs"][np.array(frames) + 1], dtype=torch.float32)
        else:
            frame_data = torch.tensor(trajectory["obs"][frames], dtype=torch.float32)
        proprio = self.proprios[idx, frames]
        act = self.actions[idx, frames]
        state = self.states[idx, frames]
        frame_data = frame_data / 255.0
        frame_data = rearrange(frame_data, "T H W C -> T C H W")
        if self.transform:
            frame_data = self.transform(frame_data)
        obs = {"visual": frame_data, "proprio": proprio}
        if self.with_reward:
            reward = self.rewards[idx, frames]
        else:
            reward = None
        return obs, act, state, reward, {}  # env_info

    def __getitem__(self, idx, **kwargs):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)


def load_metaworld_slice_train_val(
    transform,
    n_rollout=50,
    data_paths=["/data/datasets/metaworld"],
    normalize_action=False,
    split_ratio=0.8,
    num_hist=0,
    num_pred=0,
    num_frames_val=None,
    frameskip=1,
    action_skip=1,
    traj_subset=True,
    filter_first_episodes=None,
    filter_tasks=None,
    random_seed=42,
    with_reward=False,
    process_actions="concat",
):
    dset = MetaworldDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_paths=data_paths,
        normalize_action=normalize_action,
        filter_first_episodes=filter_first_episodes,
        filter_tasks=filter_tasks,
        with_reward=with_reward,
    )
    dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
        traj_dataset=dset,
        train_fraction=split_ratio,
        num_frames=num_hist + num_pred,
        frameskip=frameskip,
        action_skip=action_skip,
        traj_subset=traj_subset,
        random_seed=random_seed,
        num_frames_val=num_frames_val,
        process_actions=process_actions,
    )
    datasets = {}
    datasets["train"] = train_slices
    datasets["valid"] = val_slices
    traj_dset = {}
    traj_dset["train"] = dset_train
    traj_dset["valid"] = dset_val
    return datasets, traj_dset
