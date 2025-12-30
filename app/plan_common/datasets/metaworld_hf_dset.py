from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
from einops import rearrange

from datasets import load_dataset

from src.utils.logging import get_logger

from .traj_dset import TrajDataset, get_train_val_sliced

log = get_logger(__name__)


def decode_video_frames(video_bytes):
    """Decode MP4 bytes to numpy frames using imageio."""
    import io

    import imageio

    reader = imageio.get_reader(io.BytesIO(video_bytes), format="mp4")
    frames = [frame for frame in reader]
    reader.close()
    return np.stack(frames)


def decode_video_from_decoder(video_decoder):
    """Extract frames from a torchcodec VideoDecoder object.

    When torchcodec is installed, HuggingFace datasets auto-decodes
    video columns and returns VideoDecoder objects instead of bytes.
    """
    frames = []
    for i in range(len(video_decoder)):
        # VideoDecoder[i] returns tensor in (C, H, W) format
        frame = video_decoder[i]
        # Convert to (H, W, C) numpy array
        frame_np = frame.data.permute(1, 2, 0).numpy()
        frames.append(frame_np)
    return np.stack(frames)


class MetaworldHFDataset(TrajDataset):
    """Metaworld dataset loaded from HuggingFace parquet format with MP4 videos.

    Expected HF format (after conversion with first frame discarded):
    - video: 100 frames (indices 1-100 from original H5)
    - states: 100 states (indices 1-100 from original H5)
    - actions: 99 actions (indices 1-99 from original H5)
    - rewards: 99 rewards (indices 1-99 from original H5)

    After loading, we discard the last state to align with actions:
    - Final: 99 states, 99 actions, 99 rewards, 99 frames
    """

    def __init__(
        self,
        data_path: str,
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
        action_scale=1.0,
        filter_tasks: Optional[List[str]] = None,
        with_reward: bool = True,
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.normalize_action = normalize_action
        self.with_reward = with_reward

        # Load HuggingFace dataset from parquet files
        log.info(f"📂 Loading Metaworld HuggingFace dataset from {data_path}...")
        ds = load_dataset("parquet", data_dir=str(data_path), split="train")

        # Filter by task if specified
        if filter_tasks is not None:
            log.info(f"   Filtering for tasks {filter_tasks}...")
            ds = ds.filter(lambda x: x["task"] in filter_tasks)

        # Limit number of rollouts
        if n_rollout is not None:
            ds = ds.select(range(min(n_rollout, len(ds))))

        self.dataset = ds
        log.info(f"✅ Loaded {len(ds)} Metaworld rollouts")

        # Pre-load states, actions, rewards (lightweight, no video decoding yet)
        states = []
        actions = []
        proprio_states = []
        seq_lengths = []
        rewards = []

        for i in range(len(ds)):
            row = ds[i]
            state = np.array(row["states"])  # 100 states
            action = np.array(row["actions"])  # 99 actions
            proprio_state = state[:, :4]

            # Discard last state to align with actions (99 states, 99 actions)
            states.append(torch.tensor(state[:-1], dtype=torch.float32))
            actions.append(torch.tensor(action, dtype=torch.float32))
            proprio_states.append(torch.tensor(proprio_state[:-1], dtype=torch.float32))
            seq_lengths.append(len(action))

            if self.with_reward:
                reward = np.array(row["rewards"])  # 99 rewards
                rewards.append(torch.tensor(reward, dtype=torch.float32).unsqueeze(-1))

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
        """Load and decode video frames on demand."""
        row = self.dataset[idx]

        # Handle video decoding based on format
        video = row["video"]
        if isinstance(video, dict) and "bytes" in video:
            # Standard HF format: decode from bytes using imageio
            all_frames = decode_video_frames(video["bytes"])
        else:
            # torchcodec auto-decoded: video is a VideoDecoder object
            all_frames = decode_video_from_decoder(video)

        # No offset needed - video already starts at correct frame
        # Just index directly (frames 0-98 correspond to states 0-98)
        frame_data = torch.tensor(all_frames[frames], dtype=torch.float32)

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

        return obs, act, state, reward, {}

    def __getitem__(self, idx, **kwargs):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)


def load_metaworld_hf_slice_train_val(
    transform,
    n_rollout=50,
    data_path: str = "data/Metaworld/data",
    normalize_action=False,
    split_ratio=0.8,
    num_hist=0,
    num_pred=0,
    num_frames_val=None,
    frameskip=1,
    action_skip=1,
    traj_subset=True,
    filter_tasks=None,
    random_seed=42,
    with_reward=False,
    process_actions="concat",
):
    dset = MetaworldHFDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path,
        normalize_action=normalize_action,
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
