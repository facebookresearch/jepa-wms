# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import fnmatch
import hashlib
import json
import os
from logging import getLogger
from math import ceil
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import decord
import h5py
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.utils.data
from decord import VideoReader, cpu
from einops import rearrange, repeat
from huggingface_hub import HfApi, hf_hub_download
from scipy.spatial.transform import Rotation
from tqdm import tqdm

_GLOBAL_SEED = 0
logger = getLogger()

decord.bridge.set_bridge("native")


def get_json(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding JSON in file: {filename}")
            except Exception as e:
                print(f"An unexpected error occurred while processing {filename}: {e}")


class DROIDVideoDataset(torch.utils.data.Dataset):
    """Video classification dataset."""

    def __init__(
        self,
        data_path,
        camera_views=["wrist_mp4_path"],
        frameskip=1,
        action_skip=1,
        frames_per_clip=16,
        fps=5,
        transform=None,
        camera_frame=False,
        normalize_action=False,
        mpk_dset: bool = False,
        mpk_manifest_patterns: Sequence[str] = None,
        droid_to_rcasa_action_format: int = 1,
        local: bool = True,
        seed: int = None,
        droid_fraction: float = 1.0,
    ):
        self.data_path = data_path
        self.frames_per_clip = frames_per_clip
        self.frameskip = frameskip
        self.action_skip = action_skip
        self.fps = fps
        self.transform = transform
        self.normalize_action = normalize_action
        self.camera_frame = camera_frame
        self.mpk_dset = mpk_dset
        self.mpk_manifest_patterns = mpk_manifest_patterns
        self.droid_to_rcasa_action_format = droid_to_rcasa_action_format
        self.local = local
        self.seed = seed
        # Same sample-level across workers because same self.rng pickled across workers
        # This randomness only affects clip slicing and camera viewpoint sampling.
        # (a torch.Generator() is not picklable)
        self.rng = np.random.RandomState(seed)

        if VideoReader is None:
            raise ImportError('Unable to import "decord" which is required to read videos.')

        # Camera views
        # ---
        # wrist camera view
        # left camera view
        # right camera view
        self.camera_views = camera_views
        logger.info(f"Using DROID with camera views: {self.camera_views}")
        if self.mpk_dset:
            self._repo_id = "facebook/robotics-world-models"
            self.hf_api = HfApi()
            self._action_type = "delta-state"
            self.samples = self._load()
            num_samples_stored = len(self.samples) if normalize_action else 1
            debug = False
        else:
            self.h5_name = "trajectory.h5"
            self.samples = list(pd.read_csv(data_path, header=None, delimiter=" ").values[:, 0])
            num_samples_stored = 50 if normalize_action else 1
            debug = False

        # Apply dataset fraction slicing
        if droid_fraction < 1.0:
            original_len = len(self.samples)
            num_samples = max(1, int(original_len * droid_fraction))
            self.samples = self.samples[:num_samples]
            logger.info(f"Slicing dataset from {original_len} to {num_samples} samples ({droid_fraction*100:.1f}%)")
        else:
            logger.info(f"Not slicing DROID dataset, using {len(self.samples)} samples, 100% of video paths")
        # Taken from other custom datasets
        states = []
        actions = []
        proprio_states = []
        seq_lengths = []
        for i in tqdm(range(num_samples_stored), desc=f"Loading {num_samples_stored} DROID eps to compute mean/std"):
            buffer, action, state, _ = self.__getitem__(i, debug=debug)
            states.append(torch.tensor(state))
            actions.append(torch.tensor(action))
            proprio_states.append(torch.tensor(state))
            seq_lengths.append(len(state))
        self.states = torch.stack(states)
        self.actions = torch.stack(actions)
        self.proprios = torch.stack(proprio_states)
        self.seq_lengths = torch.tensor(seq_lengths)
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

    def __getitem__(self, idx, debug=False, **kwargs):
        path = self.samples[idx]

        # -- keep trying to load videos until you find a valid sample
        loaded_video = False
        while not loaded_video:
            if debug:
                if self.mpk_dset:
                    buffer, actions, states, extrinsics, indices = self.loadvideo_hf(path)
                else:
                    buffer, actions, states, extrinsics, indices = self.loadvideo_decord(path)
                loaded_video = True
            else:
                try:
                    if self.mpk_dset:
                        buffer, actions, states, extrinsics, indices = self.loadvideo_hf(path)
                    else:
                        buffer, actions, states, extrinsics, indices = self.loadvideo_decord(path)
                    loaded_video = True
                except Exception as e:
                    logger.info(f"Ecountered exception when loading video {path=} {e=}")
                    loaded_video = False
                    idx = self.rng.randint(0, self.__len__())
                    path = self.samples[idx]
        obs = {
            "visual": buffer,
            "proprio": states,
        }
        if self.droid_to_rcasa_action_format > 1:
            actions = self.repeat_divide_action(actions, act_repeat=self.droid_to_rcasa_action_format)
        # pad actions with dummy last action so that it has the same length as obs
        if len(actions) < len(states):
            actions = np.concatenate([actions, np.zeros((1, actions.shape[-1]))], axis=0)
        # buffer : T C H W
        return obs, actions, states, torch.tensor(0.0)
        # return buffer, actions, states, extrinsics, indices

    def repeat_divide_action(self, action, act_repeat=5):
        """
        Action repeat and divide.
        Used when a model is used to concatenated "small" actions and
            we want to feed it DROID actions that are big and 7-dimensional.
        Input: T A
        Returns: T F*A
        """
        return repeat(action, "t a -> t (f a)", f=act_repeat) / float(act_repeat)

    def transform_frame(self, poses, extrinsics):
        gripper = poses[:, -1:]
        poses = poses[:, :-1]

        def pose_to_transform(pose):
            trans = pose[:3]  # shape [3]
            theta = pose[3:6]  # euler angles, shape [3]
            Rot = Rotation.from_euler("xyz", theta, degrees=False).as_matrix()
            T = np.eye(4)
            T[:3, :3] = Rot
            T[:3, 3] = trans
            return T

        def transform_to_pose(transform):
            trans = transform[:3, 3]
            Rot = transform[:3, :3]
            angle = Rotation.from_matrix(Rot).as_euler("xyz", degrees=False)
            return np.concatenate([trans, angle], axis=0)

        new_pose = []
        for p, e in zip(poses, extrinsics):
            p_transform = pose_to_transform(p)
            e_transform = pose_to_transform(e)
            new_pose_transform = np.linalg.inv(e_transform) @ p_transform
            new_pose += [transform_to_pose(new_pose_transform)]
        new_pose = np.stack(new_pose, axis=0)

        return np.concatenate([new_pose, gripper], axis=1)

    def _load(self):
        if self.local:
            # Use local files
            paths = []
            for pattern in self.mpk_manifest_patterns:
                # Remove leading '**/' if present, since Path.glob expects relative patterns
                cleaned_pattern = pattern.lstrip("/")
                # Use glob with the full pattern, relative to data_path
                found = list(Path(self.data_path).glob(cleaned_pattern))
                if not found:
                    print(f"Warning: No files found for pattern {cleaned_pattern}")
                paths.extend(found)
            return [str(p) for p in paths]
        else:
            # Create a cache file path based on repo_id and patterns
            cache_dir = Path("mpk_dataset_cache")
            cache_dir.mkdir(exist_ok=True)

            # Create a unique filename based on repo_id and patterns
            patterns_str = "_".join(self.mpk_manifest_patterns).replace("/", "_").replace("*", "star")
            patterns_str = str(hashlib.md5(patterns_str.encode()).hexdigest())
            repo_str = self._repo_id.replace("/", "_")
            cache_file = cache_dir / f"{repo_str}_{patterns_str}_files.json"

            # Check if we're in distributed mode
            is_distributed = dist.is_initialized()
            is_main_process = not is_distributed or dist.get_rank() == 0
            if is_distributed:
                if is_main_process:
                    # Only rank 0 fetches from HuggingFace and saves to cache
                    logger.info(f"Rank 0: Fetching file list from HuggingFace repo {self._repo_id}")
                    files = self.hf_api.list_repo_files(repo_id=self._repo_id, repo_type="model")
                    paths = []
                    for pattern in self.mpk_manifest_patterns:
                        paths.extend(fnmatch.filter(files, pattern))

                    # Save to cache file
                    with open(cache_file, "w") as f:
                        json.dump(paths, f)
                    logger.info(f"Rank 0: Saved {len(paths)} file paths to cache: {cache_file}")

                # Synchronize all processes
                dist.barrier()
                # dist.barrier(device_ids=[dist.get_local_rank()])

                # All ranks (including rank 0) now read from the cache file
                logger.info(f"Rank {dist.get_rank()}:  paths from cache: {cache_file}")
                with open(cache_file, "r") as f:
                    paths = json.load(f)
            else:
                # Non-distributed mode: check cache first, then fetch if needed
                if cache_file.exists():
                    logger.info(f"Loading file paths from existing cache: {cache_file}")
                    with open(cache_file, "r") as f:
                        paths = json.load(f)
                else:
                    logger.info(f"Fetching file list from HuggingFace repo {self._repo_id}")
                    files = self.hf_api.list_repo_files(repo_id=self._repo_id, repo_type="model")
                    paths = []
                    for pattern in self.mpk_manifest_patterns:
                        paths.extend(fnmatch.filter(files, pattern))

                    # Save to cache for future use
                    with open(cache_file, "w") as f:
                        json.dump(paths, f)
                    logger.info(f"Saved {len(paths)} file paths to cache: {cache_file}")
            return paths

    def loadvideo_hf(self, path):
        """
        Returns:
            buffer: torch.Tensor of shape [T, C, H, W] with video frames
            actions: np.ndarray of shape [T-1, 7] with robot actions
            states: np.ndarray of shape [T, 7] with robot states
            extrinsics: None (not used in this dataset)
            indices: np.ndarray of shape [T] with sampled frame indices
        """
        if self.local:
            local_path = path
        else:
            local_path = hf_hub_download(repo_id="facebook/robotics-world-models", filename=path)
        trajectory = h5py.File(local_path)
        camera_view = self.camera_views[self.rng.randint(0, len(self.camera_views))]
        states = np.concatenate(
            [
                np.array(trajectory["episode_data"]["observation"]["cartesian_position"]),
                np.array(trajectory["episode_data"]["observation"]["gripper_position"])[:, None],
            ],
            axis=1,
        )  # [T, 7]

        # sample a random window of nframes
        vfps = 30
        fpc = self.frames_per_clip
        fps = self.fps if self.fps is not None else vfps
        fstp = ceil(vfps / fps)
        nframes = int(fpc * fstp)
        vlen = len(states)

        if vlen < nframes:
            raise Exception(f"Video is too short {local_path=}, {nframes=}, {vlen=}")

        ef = self.rng.randint(nframes, vlen)
        sf = ef - nframes
        indices = np.arange(sf, sf + nframes, fstp).astype(np.int64)

        states = states[indices, :][:: self.frameskip]

        # action def from robot-world-models
        # actions = _get_actions(trajectory, action_type=self._action_type)
        # actions = actions[indices, :][::self.frameskip]
        # OR action def from jepa-internal
        actions = poses_to_diffs(states[:: self.action_skip])

        buffer = trajectory["episode_data"]["observation"][camera_view][indices, :][:: self.frameskip]
        buffer = buffer / 255.0
        buffer = torch.tensor(buffer, dtype=torch.float32).permute(0, 3, 1, 2)  # T H W C -> T C H W
        if self.transform is not None:
            buffer = self.transform(buffer)
        return buffer, actions, states, None, indices

    def loadvideo_decord(self, path):
        """
        Returns:
            buffer: torch.Tensor of shape [T, C, H, W] with video frames
            actions: np.ndarray of shape [T-1, 7] with robot actions
            states: np.ndarray of shape [T, 7] with robot states
            extrinsics: np.ndarray of shape [T, 6] with camera extrinsics
            indices: np.ndarray of shape [T] with sampled frame indices
        """
        # -- load metadata
        metadata = get_json(path)
        if metadata is None:
            raise Exception(f"No metadata for video {path=}")

        # -- load trajectory info
        tpath = os.path.join(path, self.h5_name)
        trajectory = h5py.File(tpath)

        # -- randomly sample a camera view
        camera_view = self.camera_views[self.rng.randint(0, len(self.camera_views))]
        mp4_name = metadata[camera_view].split("recordings/MP4/")[-1]
        camera_name = mp4_name.split(".")[0]
        extrinsics = trajectory["observation"]["camera_extrinsics"][f"{camera_name}_left"]
        # TODO: print
        states = np.concatenate(
            [
                np.array(trajectory["observation"]["robot_state"]["cartesian_position"]),
                np.array(trajectory["observation"]["robot_state"]["gripper_position"])[:, None],
            ],
            axis=1,
        )  # [T, 7]
        vpath = os.path.join(path, "recordings/MP4", mp4_name)
        vr = VideoReader(vpath, num_threads=-1, ctx=cpu(0))
        # --
        vfps = vr.get_avg_fps()
        fpc = self.frames_per_clip
        fps = self.fps if self.fps is not None else vfps
        fstp = ceil(vfps / fps)
        nframes = int(fpc * fstp)
        vlen = len(vr)

        if vlen < nframes:
            raise Exception(f"Video is too short {vpath=}, {nframes=}, {vlen=}")

        # sample a random window of nframes
        ef = self.rng.randint(nframes, vlen)
        sf = ef - nframes
        indices = np.arange(sf, sf + nframes, fstp).astype(np.int64)
        # --
        states = states[indices, :][:: self.frameskip]
        extrinsics = extrinsics[indices, :][:: self.frameskip]
        if self.camera_frame:
            states = self.transform_frame(states, extrinsics)
        actions = poses_to_diffs(states[:: self.action_skip])
        # --
        vr.seek(0)  # go to start of video before sampling frames
        buffer = vr.get_batch(indices).asnumpy()
        # Added by Basile
        buffer = buffer / 255.0
        buffer = torch.tensor(buffer, dtype=torch.float32).permute(0, 3, 1, 2)  # T H W C -> T C H W
        # transform handles in the same way if input is already torch or numpy.
        if self.transform is not None:
            buffer = self.transform(buffer)

        return buffer, actions, states, extrinsics, indices

    def __len__(self):
        return len(self.samples)


DROID_ACTION_TYPES = [
    "action",
    "delta-action",
    "state",
    "delta-state",
]


def _get_delta(values: np.ndarray) -> np.ndarray:
    """
    values.shape: [T, D]
    Returns: delta.shape [T, D] with delta[-1] being zero.
    """
    delta = np.zeros_like(values)
    delta[:-1] = values[1:] - values[:-1]
    delta[:, 3:6][delta[:, 3:6] > np.pi] -= 2 * np.pi
    delta[:, 3:6][delta[:, 3:6] < -np.pi] += 2 * np.pi
    return delta


def _get_state(episode: dict[str, Any]) -> np.ndarray:
    """Returns the robot state."""
    cartesian_pos = episode["episode_data"]["observation"]["cartesian_position"]
    gripper_pos = episode["episode_data"]["observation"]["gripper_position"]
    if gripper_pos.ndim == 1:
        gripper_pos = gripper_pos[:].reshape(-1, 1)
    state = np.concatenate([cartesian_pos, gripper_pos], axis=1)
    return state


def _get_actions(episode: dict[str, Any], action_type: str = "action") -> np.ndarray:
    assert action_type in DROID_ACTION_TYPES
    if action_type == "action":
        return episode["episode_data"]["action"]
    elif action_type == "delta-action":
        actions = episode["episode_data"]["action"]
        return _get_delta(actions)
    elif action_type == "state":
        state = _get_state(episode)
        return state
    elif action_type == "delta-state":
        state = _get_state(episode)
        return _get_delta(state)
    else:
        raise ValueError(f"invalid {action_type = }")


def poses_to_diffs(poses):
    """
    Poses: shape [T, 7]
    Returns: shape [T-1, 7]
    """
    xyz = poses[:, :3]  # shape [T, 3]
    thetas = poses[:, 3:6]  # euler angles, shape [T, 3]
    matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in thetas]
    xyz_diff = xyz[1:] - xyz[:-1]
    angle_diff = [matrices[t + 1] @ matrices[t].T for t in range(len(matrices) - 1)]
    angle_diff = [Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in angle_diff]
    angle_diff = np.stack([d for d in angle_diff], axis=0)
    closedness = poses[:, -1:]
    closedness_delta = closedness[1:] - closedness[:-1]
    return np.concatenate([xyz_diff, angle_diff, closedness_delta], axis=1)


def compute_new_pose(pose, action):
    """
    :param pose: [B, T=1, 7]
    :param action: [B, T=1, 7]
    :returns: [B, T=1, 7]
    """
    device, dtype = pose.device, pose.dtype
    pose = pose[:, 0].cpu().numpy()
    action = action[:, 0].cpu().numpy()
    # -- compute delta xyz
    new_xyz = pose[:, :3] + action[:, :3]
    # -- compute delta theta
    thetas = pose[:, 3:6]
    delta_thetas = action[:, 3:6]
    matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in thetas]
    delta_matrices = [Rotation.from_euler("xyz", theta, degrees=False).as_matrix() for theta in delta_thetas]
    angle_diff = [delta_matrices[t] @ matrices[t] for t in range(len(matrices))]
    angle_diff = [Rotation.from_matrix(mat).as_euler("xyz", degrees=False) for mat in angle_diff]
    new_angle = np.stack([d for d in angle_diff], axis=0)  # [B, 7]
    # -- compute delta gripper
    new_closedness = pose[:, -1:] + action[:, -1:]
    new_closedness = np.clip(new_closedness, 0, 1)
    # -- new pose
    new_pose = np.concatenate([new_xyz, new_angle, new_closedness], axis=-1)
    return torch.from_numpy(new_pose).to(device).to(dtype)[:, None]
