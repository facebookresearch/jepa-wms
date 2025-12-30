# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import os
from logging import getLogger
from math import ceil
from pathlib import Path
from typing import Any, Sequence

import decord
import h5py
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from decord import VideoReader, cpu
from einops import rearrange, repeat
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from .traj_dset import TrajDataset, get_train_val_sliced

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

    def loadvideo_hf(self, path):
        """
        Returns:
            buffer: torch.Tensor of shape [T, C, H, W] with video frames
            actions: np.ndarray of shape [T-1, 7] with robot actions
            states: np.ndarray of shape [T, 7] with robot states
            extrinsics: None (not used in this dataset)
            indices: np.ndarray of shape [T] with sampled frame indices
        """
        trajectory = h5py.File(path)
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
            raise Exception(f"Video is too short {path=}, {nframes=}, {vlen=}")

        ef = self.rng.randint(nframes, vlen)
        sf = ef - nframes
        indices = np.arange(sf, sf + nframes, fstp).astype(np.int64)

        states = states[indices, :][:: self.frameskip]
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


class DROIDTrajDataset(TrajDataset):
    """
    DROID dataset that inherits from TrajDataset and supports trajectory slicing.

    This class loads entire trajectories (not random slices) and provides
    get_seq_length() so it can be wrapped with TrajSlicerDataset for
    consistent training sample counting across different num_frames settings.
    """

    def __init__(
        self,
        data_path,
        camera_views=["wrist_mp4_path"],
        frameskip=1,
        action_skip=1,
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
        self.rng = np.random.RandomState(seed)

        if VideoReader is None:
            raise ImportError('Unable to import "decord" which is required to read videos.')

        self.camera_views = camera_views
        logger.info(f"Using DROID TrajDataset with camera views: {self.camera_views}")

        if self.mpk_dset:
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
            logger.info(f"Not slicing DROID TrajDataset, using {len(self.samples)} samples")

        # Pre-compute sequence lengths for all videos
        self._seq_lengths = self._compute_seq_lengths()

        # Compute normalization statistics
        states = []
        actions = []
        proprio_states = []
        seq_lengths = []
        for i in tqdm(range(min(num_samples_stored, len(self.samples))),
                      desc=f"Loading {num_samples_stored} DROID eps to compute mean/std"):
            obs, action, state, _ = self.__getitem__(i, debug=debug)
            states.append(torch.tensor(state))
            actions.append(torch.tensor(action))
            proprio_states.append(torch.tensor(state))
            seq_lengths.append(len(state))

        # Pad to max length for stacking
        max_len = max(seq_lengths)
        padded_states = [torch.nn.functional.pad(s, (0, 0, 0, max_len - len(s))) for s in states]
        padded_actions = [torch.nn.functional.pad(a, (0, 0, 0, max_len - len(a))) for a in actions]
        padded_proprios = [torch.nn.functional.pad(p, (0, 0, 0, max_len - len(p))) for p in proprio_states]

        self.states = torch.stack(padded_states)
        self.actions = torch.stack(padded_actions)
        self.proprios = torch.stack(padded_proprios)
        self.seq_lengths_tensor = torch.tensor(seq_lengths)
        self.rewards = None

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]

        if normalize_action:
            self.action_mean, self.action_std = self.get_data_mean_std(self.actions, self.seq_lengths_tensor)
            self.state_mean, self.state_std = self.get_data_mean_std(self.states, self.seq_lengths_tensor)
            self.proprio_mean, self.proprio_std = self.get_data_mean_std(self.proprios, self.seq_lengths_tensor)
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

    def _load(self):
        paths = []
        for pattern in self.mpk_manifest_patterns:
            cleaned_pattern = pattern.lstrip("/")
            found = list(Path(self.data_path).glob(cleaned_pattern))
            if not found:
                print(f"Warning: No files found for pattern {cleaned_pattern}")
            paths.extend(found)
        return [str(p) for p in paths]

    def _compute_seq_lengths(self):
        """Compute the length of each trajectory in the dataset."""
        seq_lengths = []
        for path in tqdm(self.samples, desc="Computing sequence lengths"):
            try:
                if self.mpk_dset:
                    trajectory = h5py.File(path)
                    vlen = len(np.array(trajectory["episode_data"]["observation"]["cartesian_position"]))
                else:
                    metadata = get_json(path)
                    if metadata is None:
                        seq_lengths.append(0)
                        continue
                    camera_view = self.camera_views[0]
                    mp4_name = metadata[camera_view].split("recordings/MP4/")[-1]
                    vpath = os.path.join(path, "recordings/MP4", mp4_name)
                    vr = VideoReader(vpath, num_threads=-1, ctx=cpu(0))
                    vlen = len(vr)

                # Apply fps downsampling
                vfps = 30
                fps = self.fps if self.fps is not None else vfps
                fstp = ceil(vfps / fps)
                effective_len = vlen // fstp
                seq_lengths.append(effective_len)
            except Exception as e:
                logger.warning(f"Failed to get seq length for {path}: {e}")
                seq_lengths.append(0)
        return seq_lengths

    def get_seq_length(self, idx: int) -> int:
        """Returns the length of the idx-th trajectory."""
        return self._seq_lengths[idx]

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

    def __getitem__(self, idx, subtask=None, debug=False):
        """
        Returns the FULL trajectory for the given index.

        Returns:
            obs: dict with 'visual' (T, C, H, W) and 'proprio' (T, 7)
            actions: (T-1, 7) or (T, 7) with last action padded
            states: (T, 7)
            reward: scalar tensor
        """
        path = self.samples[idx]

        loaded_video = False
        while not loaded_video:
            if debug:
                buffer, actions, states, extrinsics = self._load_full_trajectory(path)
                loaded_video = True
            else:
                try:
                    buffer, actions, states, extrinsics = self._load_full_trajectory(path)
                    loaded_video = True
                except Exception as e:
                    logger.info(f"Exception loading video {path=} {e=}")
                    loaded_video = False
                    idx = self.rng.randint(0, self.__len__())
                    path = self.samples[idx]

        obs = {
            "visual": buffer,
            "proprio": states,
        }

        if self.droid_to_rcasa_action_format > 1:
            actions = self._repeat_divide_action(actions, act_repeat=self.droid_to_rcasa_action_format)

        # Pad actions with dummy last action so that it has the same length as obs
        if len(actions) < len(states):
            actions = np.concatenate([actions, np.zeros((1, actions.shape[-1]))], axis=0)

        return obs, actions, states, torch.tensor(0.0), None

    def _repeat_divide_action(self, action, act_repeat=5):
        return repeat(action, "t a -> t (f a)", f=act_repeat) / float(act_repeat)

    def _load_full_trajectory(self, path):
        """Load the entire trajectory (all frames) from a video."""
        if self.mpk_dset:
            return self._load_full_trajectory_hf(path)
        else:
            return self._load_full_trajectory_decord(path)

    def _load_full_trajectory_hf(self, path):
        """Load full trajectory from HuggingFace-style h5 file."""
        trajectory = h5py.File(path)
        camera_view = self.camera_views[self.rng.randint(0, len(self.camera_views))]

        states = np.concatenate(
            [
                np.array(trajectory["episode_data"]["observation"]["cartesian_position"]),
                np.array(trajectory["episode_data"]["observation"]["gripper_position"])[:, None],
            ],
            axis=1,
        )  # [T, 7]

        vfps = 30
        fps = self.fps if self.fps is not None else vfps
        fstp = ceil(vfps / fps)
        vlen = len(states)

        # Sample ALL frames at the target fps
        indices = np.arange(0, vlen, fstp).astype(np.int64)

        states = states[indices, :][:: self.frameskip]
        actions = poses_to_diffs(states[:: self.action_skip])

        buffer = trajectory["episode_data"]["observation"][camera_view][indices, :][:: self.frameskip]
        buffer = buffer / 255.0
        buffer = torch.tensor(buffer, dtype=torch.float32).permute(0, 3, 1, 2)  # T H W C -> T C H W

        if self.transform is not None:
            buffer = self.transform(buffer)

        return buffer, actions, states, None

    def _load_full_trajectory_decord(self, path):
        """Load full trajectory from decord video files."""
        metadata = get_json(path)
        if metadata is None:
            raise Exception(f"No metadata for video {path=}")

        tpath = os.path.join(path, "trajectory.h5")
        trajectory = h5py.File(tpath)

        camera_view = self.camera_views[self.rng.randint(0, len(self.camera_views))]
        mp4_name = metadata[camera_view].split("recordings/MP4/")[-1]
        camera_name = mp4_name.split(".")[0]
        extrinsics = trajectory["observation"]["camera_extrinsics"][f"{camera_name}_left"]

        states = np.concatenate(
            [
                np.array(trajectory["observation"]["robot_state"]["cartesian_position"]),
                np.array(trajectory["observation"]["robot_state"]["gripper_position"])[:, None],
            ],
            axis=1,
        )  # [T, 7]

        vpath = os.path.join(path, "recordings/MP4", mp4_name)
        vr = VideoReader(vpath, num_threads=-1, ctx=cpu(0))

        vfps = vr.get_avg_fps()
        fps = self.fps if self.fps is not None else vfps
        fstp = ceil(vfps / fps)
        vlen = len(vr)

        # Sample ALL frames at the target fps
        indices = np.arange(0, vlen, fstp).astype(np.int64)

        states = states[indices, :][:: self.frameskip]
        extrinsics = extrinsics[indices, :][:: self.frameskip]

        if self.camera_frame:
            states = self._transform_frame(states, extrinsics)

        actions = poses_to_diffs(states[:: self.action_skip])

        vr.seek(0)
        buffer = vr.get_batch(indices).asnumpy()
        buffer = buffer / 255.0
        buffer = torch.tensor(buffer, dtype=torch.float32).permute(0, 3, 1, 2)

        if self.transform is not None:
            buffer = self.transform(buffer)

        return buffer, actions, states, extrinsics

    def _transform_frame(self, poses, extrinsics):
        """Transform poses to camera frame coordinates."""
        gripper = poses[:, -1:]
        poses = poses[:, :-1]

        def pose_to_transform(pose):
            trans = pose[:3]
            theta = pose[3:6]
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


def load_droid_slice_train_val(
    transform,
    data_path: str,
    normalize_action: bool = True,
    split_ratio: float = 0.9,
    num_hist: int = 3,
    num_pred: int = 1,
    num_frames_val: int = None,
    frameskip: int = 1,
    action_skip: int = 1,
    traj_subset: bool = True,
    random_seed: int = 42,
    process_actions: str = "concat",
    # DROID-specific parameters
    camera_views: list = None,
    fps: int = 5,
    camera_frame: bool = False,
    mpk_dset: bool = False,
    mpk_manifest_patterns: list = None,
    droid_to_rcasa_action_format: int = 1,
    droid_fraction: float = 1.0,
):
    """
    Load DROID dataset with trajectory slicing for fair comparison across num_frames.

    This function creates a DROIDTrajDataset and wraps it with TrajSlicerDataset
    to ensure consistent training sample counting across different num_frames settings.

    Args:
        transform: Image transform to apply
        data_path: Path to DROID dataset
        normalize_action: Whether to normalize actions
        split_ratio: Train/val split ratio
        num_hist: Number of history frames
        num_pred: Number of prediction frames
        num_frames_val: Number of frames for validation (defaults to num_hist + num_pred)
        frameskip: Frame skip for observations
        action_skip: Frame skip for actions
        traj_subset: Whether to use TrajSubset for splitting
        random_seed: Random seed for reproducibility
        process_actions: How to process actions ("concat" or "sum")
        camera_views: List of camera views to use
        fps: Frames per second
        camera_frame: Whether to transform to camera frame
        mpk_dset: Whether using MPK dataset format
        mpk_manifest_patterns: Patterns for MPK manifest files
        droid_to_rcasa_action_format: Action format conversion factor
        droid_fraction: Fraction of dataset to use

    Returns:
        datasets: dict with 'train' and 'valid' sliced datasets
        traj_dsets: dict with 'train' and 'valid' trajectory datasets
    """
    if camera_views is None:
        camera_views = ["wrist_mp4_path"]

    dset = DROIDTrajDataset(
        data_path=data_path,
        camera_views=camera_views,
        fps=fps,
        transform=transform,
        camera_frame=camera_frame,
        normalize_action=normalize_action,
        mpk_dset=mpk_dset,
        mpk_manifest_patterns=mpk_manifest_patterns,
        droid_to_rcasa_action_format=droid_to_rcasa_action_format,
        seed=random_seed,
        droid_fraction=droid_fraction,
        frameskip=frameskip,
        action_skip=action_skip,
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
