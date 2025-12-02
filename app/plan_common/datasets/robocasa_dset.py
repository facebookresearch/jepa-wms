import json
import os
from logging import getLogger
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from einops import rearrange
from scipy.spatial.transform import Rotation as R

from .droid_dset import poses_to_diffs
from .traj_dset import TrajDataset, TrajSlicerDataset, get_train_val_sliced

logger = getLogger()

# Task category mappings for better directory navigation
TASK_CATEGORY_MAPPING = {
    # Single-stage tasks
    "PnPSinkToCounter": "kitchen_pnp",
    "PnPCounterToCab": "kitchen_pnp",
    "PnPCabToCounter": "kitchen_pnp",
    "PnPCounterToSink": "kitchen_pnp",
    "PnPCounterToMicrowave": "kitchen_pnp",
    "PnPMicrowaveToCounter": "kitchen_pnp",
    "PnPCounterToStove": "kitchen_pnp",
    "PnPStoveToCounter": "kitchen_pnp",
    "OpenSingleDoor": "kitchen_doors",
    "CloseSingleDoor": "kitchen_doors",
    "OpenDoubleDoor": "kitchen_doors",
    "CloseDoubleDoor": "kitchen_doors",
    "OpenDrawer": "kitchen_drawer",
    "CloseDrawer": "kitchen_drawer",
    "TurnOnSinkFaucet": "kitchen_sink",
    "TurnOffSinkFaucet": "kitchen_sink",
    "TurnSinkSpout": "kitchen_sink",
    "TurnOnStove": "kitchen_stove",
    "TurnOffStove": "kitchen_stove",
    "CoffeeSetupMug": "kitchen_coffee",
    "CoffeeServeMug": "kitchen_coffee",
    "CoffeePressButton": "kitchen_coffee",
    "TurnOnMicrowave": "kitchen_microwave",
    "TurnOffMicrowave": "kitchen_microwave",
    "NavigateKitchen": "kitchen_navigate",
    # Multi-stage tasks
    "ArrangeVegetables": "chopping_food",
    "MicrowaveThawing": "defrosting_food",
    "RestockPantry": "restocking_supplies",
    "PreSoakPan": "washing_dishes",
    "PrepareCoffee": "brewing",
}

# Default horizons for tasks (used if not specified)
TASK_HORIZONS = {
    "PnPSinkToCounter": 500,
    "PnPCounterToCab": 500,
    "PnPCabToCounter": 500,
    "PnPCounterToSink": 700,
    "PnPCounterToMicrowave": 600,
    "PnPMicrowaveToCounter": 500,
    "PnPCounterToStove": 500,
    "PnPStoveToCounter": 500,
    "OpenSingleDoor": 500,
    "CloseSingleDoor": 500,
    "OpenDoubleDoor": 1000,
    "CloseDoubleDoor": 700,
    "OpenDrawer": 500,
    "CloseDrawer": 500,
    "TurnOnSinkFaucet": 500,
    "TurnOffSinkFaucet": 500,
    "TurnSinkSpout": 500,
    "TurnOnStove": 500,
    "TurnOffStove": 500,
    "CoffeeSetupMug": 600,
    "CoffeeServeMug": 600,
    "CoffeePressButton": 300,
    "TurnOnMicrowave": 500,
    "TurnOffMicrowave": 500,
    "NavigateKitchen": 500,
    "ArrangeVegetables": 1200,
    "MicrowaveThawing": 1000,
    "RestockPantry": 1000,
    "PreSoakPan": 1500,
    "PrepareCoffee": 1000,
}


class RoboCasaDataset(TrajDataset):
    """
    Dataset for RoboCasa robotics tasks.

    Supports both single-stage and multi-stage tasks, as well as
    both human demonstrations and MimicGen generated data.
    """

    def __init__(
        self,
        n_rollout=50,
        transform=None,
        data_path=None,
        filter_tasks=["PnPCounterToCab"],  # Can be a single task name or a list of task names
        filter_first_episodes=10,
        camera_views=["robot0_agentview_left"],
        normalize_action=True,
        use_human=True,
        use_mg=True,
        manip_only=True,
        with_reward=True,
        output_rcasa_state=False,
        output_rcasa_info=False,
        rcasa_to_droid_action_format=False,
        custom_teleop_dset=False,
    ):
        self.transform = transform
        self.data_path = data_path
        self.camera_views = camera_views
        self.use_human = use_human
        self.use_mg = use_mg
        self.manip_only = manip_only
        self.with_reward = with_reward
        self.output_rcasa_state = output_rcasa_state
        self.output_rcasa_info = output_rcasa_info
        self.rcasa_to_droid_action_format = rcasa_to_droid_action_format
        self.custom_teleop_dset = custom_teleop_dset
        if filter_tasks is None:
            self.filter_tasks = list(TASK_CATEGORY_MAPPING.keys())
        elif isinstance(filter_tasks, str):
            self.filter_tasks = [filter_tasks]
        else:
            self.filter_tasks = list(filter_tasks)
        self.file_paths = []
        self.task_info = {}
        self.proprio_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]

        self.trajectories = []
        self.seq_lengths = []
        self.actions = []
        self.proprios = []

        action_all = []
        proprio_all = []
        rewards = []  # Store rewards if needed

        # Handle custom teleop dataset if specified
        if self.custom_teleop_dset:
            # Use the custom teleop dataset path instead
            dataset_root = os.environ.get("DATASET_ROOT")
            custom_path = f"{dataset_root}/robocasa/jimmy_gen"
            logger.info(f"Using custom teleop dataset from {custom_path}")

            # Find all h5/hdf5 files in the custom path
            for root, _, files in os.walk(custom_path):
                for file in files:
                    # use custom teleop extracted observations
                    if file.endswith("im256.hdf5"):
                        self.file_paths.append(os.path.join(root, file))

            if len(self.file_paths) == 0:
                raise ValueError(f"No hdf5 files found in custom teleop dataset path: {custom_path}")

            logger.info(f"Found {len(self.file_paths)} files in custom teleop dataset")

            # For custom dataset, we'll use a generic task info
            for task in self.filter_tasks:
                self.task_info[task] = {
                    "type": "custom_teleop",
                    "category": "teleop",
                    "horizon": 500,  # Default horizon
                }
        else:
            # Process each task
            for task_name in self.filter_tasks:
                # Determine if single or multi stage task
                is_single_stage = task_name in TASK_CATEGORY_MAPPING and any(
                    cat in TASK_CATEGORY_MAPPING[task_name]
                    for cat in [
                        "kitchen_pnp",
                        "kitchen_doors",
                        "kitchen_drawer",
                        "kitchen_sink",
                        "kitchen_stove",
                        "kitchen_coffee",
                        "kitchen_microwave",
                        "kitchen_navigate",
                    ]
                )

                task_type = "single_stage" if is_single_stage else "multi_stage"
                category = TASK_CATEGORY_MAPPING.get(task_name, "")

                # Get recommended horizon for this task
                horizon = TASK_HORIZONS.get(task_name, 500)

                self.task_info[task_name] = {"type": task_type, "category": category, "horizon": horizon}

                # Find human dataset (outside of mg directory)
                if self.use_human:
                    base_task_folder = os.path.join(data_path, task_type, category, task_name)

                    human_files = []

                    # Check immediate child directories that aren't 'mg'
                    if os.path.exists(base_task_folder):
                        for item in os.listdir(base_task_folder):
                            item_path = os.path.join(base_task_folder, item)

                            # Skip the mg directory and non-directories
                            if item == "mg" or not os.path.isdir(item_path):
                                continue

                            # Check for hdf5 files in this date directory
                            for root, _, files in os.walk(item_path):
                                for file in files:
                                    if file.endswith("im128_randcams.hdf5") or file.endswith("im128.hdf5"):
                                        human_files.append(os.path.join(root, file))

                    if human_files:
                        human_files.sort()  # Ensure consistent ordering
                        self.file_paths.extend(human_files)
                        logger.info(f"Found {len(human_files)} human files for task {task_name}")

                # Find MimicGen dataset
                if self.use_mg:
                    mg_task_folder = os.path.join(data_path, task_type, category, task_name, "mg")

                    # Find all MimicGen h5 files for this task
                    mg_files = []
                    if os.path.exists(mg_task_folder):
                        for root, dirs, files in os.walk(mg_task_folder):
                            for file in files:
                                if file.endswith("im128_randcams.hdf5") or file.endswith("im128.hdf5"):
                                    mg_files.append(os.path.join(root, file))

                    if mg_files:
                        mg_files.sort()  # Ensure consistent ordering
                        self.file_paths.extend(mg_files)
                        logger.info(f"Found {len(mg_files)} MimicGen files for task {task_name}")

            if len(self.file_paths) == 0:
                raise ValueError(f"No hdf5 files found for any of the specified tasks: {self.filter_tasks}")

        logger.info(f"Total dataset: {len(self.file_paths)} files across {len(self.filter_tasks)} tasks")

        # Load data from all files
        for file_path in self.file_paths:
            with h5py.File(file_path, "r") as f:
                env_args = json.loads(f["data"].attrs["env_args"]) if "env_args" in f["data"].attrs else {}
                task_name = env_args["env_name"] if "env_name" in env_args else "PnPCounterTop"
                demos = list(f["data"].keys())
                demos_sorted = sorted(demos, key=lambda x: int(x.split("_")[1]))
                if filter_first_episodes is not None and filter_first_episodes < len(demos_sorted):
                    logger.info(
                        f"Filtering first {filter_first_episodes}/{len(demos_sorted)} episodes from {file_path}"
                    )
                    demos_filtered = demos_sorted[:filter_first_episodes]
                else:
                    demos_filtered = demos_sorted
                for demo_key in demos_filtered:
                    demo = f["data"][demo_key]
                    model_xml = demo.attrs["model_file"] if "model_file" in demo.attrs else None
                    ep_meta = json.loads(demo.attrs["ep_meta"]) if "ep_meta" in demo.attrs else None
                    if "meta_data_info" in demo:
                        meta_data_info = {}
                        for key in demo["meta_data_info"].keys():
                            meta_data_info[key] = np.array(demo["meta_data_info"][key][:])
                    else:
                        meta_data_info = None
                    acts = demo["actions"][:, :7] if self.manip_only else demo["actions"][:]
                    if self.with_reward:
                        reward = torch.tensor(demo["rewards"][:]).unsqueeze(1)
                        rewards.append(reward)
                    # Get trajectory length
                    traj_len = acts.shape[0]
                    if "obs" in demo:
                        obs = demo["obs"]
                        if "robot0_eef_pos" in obs:
                            proprio_data = []
                            for k in self.proprio_keys:
                                if k in obs:
                                    proprio_data.append(obs[k][:traj_len])
                            if proprio_data:
                                proprio = np.concatenate(proprio_data, axis=1)
                                proprio = self.proprio_to_droid_format(proprio)
                            else:
                                proprio = np.zeros((traj_len, 1))
                        else:
                            proprio = np.zeros((traj_len, 1))
                    else:
                        logger.info(f"No 'obs' found in demo, creating dummy observations")
                        proprio = np.zeros((traj_len, 7))  # Create 7-dim dummy proprio to match DROID format

                    is_mg = "/mg/" in str(file_path)
                    # Store only the metadata, not images
                    self.trajectories.append(
                        {
                            "file_path": file_path,
                            "demo_key": demo_key,
                            "task_name": task_name,
                            "is_mg": is_mg,
                            "traj_len": traj_len,
                        }
                    )
                    if self.output_rcasa_info:
                        self.trajectories[-1].update(
                            {
                                "model_xml": model_xml,
                                "ep_meta": ep_meta,
                                "meta_data_info": meta_data_info,
                            }
                        )

                    action_all.append(torch.tensor(acts))
                    proprio_all.append(torch.tensor(proprio))
                    # state_all.append(states)
                    self.seq_lengths.append(traj_len)

        # Process collected data
        self.actions = torch.cat(action_all)
        self.proprios = torch.cat(proprio_all)
        self.rewards = torch.cat(rewards) if self.with_reward else None
        # states do not have same dims depending on the datasets / tasks

        self.action_dim = self.actions.shape[1]
        self.proprio_dim = self.proprios.shape[1]
        self.state_dim = self.proprio_dim

        if normalize_action:
            self.action_mean = self.actions.mean(dim=0)
            self.action_std = self.actions.std(dim=0)
            self.action_std[self.action_std < 1e-5] = 1.0

            self.proprio_mean = self.proprios.mean(dim=0)
            self.proprio_std = self.proprios.std(dim=0)
            self.proprio_std[self.proprio_std < 1e-5] = 1.0

            self.state_mean = self.proprio_mean
            self.state_std = self.proprio_std
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)

        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std

        logger.info(f"Loaded {len(self.trajectories)} trajectories")

    def proprio_to_droid_format(self, proprio):
        """
        Convert proprioceptive data to Droid format.
        Args:
            proprio: tensor of shape (T, 9)
        Returns:
            tensor of shape (T, 3 + 3 + 1) for position, orientation, and gripper state
        """
        eef_pos = proprio[:, :3]
        eef_quat = proprio[:, 3:7]
        gripper_qpos = proprio[:, 7:9]
        eef_euler = self.eef_quat_to_xyz(eef_quat)
        gripper_state = self.gripper_2d_to_1d(gripper_qpos)
        droid_proprio = np.concatenate((eef_pos, eef_euler, gripper_state), axis=1)
        return droid_proprio

    def eef_quat_to_xyz(self, eef_quat):
        # shape (T, 4)
        # If your quaternion is [w, x, y, z], convert to [x, y, z, w] for scipy
        eef_quat_xyzw = np.concatenate(
            [eef_quat[:, 1:2], eef_quat[:, 2:3], eef_quat[:, 3:4], eef_quat[:, 0:1]], axis=1
        )
        # eef_quat_xyzw = np.array([eef_quat[:, 1], eef_quat[:, 2], eef_quat[:, 3], eef_quat[:, 0]]).transpose(1, 0)
        # Convert to Euler angles (xyz order, radians)
        eef_euler = R.from_quat(eef_quat_xyzw).as_euler("xyz", degrees=False)
        return eef_euler  # shape (T, 3)

    def gripper_2d_to_1d(self, gripper_qpos):
        """
        Convert 2D gripper position to 1D representation.
        Args:
            gripper_qpos: tensor of shape (T, 2) for gripper position
        Returns:
            tensor of shape (T, 1) for gripper state
        """
        return gripper_qpos[:, 0:1] - gripper_qpos[:, 1:2]

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_frames(self, idx, frames, subtask=None):
        """
        Get a specific frame range from a trajectory.

        Args:
            idx: trajectory index
            frames: list or range of frame indices
        """
        trajectory_info = self.trajectories[idx]
        trajectory = h5py.File(trajectory_info["file_path"], "r")["data"][trajectory_info["demo_key"]]
        if subtask is not None:
            task_segments = trajectory["meta_data_info/current_task_segment"][:]
            required_segments = []

            if subtask == "reach-pick-place":
                required_segments = [0, 1, 2]
                frames = [f for f in frames if task_segments[f] in required_segments]
            elif subtask == "reach-pick":
                required_segments = [0, 1]
                frames = [f for f in frames if task_segments[f] in required_segments]
            elif subtask == "pick-place":
                required_segments = [1, 2]
                frames = [f for f in frames if task_segments[f] in required_segments]
            elif subtask == "reach":
                required_segments = [0]
                frames = [f for f in frames if task_segments[f] == 0]
            elif subtask == "pick":
                required_segments = [1]
                frames = [f for f in frames if task_segments[f] == 1]
            elif subtask == "place":
                required_segments = [2]
                frames = [f for f in frames if task_segments[f] == 2]

            # Check if all required segments are present in the filtered frames
            present_segments = set(task_segments[frames])
            missing_segments = set(required_segments) - present_segments

            if missing_segments:
                raise ValueError(
                    f"Trajectory {idx} does not contain all required task segments for subtask '{subtask}'. Missing segments: {missing_segments}"
                )

            if len(frames) == 0:
                raise ValueError(f"No frames match the subtask '{subtask}' in trajectory {idx}")

        # Handle actions and states regardless of obs presence
        if self.rcasa_to_droid_action_format:
            # We'll need proprio for this, which we'll handle below
            pass
        else:
            act = torch.tensor(trajectory["actions"][frames, : self.action_dim])  # [B T 7]
        state = torch.tensor(trajectory["states"][frames]) if self.output_rcasa_state else None

        if "obs" in trajectory:
            obs = trajectory["obs"]
            proprio_data = []
            for k in self.proprio_keys:
                if k in obs:
                    proprio_data.append(obs[k][frames])
            if proprio_data:
                proprio = np.concatenate(proprio_data, axis=1)
                proprio = self.proprio_to_droid_format(proprio)
            images = trajectory["obs"][f"{self.camera_views[0]}_image"][frames]
            if "right" in self.camera_views[0]:
                images = images[:, ::-1, ::-1, :].copy()
            images = torch.from_numpy(images).float() / 255.0  # Normalize to [0, 1]
            images = rearrange(images, "t h w c -> t c h w")  # THWC -> TCHW
            if self.transform:
                images = self.transform(images)
        else:
            proprio = torch.zeros((len(frames), 7))  # 7-dim dummy proprio
            images = torch.zeros((len(frames), 3, 224, 224))  # Dummy RGB images

        if self.rcasa_to_droid_action_format:
            act = torch.tensor(poses_to_diffs(proprio))  # [B T-1 7]
        obs = {"visual": images, "proprio": proprio}
        if self.with_reward:
            reward = torch.from_numpy(trajectory["rewards"][frames])
        else:
            reward = None
        return obs, act, state, reward, trajectory_info

    def __getitem__(self, idx, subtask=None):
        seq_len = self.get_seq_length(idx)
        return self.get_frames(idx, range(seq_len), subtask=subtask)

    def __len__(self):
        return len(self.trajectories)


def load_robocasa_slice_train_val(
    transform,
    n_rollout=50,
    data_path=None,
    filter_tasks=None,
    filter_first_episodes=10,
    normalize_action=True,
    split_ratio=0.8,
    num_hist=0,
    num_pred=0,
    num_frames_val=None,
    frameskip=1,
    action_skip=1,
    camera_views=["robot0_agentview_left"],
    use_human=True,
    use_mg=True,
    traj_subset=True,
    random_seed=42,
    with_reward=True,
    output_rcasa_state=False,
    output_rcasa_info=False,
    rcasa_to_droid_action_format=False,
    process_actions="concat",
    custom_teleop_dset=False,
):
    """
    Load RoboCasa dataset and split into train and validation sets.

    Args:
        transform: image transform function
        n_rollout: maximum number of trajectories to load per file
        data_path: path to robocasa dataset
        filter_tasks: name(s) of the task(s) to load (string or list of strings)
        normalize_action: whether to normalize actions
        split_ratio: ratio of train to total data
        num_hist: number of history frames
        num_pred: number of prediction frames
        num_frames_val: number of frames for validation (if None, same as train)
        frameskip: number of frames to skip
        action_skip: number of actions to skip
        camera_views: list of camera views to include
        use_human: whether to use human demonstrations
        use_mg: whether to use MimicGen demonstrations
        traj_subset: whether to use a subset of trajectories for training
        random_seed: random seed for reproducibility

    Returns:
        datasets: dictionary with 'train' and 'valid' sliced datasets
        traj_dset: dictionary with 'train' and 'valid' trajectory datasets
    """
    dset = RoboCasaDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path,
        filter_tasks=filter_tasks,
        filter_first_episodes=filter_first_episodes,
        camera_views=camera_views,
        normalize_action=normalize_action,
        use_human=use_human,
        use_mg=use_mg,
        with_reward=with_reward,
        output_rcasa_state=output_rcasa_state,
        output_rcasa_info=output_rcasa_info,
        rcasa_to_droid_action_format=rcasa_to_droid_action_format,
        custom_teleop_dset=custom_teleop_dset,
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
