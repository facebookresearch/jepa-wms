# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
from typing import Callable

import decord
import torch
import torch.utils.data

from src.datasets.data_manager import init_data as init_data_src

from .droid_dset import DROIDVideoDataset
from .metaworld_hf_dset import load_metaworld_hf_slice_train_val
from .point_maze_dset import load_point_maze_slice_train_val
from .pusht_dset import load_pusht_slice_train_val
from .robocasa_dset import load_robocasa_slice_train_val
from .wall_dset import load_wall_slice_train_val

# ----------------

_GLOBAL_SEED = 0
logger = getLogger()


def init_data(
    # Required parameters
    data_paths,
    batch_size,
    # Dataset configuration
    dataset_type="custom",
    val_data_paths=None,
    datasets_weights=None,
    val_datasets_weights=None,
    seed=42,
    with_reward=False,
    transform=None,
    # Custom dataset parameters
    split_ratio=0.9,
    frameskip=None,
    action_skip=1,
    normalize_action=True,  # normalized both action and proprio when building TrajDataset
    traj_subset=True,
    filter_first_episodes=100,
    filter_tasks=None,
    num_hist=3,
    num_pred=1,
    process_actions="concat",
    use_hf_format=False,
    # Droid-specific parameters
    camera_views=0,
    camera_frame=False,
    droid_to_rcasa_action_format=False,
    rcasa_to_droid_action_format=False,
    fps=5,
    dataset_fpcs=[16],
    mpk_manifest_patterns: list[str] = None,
    custom_teleop_dset=False,
    droid_fraction=1,
    val_droid_fraction=1,
    # RoboCasa-specific parameters
    output_rcasa_state=False,
    output_rcasa_info=False,
    # Validation parameters
    num_frames_val=None,
    val_dataset_batch_size=4,
    val_dataset_drop_last=False,
    val_dataset_fpcs=[16],
    val_dataset_camera_views=0,
    val_viz_rank0_loader: bool = False,
    # DataLoader parameters
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    drop_last=True,
    collator=None,
    # Distributed training parameters
    rank=0,
    world_size=1,
    # Other parameters (deprecated/unused)
    filter_short_videos=False,
    duration=None,
    **kwargs,
) -> tuple[Callable]:
    print(f"{data_paths=}")
    shuffle = True
    if dataset_type == "custom":
        if all("droid" in p for p in data_paths) or all("franka_custom" in p for p in data_paths):
            # We never pass the normalize_action argument to DROIDVideoDataset
            dataset = DROIDVideoDataset(
                data_path=data_paths[0],
                frames_per_clip=dataset_fpcs[0],
                transform=transform,
                fps=fps,
                camera_views=camera_views,
                mpk_manifest_patterns=mpk_manifest_patterns,
                mpk_dset=all("franka_custom" in p for p in data_paths),
                camera_frame=camera_frame,
                droid_to_rcasa_action_format=droid_to_rcasa_action_format,
                seed=seed,
                droid_fraction=droid_fraction,
                action_skip=action_skip,
            )
            if val_data_paths:
                val_dataset = DROIDVideoDataset(
                    data_path=val_data_paths[0],
                    frames_per_clip=val_dataset_fpcs[0],
                    transform=transform,
                    fps=fps,
                    camera_views=val_dataset_camera_views,
                    mpk_manifest_patterns=mpk_manifest_patterns,
                    mpk_dset=all("franka_custom" in p for p in val_data_paths),
                    camera_frame=camera_frame,
                    droid_to_rcasa_action_format=droid_to_rcasa_action_format,
                    seed=seed,
                    droid_fraction=val_droid_fraction,
                    action_skip=action_skip,
                )
            datasets = {"train": dataset, "valid": val_dataset}
            traj_dsets = {"train": dataset, "valid": val_dataset}
        elif all("metaworld" in p.lower() for p in data_paths) or all("tdmpc2" in p for p in data_paths):
            datasets, traj_dsets = load_metaworld_hf_slice_train_val(
                transform,
                n_rollout=None,
                data_path=data_paths[0],
                normalize_action=normalize_action,
                split_ratio=split_ratio,
                num_hist=num_hist,
                num_pred=num_pred,
                num_frames_val=num_frames_val,
                frameskip=frameskip,
                action_skip=action_skip,
                traj_subset=traj_subset,
                filter_tasks=filter_tasks,
                with_reward=with_reward,
                random_seed=seed,
                process_actions=process_actions,
            )
            dataset = datasets["train"]
            shuffle = True
        elif all("pusht" in p for p in data_paths):
            datasets, traj_dsets = load_pusht_slice_train_val(
                transform,
                n_rollout=None,
                data_path=data_paths[0],
                normalize_action=normalize_action,
                split_ratio=split_ratio,
                num_hist=num_hist,
                num_pred=num_pred,
                num_frames_val=num_frames_val,
                frameskip=frameskip,
                action_skip=action_skip,
                with_velocity=True,
                random_seed=seed,
                process_actions=process_actions,
            )
            dataset = datasets["train"]
            shuffle = False
        elif all("point_maze" in p for p in data_paths):
            datasets, traj_dsets = load_point_maze_slice_train_val(
                transform,
                n_rollout=None,
                data_path=data_paths[0],
                normalize_action=normalize_action,
                split_ratio=split_ratio,
                # num_frames=16,
                num_hist=num_hist,
                num_pred=num_pred,
                num_frames_val=num_frames_val,
                frameskip=frameskip,
                action_skip=action_skip,
                traj_subset=traj_subset,
                random_seed=seed,
                process_actions=process_actions,
            )
            dataset = datasets["train"]
            shuffle = False
        elif all("wall" in p for p in data_paths):
            datasets, traj_dsets = load_wall_slice_train_val(
                transform,
                n_rollout=None,
                data_path=data_paths[0],
                normalize_action=normalize_action,
                split_ratio=split_ratio,
                num_hist=num_hist,
                num_pred=num_pred,
                num_frames_val=num_frames_val,
                frameskip=frameskip,
                action_skip=action_skip,
                traj_subset=traj_subset,
                random_seed=seed,
                process_actions=process_actions,
            )
            dataset = datasets["train"]
        elif all("robocasa" in p for p in data_paths):
            datasets, traj_dsets = load_robocasa_slice_train_val(
                transform,
                n_rollout=None,
                data_path=data_paths[0],
                normalize_action=normalize_action,
                split_ratio=split_ratio,
                num_hist=num_hist,
                num_pred=num_pred,
                num_frames_val=num_frames_val,
                frameskip=frameskip,
                action_skip=action_skip,
                traj_subset=traj_subset,
                filter_tasks=filter_tasks,
                filter_first_episodes=filter_first_episodes,
                with_reward=with_reward,
                random_seed=seed,
                process_actions=process_actions,
                output_rcasa_state=output_rcasa_state,
                output_rcasa_info=output_rcasa_info,
                rcasa_to_droid_action_format=rcasa_to_droid_action_format,
                custom_teleop_dset=custom_teleop_dset,
                camera_views=val_dataset_camera_views,
            )
            dataset = datasets["train"]
        else:
            raise Exception(f"Unknown dataset: {data_paths}")
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
        )
        data_loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=collator,
            sampler=dist_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=pin_mem,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0) and persistent_workers,
        )
        val_dist_sampler = torch.utils.data.distributed.DistributedSampler(
            datasets["valid"],
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
        )
        val_data_loader = torch.utils.data.DataLoader(
            datasets["valid"],
            collate_fn=collator,
            sampler=val_dist_sampler,
            batch_size=val_dataset_batch_size,
            drop_last=val_dataset_drop_last,
            pin_memory=pin_mem,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0) and persistent_workers,
        )
        logger.info("VideoDataset unsupervised data loader created")
        if rank == 0 and val_viz_rank0_loader:
            viz_val_data_loader = torch.utils.data.DataLoader(
                datasets["valid"],
                collate_fn=collator,
                sampler=None,  # No sampler means it will see all validation data
                shuffle=False,  # No need to shuffle for visualization
                batch_size=val_dataset_batch_size,
                drop_last=val_dataset_drop_last,
                pin_memory=pin_mem,
                num_workers=num_workers,
                persistent_workers=(num_workers > 0) and persistent_workers,
            )
            logger.info("Created non-distributed validation loader for visualization")
        else:
            viz_val_data_loader = None
        return (
            dataset,
            datasets["valid"],
            traj_dsets["train"],
            traj_dsets["valid"],
            data_loader,
            val_data_loader,
            dist_sampler,
            viz_val_data_loader,
        )
    else:
        decord.bridge.set_bridge("native")
        data_loader, dist_sampler = init_data_src(
            data=dataset_type,
            root_path=data_paths,
            batch_size=batch_size,
            training=True,
            dataset_fpcs=dataset_fpcs,
            fps=fps,
            transform=transform,
            rank=rank,
            world_size=world_size,
            datasets_weights=datasets_weights,
            persistent_workers=persistent_workers,
            collator=collator,
            num_workers=num_workers,
            pin_mem=pin_mem,
            log_dir=None,
        )
        if val_data_paths is None:
            val_data_loader, val_dist_sampler = None, None
        else:
            val_data_loader, val_dist_sampler = init_data_src(
                data=dataset_type,
                root_path=val_data_paths,
                batch_size=batch_size,
                training=False,
                dataset_fpcs=val_dataset_fpcs,
                fps=fps,
                transform=transform,
                rank=rank,
                world_size=world_size,
                datasets_weights=val_datasets_weights,
                persistent_workers=persistent_workers,
                collator=collator,
                num_workers=num_workers,
                pin_mem=pin_mem,
                log_dir=None,
            )
        return None, None, data_loader.dataset, None, data_loader, val_data_loader, dist_sampler, None


def print_dataloader_attributes(dataloader):
    print("DataLoader Attributes:")
    for attr, value in vars(dataloader).items():
        print(f"{attr}: {value}")
