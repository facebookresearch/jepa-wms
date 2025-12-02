# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import logging
import os
from time import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from omegaconf import OmegaConf

from app.plan_common.datasets.preprocessor import Preprocessor
from app.plan_common.datasets.transforms import make_inverse_transforms, make_transforms
from app.plan_common.datasets.utils import init_data
from src.utils.yaml_utils import expand_env_vars
from evals.simu_env_planning.envs.init import make_env
from evals.simu_env_planning.planning.common.gc_logger import Logger
from evals.simu_env_planning.planning.common.parser import parse_cfg
from evals.simu_env_planning.planning.gc_agent import GC_Agent
from evals.simu_env_planning.planning.plan_evaluator import PlanEvaluator
from evals.simu_env_planning.planning.utils import aggregate_results, compute_task_distribution, set_seed
from src.datasets.utils.utils import get_dataset_paths

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

from src.utils.distributed import init_distributed

# ------------------------------

logging.basicConfig()
log = logging.getLogger()

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


def main(args_eval, resume_preempt=False):

    # Expand environment variables in the config
    args_eval = expand_env_vars(args_eval)

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #
    eval_tag = args_eval.get("tag", None)
    # -- PRETRAIN
    args_pretrain = args_eval.get("model_kwargs")
    module_name = args_pretrain.get("module_name")
    pretrain_folder = args_eval.get("folder", None)

    # -- log/checkpointing paths
    folder = os.path.join(pretrain_folder, "simu_env_planning/")
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    # -- Save args_eval.yaml
    yaml_file_path = os.path.join(folder, "args_eval.yaml")
    with open(yaml_file_path, "w") as yaml_file:
        yaml.dump(args_eval, yaml_file, default_flow_style=False)
    log.info(f"Saved args_eval to {yaml_file_path}")

    # -- Distributed
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    world_size, rank = init_distributed()
    log.info(f"Initialized (rank/world-size) {rank}/{world_size}")
    model_kwargs = args_eval["model_kwargs"]

    # -- Initialize model
    if importlib.util.find_spec(module_name) is None:
        raise NotImplementedError(f"Module {module_name} not found")
    log.info(f"Module found: {module_name}")
    cfgs_data = model_kwargs.get("data", {})
    cfgs_data_aug = model_kwargs.get("data_aug", {})
    wrapper_kwargs = model_kwargs.get("wrapper_kwargs", {})
    pretrain_kwargs = model_kwargs.get("pretrain_kwargs", {})
    dset, preprocessor = make_datasets(cfgs_data, cfgs_data_aug, world_size, rank)
    args_eval["frameskip"] = cfgs_data["custom"]["frameskip"]
    args_eval["work_dir"] = folder
    checkpoint = args_eval["model_kwargs"].get("checkpoint")
    model = init_module(
        folder=pretrain_folder,
        checkpoint=checkpoint,
        module_name=module_name,
        model_kwargs=pretrain_kwargs,
        wrapper_kwargs=wrapper_kwargs,
        cfgs_data=cfgs_data,
        device=device,
        dset=dset,
        preprocessor=preprocessor,
    )
    log.info("Loaded encoder and predictor")

    # -- Launch eval
    main_distributed_episodes_eval(args_eval, model=model, dset=dset, preprocessor=preprocessor, rank=rank)


def main_distributed_episodes_eval(cfg: dict, model=None, dset=None, preprocessor=None, rank=0, device="cuda:0"):
    """
    Should work with one or more GPUs, even with distribute_multitask_eval=False.
    If world_size > 1 and distribute_multitask_eval=False, will all have same task_indices
    and potentially different results if we have a seed_shift in local_rngs but only rank 0
    will be logged. world_size=1 and distribute_multitask_eval=True should also work.
    To avoid timeout when gathering results from all ranks we create dummy episodes, The logic
    raises an assertion error in case the total_evals_episodes = tasks * eval_episodes < world_size.
    """
    # Setup the config
    start_time = time()
    cfg = OmegaConf.create(cfg)
    cfg = parse_cfg(cfg)
    set_seed(cfg.meta.seed)
    cfg.rank = rank
    cfg.world_size = dist.get_world_size()
    cfg.device = device
    cfg.num_active_gpus = cfg.world_size
    cfg.active_ranks = [i for i in range(cfg.world_size)]
    log.info(f"{cfg.active_ranks=}")
    if cfg.rank == 0:
        log.info(f"{'Work dir:'} {cfg.work_dir}")
    cfg.task_specification.goal_source = cfg.task_specification.get("goal_source", "expert")
    # DEFINE cfg.action_ratio := simu_actions / wm_fw_passes
    # TODO, replace all mentions of cfg.frameskip by cfg.action_ratio in the logging logic
    # of this script
    if cfg.planner.repeat_actskip:
        cfg.action_ratio = 1
    else:
        cfg.action_ratio = cfg.frameskip // model.action_skip
    log.info("First env creation just to define cfg.action_dim")
    env = make_env(cfg)  # needed here to define cfg.action_dim

    # We assume we have more episodes than GPUs
    cfg.planner.distribute_planner = False
    cfg.local_seed = cfg.meta.seed
    if cfg.distributed.distribute_multitask_eval:
        # Set a unique seed for the sampler based on the process rank
        if cfg.distributed.seed_shift == "horizon_1000":
            seed_shift = cfg.planner.horizon * 1000
        else:
            if isinstance(cfg.distributed.seed_shift, int) or isinstance(cfg.distributed.seed_shift, float):
                seed_shift = cfg.distributed.seed_shift
            else:
                ValueError("cfg.distributed.seed_shift does not have correct format")
        # We do not want to put local rng samplers in mujoco envs so put a different
        # global seed for each process, to ensure independence of environments
        cfg.local_seed += cfg.rank * seed_shift
        if not cfg.distributed.local_rng_samplers:
            set_seed(cfg.local_seed)
            log.info(f"Local Seed={cfg.local_seed} set for entire eval for rank {cfg.rank}")
        else:
            # In gc_planning will be used to seed envs
            log.info(f"Initialized local rng with seed {cfg.local_seed} for rank {cfg.rank}")

    if cfg.meta.quick_debug:
        log.info("Quick debug mode enabled.")
        cfg.meta.eval_episodes = 1
        cfg.planner.iterations = 2
        cfg.planner.num_samples = 2
        cfg.planner.num_elites = 2
        cfg.logging.tqdm_silent = False
    if cfg.planner.planner_name in ["cem", "mppi", "nevergrad"]:
        assert cfg.planner.num_elites <= cfg.planner.num_samples, "num_elites should be <= num_samples"
        assert cfg.planner.num_elites > 1, "num_elites should be > 1"

    # -------------------------
    # Build Logger, Agent, and start the evaluation loop
    logger = Logger(cfg)

    agent = GC_Agent(cfg, model, dset=dset, preprocessor=preprocessor)
    cfg.task_indices, cfg.episodes_per_task = compute_task_distribution(cfg)
    log.info(f"Rank {cfg.rank}: \n {cfg.task_indices=} \n {cfg.episodes_per_task=}")
    # The multitask wrapper allows to iterate over the task-specific envs
    env = make_env(cfg)
    evaluator = PlanEvaluator(cfg, agent)
    results = dict()
    processed_episodes = set()
    for task_pos, (task_idx, episodes) in enumerate(zip(cfg.task_indices, cfg.episodes_per_task)):
        (
            ep_rewards,
            ep_successes,
            ep_expert_successes,
            ep_success_dists,
            ep_end_distances,
            ep_end_distances_xyz,
            ep_end_distances_orientation,
            ep_end_distances_closure,
            ep_times,
            ep_state_distances,
            ep_total_lpips,
            ep_total_emb_l2,
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        # We want each episode to be independent from the others, even of the same task
        for i, ep in enumerate(episodes):
            log.info(
                f"Evaluating task {cfg.tasks[task_idx]}, episode ({i + 1}/{len(cfg.episodes_per_task[task_pos])})"
            )
            episode_start_time = time()
            (
                expert_success,
                success,
                ep_reward,
                success_dist,
                end_distance,
                end_distance_xyz,
                end_distance_orientation,
                end_distance_closure,
                state_dist,
                total_lpips,
                total_emb_l2,
            ) = evaluator.eval(cfg, agent, env, task_idx=task_idx, ep=ep)
            dist.barrier()  # should not provoke any error
            episode_end_time = time()
            # Check for duplicate task and episode index
            if (task_idx, ep) in processed_episodes:
                continue  # Skip duplicate dummy episodes logging
            processed_episodes.add((task_idx, ep))
            ep_rewards.append(ep_reward)
            ep_successes.append(success)
            ep_expert_successes.append(expert_success)
            ep_success_dists.append(success_dist)
            ep_end_distances.append(end_distance)
            ep_end_distances_xyz.append(end_distance_xyz)
            ep_end_distances_orientation.append(end_distance_orientation)
            ep_end_distances_closure.append(end_distance_closure)
            ep_times.append(episode_end_time - episode_start_time)
            ep_state_distances.append(state_dist)
            ep_total_lpips.append(total_lpips)
            ep_total_emb_l2.append(total_emb_l2)
        # Mean over episodes for each task
        results.update(
            {
                f"episode_reward+{cfg.tasks[task_idx]}": (np.nansum(ep_rewards), len(ep_rewards)),
                f"episode_success+{cfg.tasks[task_idx]}": (np.nansum(ep_successes), len(ep_successes)),
                f"ep_expert_succ+{cfg.tasks[task_idx]}": (np.nansum(ep_expert_successes), len(ep_expert_successes)),
                f"ep_succ_dist+{cfg.tasks[task_idx]}": (np.nansum(ep_success_dists), len(ep_success_dists)),
                f"ep_end_dist+{cfg.tasks[task_idx]}": (np.nansum(ep_end_distances), len(ep_end_distances)),
                f"ep_end_dist_xyz+{cfg.tasks[task_idx]}": (np.nansum(ep_end_distances_xyz), len(ep_end_distances_xyz)),
                f"ep_end_dist_orientation+{cfg.tasks[task_idx]}": (
                    np.nansum(ep_end_distances_orientation),
                    len(ep_end_distances_orientation),
                ),
                f"ep_end_dist_closure+{cfg.tasks[task_idx]}": (
                    np.nansum(ep_end_distances_closure),
                    len(ep_end_distances_closure),
                ),
                f"ep_time+{cfg.tasks[task_idx]}": (np.nansum(ep_times), len(ep_times)),
                f"ep_state_dist+{cfg.tasks[task_idx]}": (np.nansum(ep_state_distances), len(ep_state_distances)),
                f"ep_total_lpips+{cfg.tasks[task_idx]}": (np.nansum(ep_total_lpips), len(ep_total_lpips)),
                f"ep_total_emb_l2+{cfg.tasks[task_idx]}": (np.nansum(ep_total_emb_l2), len(ep_total_emb_l2)),
            }
        )

    if cfg.distributed.distribute_multitask_eval:
        all_results = [None] * cfg.world_size
        log.info(f"{rank=}: {results=}")
        if rank == 0:
            dist.gather_object(results, object_gather_list=all_results, dst=0)
        else:
            dist.gather_object(results, object_gather_list=None, dst=0)
            combined_results = {}
        if rank == 0:
            combined_results = aggregate_results(cfg, all_results)
            log.info(f"{combined_results=}")
    else:
        combined_results = {key: value[0] / value[1] if value[1] > 0 else 0 for key, value in results.items()}
    if cfg.rank == 0:
        metrics = {"total_time": time() - start_time}
        # Create average over tasks
        logger.pprint_multitask(combined_results | metrics, cfg)
        logger.log(combined_results | metrics, multitask=cfg.task_specification.multitask)


def init_module(
    folder,
    checkpoint,
    module_name,
    model_kwargs,
    device,
    cfgs_data=None,
    wrapper_kwargs=None,
    dset=None,
    preprocessor=None,
):
    """
    Build (frozen) model and initialize from pretrained checkpoint
    """
    model = importlib.import_module(f"{module_name}").init_module(
        folder=folder,
        checkpoint=checkpoint,
        model_kwargs=model_kwargs,
        device=device,
        dset=dset,
        preprocessor=preprocessor,
        cfgs_data=cfgs_data,
        wrapper_kwargs=wrapper_kwargs,
    )
    return model


def make_datasets(cfgs_data, cfgs_data_aug, world_size=1, rank=0):
    img_size = cfgs_data["img_size"]
    transform = make_transforms(
        img_size=img_size,
        normalize=cfgs_data_aug["normalize"],
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
    )
    inverse_transform = make_inverse_transforms(
        img_size=img_size,
        **cfgs_data_aug,
    )
    log.info(f"{inverse_transform.std=}, {inverse_transform.mean=}")

    # Extract nested structures (matching train.py structure)
    cfgs_validation = cfgs_data.get("validation", {})
    cfgs_loader = cfgs_data.get("loader", {})
    cfgs_custom = cfgs_data.get("custom", {})
    cfgs_droid = cfgs_data.get("droid", {})

    datasets = cfgs_data.get("datasets", [])
    dataset_paths = get_dataset_paths(datasets)
    val_datasets = cfgs_validation.get("val_datasets", [])
    val_dataset_paths = get_dataset_paths(val_datasets) if val_datasets else None

    # Prepare data kwargs from config, flattening nested structures and filtering out non-init_data fields
    excluded_keys = [
        "datasets",
        "val_datasets",
        "img_size",
        # subfields of cfgs_data
        "validation",
        "loader",
        "custom",
        "droid",
    ]
    data_kwargs = {k: v for k, v in cfgs_data.items() if k not in excluded_keys}

    data_kwargs.update(cfgs_validation)
    data_kwargs.update(cfgs_loader)
    data_kwargs.update(cfgs_custom)
    data_kwargs.update(cfgs_droid)

    data_kwargs.update(
        {
            "data_paths": dataset_paths,
            "val_data_paths": val_dataset_paths,
            "transform": transform,
            "world_size": world_size,
            "rank": rank,
            "num_workers": 0,
            "filter_first_episodes": 20,
            # robocasa
            "output_rcasa_state": True,
            "output_rcasa_info": True,
        }
    )

    (
        dataset,
        val_dataset,
        traj_dataset,
        val_traj_dataset,
        unsupervised_loader,
        val_unsupervised_loader,
        unsupervised_sampler,
        viz_val_data_loader,
    ) = init_data(**data_kwargs)

    dset = val_traj_dataset
    data_preprocessor = Preprocessor(
        action_mean=dset.action_mean,
        action_std=dset.action_std,
        state_mean=dset.state_mean,
        state_std=dset.state_std,
        proprio_mean=dset.proprio_mean,
        proprio_std=dset.proprio_std,
        transform=transform,
        inverse_transform=inverse_transform,
    )
    return dset, data_preprocessor
