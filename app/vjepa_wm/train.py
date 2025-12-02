# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass
import time
from collections import defaultdict

import imageio
import matplotlib
import numpy as np
import submitit
import torch
import torch.multiprocessing as mp
import wandb
from einops import rearrange
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from app.plan_common.datasets.preprocessor import Preprocessor
from app.plan_common.datasets.transforms import make_inverse_transforms, make_transforms
from app.plan_common.datasets.utils import init_data
from app.plan_common.models.lpips import LPIPS
from app.plan_common.models.wm_heads import (
    WorldModelPoseReadoutHead,
    WorldModelRewardReadoutHead,
    WorldModelViTImageHead,
)
from src.utils.yaml_utils import dump_yaml, convert_to_dict_recursive, expand_env_vars
from app.vjepa_wm.utils import (
    build_plan_eval_args,
    build_unroll_decode_eval_args,
    clean_state_dict,
    init_opt,
    init_opt_muon,
    init_video_model,
    load_checkpoint,
)
from app.vjepa_wm.video_wm import VideoWM
from evals.main_distributed import launch_evals_with_parsed_args as launch_evals
from src.datasets.utils.utils import get_dataset_paths
from src.utils.cluster import slurm_account_partition_and_qos
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer

# --
log_timings = True
log_freq = 10
checkpoint_freq = 1
DEFAULT_EVAL_FREQ = 50
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


logger = get_logger(__name__)


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #
    args = expand_env_vars(args)
    folder = args.get("folder")
    # -- META
    cfgs_meta = args.get("meta")
    load_model = cfgs_meta.get("load_checkpoint") or resume_preempt
    load_opt_scale_epoch = cfgs_meta.get("load_opt_scale_epoch", False)
    freeze_encoder = cfgs_meta.get("freeze_encoder", True)
    r_file = cfgs_meta.get("read_checkpoint", None)
    pretrained_path = cfgs_meta.get("pretrained_path", "none")
    seed = cfgs_meta.get("seed", _GLOBAL_SEED)

    eval_freq = cfgs_meta.get("eval_freq", DEFAULT_EVAL_FREQ)
    plan_only_eval_mode = cfgs_meta.get("plan_only_eval_mode", False)
    unroll_decode_eval_only_mode = cfgs_meta.get("unroll_decode_eval_only_mode", False)
    light_eval_only_mode = cfgs_meta.get("light_eval_only_mode", False)

    # -- LIGHT EVALS (keep as subconfigs, extract only frequently-checked flags)
    cfgs_data_traj_rollout_eval = cfgs_meta.get("data_traj_rollout_eval", {})
    cfgs_energy_landscape_eval = cfgs_meta.get("energy_landscape_eval", {})
    do_data_traj_rollout_eval = cfgs_data_traj_rollout_eval.get("do_data_traj_rollout_eval", False)
    do_energy_landscape_eval = cfgs_energy_landscape_eval.get("do_energy_landscape_eval", False)
    data_traj_decode_gt = cfgs_data_traj_rollout_eval.get("data_traj_decode_gt", False)

    light_eval_freq = cfgs_meta.get("light_eval_freq", 100)
    save_every_freq = cfgs_meta.get("save_every_freq", -1)
    quick_debug = cfgs_meta.get("quick_debug", False)
    which_dtype = cfgs_meta.get("dtype")
    logger.info(f"{which_dtype=}")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    # -- EVALS
    cfgs_plan_evals = args.get("evals", None)
    cfgs_unroll_decode_evals = args.get("unroll_decode_evals", None)

    # -- MODEL (extract only fields needed outside init_video_model)
    cfgs_model = args.get("model")

    # Extract heads config
    cfgs_heads = cfgs_model.get("heads_cfg", {})
    heads_architectures = cfgs_heads.get("architectures", {})
    pretrain_dec_path = cfgs_heads.get("pretrain_dec_path", None)
    new_path_heads = cfgs_heads.get("new_path_heads", False)

    # Fields needed for training loop
    rollout_cfg = cfgs_model.get("rollout_cfg", {})
    rollout_steps = rollout_cfg.get("rollout_steps", 0)
    train_rollout_prefixes = rollout_cfg.get("train_rollout_prefixes", "random")
    rollout_stop_gradient = rollout_cfg.get("rollout_stop_gradient", True)
    ctxt_window_train_rollout = rollout_cfg.get("ctxt_window_train_rollout", 8)
    do_parallel_rollout = rollout_cfg.get("do_parallel_rollout", False)
    do_sequential_rollout = rollout_cfg.get("do_sequential_rollout", True)
    sampling_rollout = rollout_cfg.get("sampling_rollout", False)
    prepend_gt_rollout_parallel = rollout_cfg.get("prepend_gt", False)

    sampling_scheduler_cfg = rollout_cfg.get("sampling_scheduler", {})
    sampling_scheduler_type = sampling_scheduler_cfg.get("type", "linear")
    sampling_scheduler_start = sampling_scheduler_cfg.get("start", 0.0)
    sampling_scheduler_end = sampling_scheduler_cfg.get("end", 0.0)

    # Derived values needed for dimensions (computed values used across the code)
    action_tokens = cfgs_model.get("action_encoder", {}).get("action_tokens", 1)
    proprio_tokens = cfgs_model.get("proprio_encoder", {}).get("proprio_tokens", 1)
    action_emb_dim = cfgs_model.get("action_encoder", {}).get("action_emb_dim", 0)
    proprio_emb_dim = cfgs_model.get("proprio_encoder", {}).get("proprio_emb_dim", 0)
    use_proprio = proprio_tokens > 0 or proprio_emb_dim > 0
    use_action = action_tokens > 0 or action_emb_dim > 0
    tubelet_size_enc = cfgs_model.get("tubelet_size_enc", 2)

    cfgs_wm_encoding = cfgs_model.get("wm_encoding", {})

    if cfgs_wm_encoding.get("dup_image", False):
        assert tubelet_size_enc == 1, "Batchify video only works with tubelet_size_enc=1"

    # -- DATA (extract only fields needed outside init_data)
    cfgs_data = args.get("data")
    cfgs_validation = cfgs_data.get("validation", {})
    cfgs_loader = cfgs_data.get("loader", {})
    cfgs_custom = cfgs_data.get("custom", {})
    cfgs_droid = cfgs_data.get("droid", {})

    # Compute dataset paths
    datasets = cfgs_data.get("datasets", [])
    datasets_weights = cfgs_data.get("datasets_weights", None)
    if datasets_weights is not None:
        assert len(datasets_weights) == len(datasets), "Must have one sampling weight specified for each dataset"

    dataset_type = cfgs_data.get("dataset_type", "custom")
    if dataset_type.lower() == "mixed_dataset":
        dataset_paths = datasets
    else:
        dataset_paths = get_dataset_paths(datasets)

    val_datasets = cfgs_validation.get("val_datasets", [])
    val_dataset_paths = get_dataset_paths(val_datasets) if val_datasets else None

    # val_datasets_1 subconfig (for second validation set)
    val_datasets_1 = cfgs_validation.get("val_datasets_1", None)
    val_datasets_1_paths = None
    if val_datasets_1 is not None:
        val_datasets_1_paths = get_dataset_paths(val_datasets_1.get("names"))

    # Fields used outside init_data
    frameskip = cfgs_custom.get("frameskip", True)
    action_skip = cfgs_custom.get("action_skip", 1)
    state_skip = cfgs_custom.get("state_skip", 1)
    img_size = cfgs_data.get("img_size", 256)
    num_workers = cfgs_loader.get("num_workers", 1)
    filter_first_episodes = cfgs_custom.get("filter_first_episodes", None)
    val_viz_rank0_loader = cfgs_validation.get("val_viz_rank0_loader", False)

    # -- DATA AUGS
    cfgs_data_aug = args.get("data_aug")

    # -- LOSS
    cfgs_loss = args.get("loss")

    # -- OPTIMIZATION (simplify main_optimizer logic)
    cfgs_opt = args.get("optimization")
    train_heads = cfgs_opt["train_heads"]

    main_optimizer = cfgs_opt["main_optimizer"]
    if main_optimizer == "transition_model":
        num_epochs = cfgs_opt["transition_model"]["num_epochs"]
        ipe = cfgs_opt["transition_model"]["iterations_per_epoch"]
        train_predictor = True
        train_heads_on_predictor = False
    else:  # image_head | state_head | reward_head
        num_epochs = cfgs_opt["heads"][main_optimizer]["num_epochs"]
        ipe = cfgs_opt["heads"][main_optimizer]["iterations_per_epoch"]
        train_predictor = cfgs_opt["heads"]["train_predictor"]
        train_heads_on_predictor = cfgs_opt["heads"]["train_heads_on_predictor"]

    # -- LOGGING
    cfgs_logging = args.get("logging")
    tag = cfgs_logging.get("write_tag", "jepa")
    latest_format = cfgs_logging.get("latest_format", "pth.tar")
    cfgs_wandb = cfgs_logging.get("wandb")

    if light_eval_only_mode:
        light_eval_freq = 1
    if plan_only_eval_mode or light_eval_only_mode or quick_debug or unroll_decode_eval_only_mode:
        filter_first_episodes, num_workers = 10, 0

    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass
    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # -- log/checkpointing paths
    train_log_file = os.path.join(folder, f"log_r{rank}.csv")
    pref_tag = f"{tag}-" if tag else ""
    latest_file = pref_tag + f"latest.{latest_format}"
    latest_path = os.path.join(folder, latest_file)
    finetune = pretrained_path is not None
    logger.info(f"Finetuning: {finetune}")
    # finetune covers the case of training heads on top of frozen transition_model
    # if cfgs_model["pretrained_path"] is None, i.e. training head just on encoder
    head_training_mode = main_optimizer in ["image_head", "state_head"]
    resume = os.path.exists(latest_path)
    resume_finetune = os.path.exists(latest_path) and finetune
    resume_latest = os.path.exists(latest_path) and not finetune
    logger.info(f"Resuming finetuning or heads training: {resume_finetune}")
    load_path = None
    if load_model:
        if resume_finetune:
            load_path = os.path.join(folder, r_file) if r_file is not None else latest_path
            load_opt_scale_epoch = not head_training_mode
        elif resume_latest:
            load_path = latest_path
            load_opt_scale_epoch = not head_training_mode
        else:  # not resuming, i.e. not os.path.exists(latest_path)
            load_path = pretrained_path
        if load_path is None or not os.path.exists(load_path):
            load_path = None
            load_model = False

    train_csv_logger, eval_csv_logger = None, None

    def create_csv_logger(losses, total_stats, train=False):
        if train:
            csv_log_file = train_log_file
        else:
            if light_eval_only_mode:
                csv_log_file = os.path.join(folder, f"light_eval_only_eval_r{rank}.csv")
            else:
                csv_log_file = os.path.join(folder, f"eval_r{rank}.csv")
        # Get all unique keys from eval_losses and eval_total_stats
        excluded_keys = [
            "eval_data/image_rollouts",
            "eval_data/image_rollouts_noisy_actions",
            "eval_data/image_animated_rollout",
        ]
        all_keys = {key for key in list(losses.keys()) + list(total_stats.keys()) if key not in excluded_keys}
        # Sort keys lexicographically for consistent order
        sorted_keys = sorted(all_keys)
        # Create a format tuple for each key
        new_columns = [("%.5f", key) for key in sorted_keys]
        # Initialize the logger with new columns
        if train:
            global train_csv_logger_columns
            train_csv_logger_columns = ["epoch", "itr", "loss", "gpu-time(ms)", "iter-time(ms)"] + sorted_keys
        else:
            global eval_csv_logger_columns
            eval_csv_logger_columns = ["epoch", "itr"] + sorted_keys
        return CSVLogger(csv_log_file, ("%d", "epoch"), ("%d", "itr"), *new_columns)

    # -- init data-loaders/samplers
    transform = make_transforms(
        img_size=img_size,
        **cfgs_data_aug,
    )
    inverse_transform = make_inverse_transforms(img_size=img_size, **cfgs_data_aug)

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
    # Add fields from nested structures
    data_kwargs.update(cfgs_validation)
    data_kwargs.update(cfgs_loader)
    data_kwargs.update(cfgs_custom)
    data_kwargs.update(cfgs_droid)
    # Add computed/override parameters
    data_kwargs.update(
        {
            "data_paths": dataset_paths,
            "val_data_paths": val_dataset_paths,
            "transform": transform,
            "world_size": world_size,
            "rank": rank,
            # overridden by quick_debug
            "filter_first_episodes": filter_first_episodes,  # Potentially overridden by mode
            "num_workers": num_workers,
        }
    )

    val_data_iters = []
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
    val_data_iters.append((val_dataset, val_traj_dataset, val_unsupervised_loader))

    if val_datasets_1:
        # Reuse data_kwargs and override val_datasets_1-specific fields
        data_kwargs_1 = data_kwargs.copy()
        data_kwargs_1.update(
            {
                "droid_fraction": 1,
                "val_droid_fraction": 1,
                "data_paths": val_datasets_1_paths,
                "val_data_paths": val_datasets_1_paths,
                "batch_size": val_datasets_1.get("batch_size", 4),
                "drop_last": val_datasets_1.get("drop_last", True),
                "fps": val_datasets_1.get("fps", 4),
                "dataset_fpcs": val_datasets_1.get("fpcs", [8]),
                "val_dataset_fpcs": val_datasets_1.get("fpcs", [8]),
                "camera_views": val_datasets_1.get("camera_views", ["exterior_image_2_left"]),
                "droid_to_rcasa_action_format": val_datasets_1.get("droid_to_rcasa_action_format", 1),
            }
        )
        _, val_dataset_1, _, val_traj_dataset_1, _, val_unsupervised_loader_1, _, viz_val_data_loader = init_data(
            **data_kwargs_1
        )
        val_data_iters.append((val_dataset_1, val_traj_dataset_1, val_unsupervised_loader_1))
    if dataset_type == "custom" and traj_dataset is not None:
        preprocessor = Preprocessor(
            action_mean=traj_dataset.action_mean,
            action_std=traj_dataset.action_std,
            state_mean=traj_dataset.state_mean,
            state_std=traj_dataset.state_std,
            proprio_mean=traj_dataset.proprio_mean,
            proprio_std=traj_dataset.proprio_std,
            transform=transform,
            inverse_transform=inverse_transform,
        )
    else:
        preprocessor = None

    _dlen = len(unsupervised_loader)
    if ipe is None:
        ipe = _dlen
    logger.info(f"iterations per epoch/dataset length: {ipe}/{_dlen}")
    if main_optimizer == "transition_model":
        cfgs_opt["transition_model"]["iterations_per_epoch"] = ipe
    elif main_optimizer == "image_head":
        cfgs_opt["heads"]["image_head"]["iterations_per_epoch"] = ipe
    elif main_optimizer == "state_head":
        cfgs_opt["heads"]["state_head"]["iterations_per_epoch"] = ipe

    # Logger
    class Trainer:
        def __init__(self, config):
            if quick_debug:
                config["debug"] = quick_debug
            self.config = config
            self.ipe = ipe
            self.use_wandb = config.get("use_wandb", False)
            self.disable_wandb_media = config.get("disable_wandb_media", False)
            self.log_media_locally = config.get("log_media_locally", False)
            self.local_log_dir = None
            if self.log_media_locally and rank == 0:
                self.local_log_dir = os.path.join(folder, "local_logs")
                os.makedirs(self.local_log_dir, exist_ok=True)
            if self.use_wandb and rank == 0:
                project_name = config.get("project", "vjepa_wm") if not config["debug"] else "vjepa_wm_debug"
                wandb_run_id_file = os.path.join(folder, "wandb_run_id.txt")
                if os.path.exists(wandb_run_id_file):
                    with open(wandb_run_id_file, "r") as f:
                        wandb_run_id = f.read().strip()
                    wandb.init(project=project_name, id=wandb_run_id, resume="allow", dir=folder)
                    logger.info(f"Resuming Wandb run {wandb_run_id}")
                else:
                    wandb.init(project=project_name, dir=folder)
                    with open(wandb_run_id_file, "w") as f:
                        f.write(wandb.run.id)
                wandb.run.name = os.path.basename(folder)
                self.job_set = set()

        def log(self, epoch, itr, losses, total_stats, eval_losses=None, eval_total_stats=None, image_stats=None):
            log_dict = {
                "epoch": epoch + 1,
                "itr": itr,
            }
            for key, value in losses.items():
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().item()
                log_dict[key] = value
            for key, value in total_stats.items():
                log_dict[key] = value
            if eval_losses is not None:
                for key, value in eval_losses.items():
                    if isinstance(value, torch.Tensor):
                        value = value.detach().cpu().item()
                    log_dict[key] = value
            if eval_total_stats is not None:
                for key, value in eval_total_stats.items():
                    log_dict[key] = value
            if image_stats:  # not None or nonempty
                if self.log_media_locally and rank == 0:
                    self.log_media_local(image_stats, epoch, itr)
                if not self.disable_wandb_media:
                    log_dict.update(image_stats)
            if "loss" in log_dict.keys():
                logger.info("[%d, %5d] " "loss: %.3f | " % (epoch + 1, itr, log_dict["loss"]))
            if self.use_wandb and rank == 0:
                wandb.log(log_dict)

        def log_media_local(self, image_stats, epoch=None, itr=None):
            step = epoch * ipe + itr
            for key, value in image_stats.items():
                subfolder = os.path.join(self.local_log_dir, "/".join(key.split("/")))
                os.makedirs(subfolder, exist_ok=True)
                filename = f"{step}.gif" if isinstance(value, wandb.Video) else f"{step}.pdf"
                if isinstance(value, wandb.Video):
                    frames = value._prepare_video(value.data)
                    duration = 1.0 / 10  # assuming 10 FPS
                    imageio.mimsave(os.path.join(subfolder, filename), frames, duration=duration, loop=0)
                elif isinstance(value, wandb.Image):
                    value.image.save(os.path.join(subfolder, filename))
                elif isinstance(value, matplotlib.figure.Figure):
                    value.savefig(os.path.join(subfolder, filename), bbox_inches=None)

    trainer = Trainer(cfgs_wandb)

    # -- init model
    if use_action:
        actions_per_vid_feat = tubelet_size_enc * frameskip // action_skip
        model_action_dim = traj_dataset.action_dim * tubelet_size_enc * frameskip // action_skip
    else:
        actions_per_vid_feat, model_action_dim = None, None
    if use_proprio:
        proprio_multiplier = tubelet_size_enc * frameskip // state_skip
        model_proprio_dim = traj_dataset.proprio_dim * tubelet_size_enc // state_skip
    else:
        proprio_multiplier, model_proprio_dim = None, None

    # Prepare model kwargs by flattening nested configs and filtering out non-init_video_model fields
    excluded_keys = [
        "rollout_cfg",
        "heads_cfg",
        "pretrained_path",
        "visual_encoder",
        "action_encoder",
        "proprio_encoder",
        "predictor",
        "wm_encoding",
        "attn",
    ]
    model_kwargs = {k: v for k, v in cfgs_model.items() if k not in excluded_keys}
    if "visual_encoder" in cfgs_model:
        model_kwargs.update(cfgs_model["visual_encoder"])
    if "action_encoder" in cfgs_model:
        model_kwargs.update(cfgs_model["action_encoder"])
    if "proprio_encoder" in cfgs_model:
        model_kwargs.update(cfgs_model["proprio_encoder"])
    if "predictor" in cfgs_model:
        model_kwargs.update(cfgs_model["predictor"])
    model_kwargs.update(
        {
            "device": device,
            "img_size": img_size,
            "action_dim": model_action_dim,  # Computed from dataset
            "proprio_dim": model_proprio_dim,  # Computed from dataset
            "cfgs_attn_pattern": cfgs_model.get("attn", None),  # Pass attn subconfig
            "use_proprio": use_proprio,  # Computed derived value
            "use_action": use_action,  # Computed derived value
        }
    )
    predictor, encoder, action_encoder, proprio_encoder = init_video_model(**model_kwargs)

    heads = {}
    if train_heads or pretrain_dec_path is not None:
        if "image_head" in heads_architectures:
            image_head_type = heads_architectures["image_head"]["kind"]
            if image_head_type is not None and image_head_type.lower() != "none":
                if image_head_type == "vit":
                    decoder = WorldModelViTImageHead(
                        head_config=dict(heads_architectures["image_head"]["config"]),
                        inverse_transform=inverse_transform,
                        device=device,
                    )
                heads["image_head"] = decoder
        if "state_head" in heads_architectures:
            state_decoder = WorldModelPoseReadoutHead(
                head_config=dict(heads_architectures["state_head"]["config"]), device=device
            )
            heads["state_head"] = state_decoder
        if "reward_head" in heads_architectures:
            reward_decoder = WorldModelRewardReadoutHead(
                head_config=dict(heads_architectures["reward_head"]["config"]), device=device
            )
            heads["reward_head"] = reward_decoder

    # -- init optimizer and scheduler
    if train_predictor and predictor is not None:
        if cfgs_opt["transition_model"].get("use_muon", False):
            optimizer, scaler, scheduler, wd_scheduler = init_opt_muon(
                predictor=predictor,
                action_encoder=action_encoder,
                proprio_encoder=proprio_encoder,
                encoder=encoder,
                freeze_encoder=freeze_encoder,
                **cfgs_opt["transition_model"],
            )
        else:
            optimizer, scaler, scheduler, wd_scheduler = init_opt(
                predictor=predictor,
                action_encoder=action_encoder,
                proprio_encoder=proprio_encoder,
                encoder=encoder,
                freeze_encoder=freeze_encoder,
                **cfgs_opt["transition_model"],
            )
        clip_grad = cfgs_opt["transition_model"]["clip_grad"]
        use_radamw = cfgs_opt["transition_model"]["use_radamw"]
        if sampling_scheduler_type == "linear":
            # linear decay from sampling_scheduler_start to sampling_scheduler_end
            rollout_sampling_scheduler = (
                sampling_scheduler_start - i * (sampling_scheduler_start - sampling_scheduler_end) / (ipe * num_epochs)
                for i in range(int(ipe * num_epochs) + 1)
            )
        elif sampling_scheduler_type == "exponential":
            # exponential decay from sampling_scheduler_start to sampling_scheduler_end
            rollout_sampling_scheduler = (
                sampling_scheduler_start
                * (sampling_scheduler_end / sampling_scheduler_start) ** (i / (ipe * num_epochs))
                for i in range(int(ipe * num_epochs) + 1)
            )
        elif sampling_scheduler_type == "sigmoid":
            rollout_sampling_scheduler = (
                sampling_scheduler_start
                + (sampling_scheduler_end - sampling_scheduler_start)
                * (1 / (1 + np.exp(-10 * (i / (ipe * num_epochs) - 0.5))))
                for i in range(int(ipe * num_epochs) + 1)
            )
    else:
        optimizer, scaler, scheduler, wd_scheduler, clip_grad, use_radamw = None, None, None, None, None, None
    if train_heads:
        for name, head in heads.items():
            head.init_opt(**dict(cfgs_opt["heads"][name]))

    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        # to resume predictor or head training
        load_heads = heads and not pretrain_dec_path and resume
        logger.info(f"Load heads: {load_heads}")
        (
            predictor,
            action_encoder,
            proprio_encoder,
            heads,
            optimizer,
            scaler,
            start_epoch,
        ) = load_checkpoint(
            r_path=load_path,
            predictor=predictor,
            action_encoder=action_encoder,
            proprio_encoder=proprio_encoder,
            heads=heads,
            opt=optimizer,
            scaler=scaler,
            load_opt_scale_epoch=load_opt_scale_epoch,
            load_heads=load_heads,
            train_heads=train_heads,
            train_predictor=train_predictor,
        )
        # Only resume the schedulers if we resume a pretraining or a finetuning
        # Not if we start a finetuning: we reset them
        if load_opt_scale_epoch and scheduler is not None and wd_scheduler is not None:
            for _ in range(start_epoch * ipe):
                scheduler.step()
                wd_scheduler.step()
        if light_eval_only_mode:
            start_epoch -= 1

    # Load pretrained heads from pretrain_dec_path
    if pretrain_dec_path is not None:
        for name, head in heads.items():
            if new_path_heads[name]:
                head_path = pretrain_dec_path[name].removesuffix(".pth.tar") + "_" + name + ".pth.tar"
                head.load_checkpoint(head_path)
                logger.info(f"loaded pretrained head named {name}")
            else:
                checkpoint = torch.load(pretrain_dec_path[name], map_location=torch.device("cpu"))
                epoch = checkpoint["epoch"]
                pretrained_dict = clean_state_dict(checkpoint[name])
                msg = head.model.load_state_dict(pretrained_dict, strict=False)
                logger.info(f"loaded pretrained head named {name} from epoch {epoch} with msg: {msg}")
                del checkpoint

    # DDP wrapping after loading state_dicts
    if not freeze_encoder:
        encoder = DDP(encoder, static_graph=False, find_unused_parameters=False)
    if train_predictor:
        if action_encoder is not None:
            action_encoder = DDP(action_encoder, static_graph=False, find_unused_parameters=False)
        if proprio_encoder is not None:
            proprio_encoder = DDP(proprio_encoder, static_graph=False, find_unused_parameters=False)
        predictor = DDP(predictor, static_graph=False, find_unused_parameters=False)
    for name in heads.keys():
        heads[name].model = DDP(heads[name].model, static_graph=False, find_unused_parameters=False)

    # Prepare VideoWM kwargs from config
    wm_kwargs = {
        "device": device,
        # Model components
        "encoder": encoder,
        "predictor": predictor,
        "action_encoder": action_encoder,
        "proprio_encoder": proprio_encoder,
        # Computed dimensions
        "action_dim": model_action_dim,
        "proprio_dim": model_proprio_dim,
        "use_proprio": use_proprio,
        "use_action": use_action,
        # From cfgs_model (pass directly from config)
        "action_tokens": action_tokens,
        "proprio_tokens": proprio_tokens,
        "grid_size": cfgs_model.get("grid_size", 14),
        "tubelet_size_enc": cfgs_model.get("tubelet_size_enc", 2),
        "action_conditioning": cfgs_model.get("action_conditioning", "token"),
        "proprio_encoding": cfgs_model.get("proprio_encoding", "feature"),
        "enc_type": cfgs_model["visual_encoder"].get("enc_type", "vjepa"),
        "pred_type": cfgs_model["predictor"].get("pred_type", "dino_wm"),
        "action_encoder_inpred": cfgs_model["action_encoder"].get("action_encoder_inpred", False),
        "proprio_encoder_inpred": cfgs_model["proprio_encoder"].get("proprio_encoder_inpred", False),
        **cfgs_wm_encoding,
        # From cfgs_data
        "action_skip": cfgs_data.get("action_skip", 1),
        "frameskip": cfgs_data.get("frameskip", 1),
        "img_size": cfgs_data.get("img_size", 256),
        # Heads
        "heads": heads,
        # Optimization
        "scaler": scaler,
        "optimizer": optimizer,
        "clip_grad": clip_grad,
        "mixed_precision": mixed_precision,
        "use_radamw": use_radamw,
        # Loss config (pass subconfig directly)
        "cfgs_loss": cfgs_loss,
    }
    world_model = VideoWM(**wm_kwargs)

    # -- Initialize LPIPS once for evaluation
    lpips = LPIPS().eval().to(device)

    def save_checkpoint(epoch, path):
        if rank != 0:
            return
        save_dict = {
            "predictor": world_model.predictor.state_dict() if world_model.predictor is not None else None,
            "opt": optimizer.state_dict() if optimizer is not None else None,
            "scaler": None if scaler is None else scaler.state_dict(),
            "epoch": epoch,
        }
        if world_model.action_encoder is not None and not cfgs_model["action_encoder"].get(
            "action_encoder_inpred", False
        ):
            save_dict.update({"action_encoder": world_model.action_encoder.state_dict()})
        if (
            world_model.proprio_encoder is not None
            and use_proprio
            and not cfgs_model["proprio_encoder"].get("proprio_encoder_inpred", False)
        ):
            save_dict.update({"proprio_encoder": world_model.proprio_encoder.state_dict()})
        if train_heads:
            for name, head in world_model.heads.items():
                head_path = path.removesuffix(".pth.tar") + "_" + name + ".pth.tar"
                head.save_checkpoint(epoch, head_path)
        try:
            torch.save(save_dict, path)
        except Exception as e:
            logger.info(f"Encountered exception when saving checkpoint: {e}")

    logger.info("Initializing loader...")
    train_loader = iter(unsupervised_loader)
    if val_data_iters != [(None, None, None)]:
        val_loader_iters = []
        for vl_dset, vl_traj_dset, vl_loader in val_data_iters:
            val_loader_iters.append(iter(vl_loader))

    if viz_val_data_loader is not None:
        viz_val_loader_iter = iter(viz_val_data_loader)

    def load_clips(sample):
        all_clips = []
        return sample[0][0].to(device, non_blocking=True), None, None

    def get_batch(train=True, idx=0):
        nonlocal train_loader, val_loader_iters
        if dataset_type == "custom":
            try:
                if train:
                    obs, action, state, reward = next(train_loader)
                else:
                    obs, action, state, reward = next(val_loader_iters[idx])
            except StopIteration as e:
                logger.info(f"Exception {e=}")
                logger.info(f"Exhausted data loaders idx {idx} with {train=}. Refreshing...")
                if train:
                    train_loader = iter(unsupervised_loader)
                    obs, action, state, reward = next(train_loader)
                else:
                    val_loader_iters[idx] = iter(val_data_iters[idx][2])
                    obs, action, state, reward = next(val_loader_iters[idx])
            for k in obs.keys():
                obs[k] = obs[k].to(device, dtype=dtype, non_blocking=True)
            action = action.to(device, dtype=dtype, non_blocking=True)
            state = state.to(device, dtype=dtype, non_blocking=True)
            reward = reward.to(device, dtype=dtype, non_blocking=True)
            return obs, action, state, reward, None, None
        else:
            try:
                if train:
                    sample = next(train_loader)
                else:
                    sample = next(val_loader_iters[idx])
            except StopIteration as e:
                logger.info(f"Exception {e=}")
                logger.info("Exhausted data loaders. Refreshing...")
                if train:
                    train_loader = iter(unsupervised_loader)
                    sample = next(train_loader)
                else:
                    val_loader_iters[idx] = iter(val_data_iters[idx][2])
                    sample = next(val_loader_iters[idx])
            all_clips, all_masks_enc, all_masks_pred = load_clips(sample)
            all_clips = {"visual": all_clips}
            return all_clips, None, None, None, all_masks_enc, all_masks_pred

    def get_viz_batch():
        """Get a visualization batch from the non-distributed validation loader (rank 0 only)"""
        nonlocal viz_val_loader_iter
        if rank != 0 or viz_val_loader_iter is None:
            return None, None, None, None, None, None
        if dataset_type == "custom":
            try:
                obs, action, state, reward = next(viz_val_loader_iter)
            except StopIteration as e:
                logger.info(f"Exception {e=}")
                logger.info("Exhausted viz data loader. Refreshing...")
                viz_val_loader_iter = iter(viz_val_data_loader)
                obs, action, state, reward = next(viz_val_loader_iter)
            for k in obs.keys():
                obs[k] = obs[k].to(device, dtype=torch.float32, non_blocking=True)
            action = action.to(device, dtype=torch.float32, non_blocking=True)
            state = state.to(device, dtype=torch.float32, non_blocking=True)
            reward = reward.to(device, dtype=torch.float32, non_blocking=True)
            return obs, action, state, reward, None, None
        else:
            try:
                sample = next(viz_val_loader_iter)
            except StopIteration as e:
                logger.info(f"Exception {e=}")
                logger.info("Exhausted viz data loader. Refreshing...")
                viz_val_loader_iter = iter(viz_val_data_loader)
                sample = next(viz_val_loader_iter)
            all_clips, all_masks_enc, all_masks_pred = load_clips(sample)
            all_clips = {"visual": all_clips}
            return all_clips, None, None, None, all_masks_enc, all_masks_pred

    # -- TRAINING LOOP
    if not (plan_only_eval_mode or unroll_decode_eval_only_mode):
        for epoch in range(start_epoch, num_epochs):
            logger.info("Epoch %d" % (epoch + 1))

            # -- update distributed-data-loader epoch
            unsupervised_sampler.set_epoch(epoch)

            loss_meter = AverageMeter()
            gpu_time_meter = AverageMeter()
            wall_time_meter = AverageMeter()

            for itr in range(ipe):
                itr_start_time = time.time()
                if quick_debug or light_eval_only_mode:
                    if itr > 5:
                        break

                def step_model(obs, action, state, reward, train=True):
                    rates = defaultdict(float)
                    if train:
                        if train_predictor:
                            rates["info/transition_model/lr"] = scheduler.step()
                            rates["info/transition_model/wd"] = wd_scheduler.step()
                        if train_heads:
                            for name, head in world_model.heads.items():
                                rates[f"info/{name}/lr"] = head.scheduler.step()
                                rates[f"info/{name}/wd"] = head.wd_scheduler.step()
                    else:
                        rates["info/transition_model/lr"] = 0.0
                        rates["info/transition_model/wd"] = 0.0
                        for name, head in world_model.heads.items():
                            rates[f"info/{name}/lr"] = 0.0
                            rates[f"info/{name}/wd"] = 0.0
                    # --

                    # Step 1. Forward
                    total_stats = {}
                    total_transition_loss = 0.0
                    total_head_loss = 0.0
                    if action is not None:
                        total_stats.update(
                            {
                                "act_mean": action.mean(),
                                "act_std": action.std(),
                                "act_min": action.min(),
                                "act_max": action.max(),
                            }
                        )
                    # 1. TRAIN PREDICTOR TO PREDICT ONE STEP IN THE FUTURE USING TEACHER FORCING
                    train_rollout_result = {}
                    parallel_rollout_result = {}
                    with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                        video_features, proprio_features, action_features = world_model.encode(obs, action)
                        if train_predictor and train and predictor is not None:
                            pred_video_features, pred_action_features, pred_proprio_features = (
                                world_model.forward_pred(
                                    video_features,
                                    action_features,
                                    proprio_features,
                                )
                            )
                            predictor_losses = world_model.compute_loss(
                                pred_video_features,
                                pred_proprio_features,
                                video_features,
                                proprio_features,
                                shift=1,
                            )
                        else:
                            pred_video_features, pred_proprio_features = None, None
                            predictor_losses = {}
                    predictor_loss = predictor_losses.get("loss", 0.0) / (rollout_steps + 1)
                    if train and train_predictor and predictor is not None:
                        total_transition_loss += predictor_loss
                    stats = defaultdict(list)
                    for k in predictor_losses:
                        stats[k].append(torch.tensor(predictor_losses[k]).unsqueeze(0))
                    # mean over 1 element in list
                    stats = {k: torch.stack(stats[k]).mean(0) for k in stats}
                    for k in stats:
                        for j in range(len(stats[k])):
                            train_rollout_result[f"train_rollout/{k}/{j+1}"] = stats[k][j].item()
                    # 2. TRAIN HEADS ON FEATURES FROM ENCODER
                    if "image_head" in world_model.heads and train_heads:
                        # Encoder produces features of videos normalized by mean and std
                        # So let's keep them for target of decoder loss
                        target_rgb_video = world_model.heads["image_head"].preprocess_rgb(
                            obs["visual"][:, ::tubelet_size_enc]
                        )
                        with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                            encoder_image_losses = world_model.heads["image_head"].compute_loss(
                                video_features,
                                target_rgb_video,
                                global_step=epoch * ipe + itr,
                            )
                        rates["info/image_head/examples_seen"] += video_features.shape[0] * video_features.shape[1]
                        encoder_image_losses = {k: v.mean() for k, v in encoder_image_losses.items()}
                        # the weight is like that because we train the head on encoder features and potentially on predictor too
                        loss = encoder_image_losses["loss"] / (train_heads_on_predictor * rollout_steps + 1)
                        if train:
                            total_head_loss += loss
                            world_model.heads["image_head"].backward(loss)
                        encoder_image_losses = {
                            "encoder_image_" + k: v.item() for k, v in encoder_image_losses.items()
                        }
                        total_stats.update(encoder_image_losses)
                    if "state_head" in world_model.heads and train_heads:
                        with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                            encoder_state_losses = world_model.heads["state_head"].compute_loss(
                                video_features, None, state[:, ::tubelet_size_enc]
                            )
                        rates["info/state_head/examples_seen"] += video_features.shape[0] * video_features.shape[1]
                        encoder_state_losses = {k: v.mean() for k, v in encoder_state_losses.items()}
                        loss = encoder_state_losses["loss"] / (train_heads_on_predictor * rollout_steps + 1)
                        if train:
                            total_head_loss += loss
                            world_model.heads["state_head"].backward(loss)
                        encoder_state_losses = {
                            "encoder_state_" + k: v.item() for k, v in encoder_state_losses.items()
                        }
                        total_stats.update(encoder_state_losses)
                    if "reward_head" in world_model.heads and train_heads:
                        with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                            encoder_reward_losses = world_model.heads["reward_head"].compute_loss(
                                video_features, reward[:, ::tubelet_size_enc]
                            )
                        rates["info/reward_head/examples_seen"] += video_features.shape[0] * video_features.shape[1]
                        encoder_reward_losses = {k: v.mean() for k, v in encoder_reward_losses.items()}
                        loss = encoder_reward_losses["loss"] / (train_heads_on_predictor * rollout_steps + 1)
                        if train:
                            total_head_loss += loss
                            world_model.heads["reward_head"].backward(loss)
                        encoder_reward_losses = {
                            "encoder_reward_" + k: v.item() for k, v in encoder_reward_losses.items()
                        }
                        total_stats.update(encoder_reward_losses)
                    # 3. TRAIN HEADS ON FEATURES FROM PREDICTOR
                    if train_heads_on_predictor and train_heads:
                        if "image_head" in world_model.heads:
                            target_rgb_video = world_model.heads["image_head"].preprocess_rgb(
                                obs["visual"][:, ::tubelet_size_enc]
                            )
                            with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                                predictor_image_losses = world_model.heads["image_head"].compute_loss(
                                    pred_video_features[:, :-1].detach(),
                                    target_rgb_video[:, 1:],
                                    global_step=epoch * ipe + itr,
                                )
                            rates["info/image_head/examples_seen"] += pred_video_features.shape[0] * (
                                pred_video_features.shape[1] - 1
                            )
                            predictor_image_losses = {k: v.mean() for k, v in predictor_image_losses.items()}
                            if train:
                                world_model.heads["image_head"].backward(predictor_image_losses["loss"])
                            predictor_image_losses = {
                                "predictor_image_" + k: v.item() for k, v in predictor_image_losses.items()
                            }
                            total_stats.update(predictor_image_losses)
                        if "state_head" in world_model.heads:
                            with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                                predictor_state_losses = world_model.heads["state_head"].compute_loss(
                                    video_features, None, state[:, ::tubelet_size_enc]
                                )
                            rates["info/state_head/examples_seen"] += video_features.shape[0] * video_features.shape[1]
                            predictor_state_losses = {k: v.mean() for k, v in predictor_state_losses.items()}
                            loss = predictor_state_losses["loss"] / (train_heads_on_predictor * rollout_steps + 1)
                            if train:
                                world_model.heads["state_head"].backward(predictor_state_losses["loss"])
                            predictor_state_losses = {
                                "predictor_state_" + k: v.item() for k, v in predictor_state_losses.items()
                            }
                            total_stats.update(predictor_state_losses)
                    # 4. TRAIN PREDICTOR ON FURTHER AUTOREGRESSIVE ROLLOUT
                    if rollout_steps > 1 and train and train_predictor:
                        if do_sequential_rollout:
                            with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                                if train_rollout_prefixes == "random":
                                    prefixes = torch.randint(video_features.shape[1] - rollout_steps, size=(1,))
                                elif train_rollout_prefixes == "first":
                                    prefixes = [0]
                                elif train_rollout_prefixes == "all":
                                    prefixes = list(range(video_features.shape[1] - rollout_steps))
                                stats = defaultdict(list)
                                for t in prefixes:
                                    rollout_losses, total_rollout_loss, _, _ = world_model.rollout(
                                        video_features=video_features,
                                        pred_video_features=pred_video_features,
                                        proprio_features=proprio_features,
                                        pred_proprio_features=pred_proprio_features,
                                        action_features=action_features,
                                        action_noise=0.0,
                                        # we have `len(prefixes)` loss terms so weight them equally
                                        loss_weight=1.0 / len(prefixes),
                                        rollout_steps=rollout_steps - 1,
                                        rollout_stop_gradient=rollout_stop_gradient,
                                        ctxt_window=ctxt_window_train_rollout,
                                        mode="sequential",
                                        t=t,
                                    )
                                    total_transition_loss += total_rollout_loss
                                    for k in rollout_losses:
                                        stats[k].append(rollout_losses[k])
                                stats = {k: torch.stack(stats[k]).mean(0) for k in stats}
                                for k in stats:
                                    for j in range(len(stats[k])):
                                        train_rollout_result[f"train_rollout/{k}/{j+2}"] = stats[k][j].item()
                        if do_parallel_rollout:
                            gt_prob = next(rollout_sampling_scheduler)
                            rates["info/transition_model/sampling_gt_prob"] = gt_prob
                            with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                                stats = defaultdict(list)
                                rollout_parallel_losses, total_rollout_parallel_loss, _, _ = world_model.rollout(
                                    video_features=video_features,
                                    proprio_features=proprio_features,
                                    action_features=action_features,
                                    pred_video_features=pred_video_features,
                                    pred_proprio_features=pred_proprio_features,
                                    action_noise=0.0,
                                    loss_weight=1.0,
                                    rollout_steps=rollout_steps - 1,
                                    rollout_stop_gradient=rollout_stop_gradient,
                                    ctxt_window=ctxt_window_train_rollout,
                                    mode="parallel",
                                    gt_prob=gt_prob,
                                    prepend_gt=prepend_gt_rollout_parallel,
                                )
                                total_transition_loss += total_rollout_parallel_loss
                                for k in rollout_parallel_losses:
                                    stats[k].append(rollout_parallel_losses[k])
                                stats = {k: torch.stack(stats[k]).mean(0) for k in stats}
                                for k in stats:
                                    for j in range(len(stats[k])):
                                        parallel_rollout_result[f"train_rollout_parallel/{k}/{j+2}"] = stats[k][
                                            j
                                        ].item()
                    total_stats.update(train_rollout_result)
                    total_stats.update(parallel_rollout_result)

                    # Construct a clean losses dict that aggregates all relevant training losses
                    losses = {}
                    losses["predictor_loss"] = (
                        total_transition_loss.item()
                        if isinstance(total_transition_loss, torch.Tensor)
                        else total_transition_loss
                    )
                    losses["head_loss"] = (
                        total_head_loss.item() if isinstance(total_head_loss, torch.Tensor) else total_head_loss
                    )
                    losses["loss"] = losses["predictor_loss"] + losses["head_loss"]

                    grad_stats, optim_stats = {}, {}
                    # 5. OPTIMIZATION STEP
                    # so far, we only computed losses and ran .backward() so accumulate gradients on several objectives. Below is the
                    # only place where we perform an actual optimization step
                    if train:
                        if train_heads:
                            for name in world_model.heads.keys():
                                # If not train_heads, world_model.heads[name].model.module.decoder_embed.weight.grad should be None
                                grad_stats[name], optim_stats[name] = world_model.heads[name].optimization_step()
                        if train_predictor:
                            world_model.backward(total_transition_loss)
                            grad_stats["transition_model"], optim_stats["transition_model"] = (
                                world_model.optimization_step()
                            )
                        for key in list(grad_stats.keys()):
                            grad_stats[f"optim/{key}/grad_norm"] = (
                                grad_stats[key].global_norm if grad_stats[key] is not None else 0.0
                            )
                            del grad_stats[key]
                        for key in list(optim_stats.keys()):
                            optim_stats[f"optim/{key}/first_moment"] = (
                                optim_stats[key].get("exp_avg").avg if optim_stats[key] is not None else 0.0
                            )
                            optim_stats[f"optim/{key}/second_moment"] = (
                                optim_stats[key].get("exp_avg_sq").avg if optim_stats[key] is not None else 0.0
                            )
                            del optim_stats[key]
                    total_stats.update(grad_stats)
                    total_stats.update(optim_stats)
                    total_stats.update(dict(rates))
                    # 6. ONCE IN A WHILE DO LONG-ROLLOUT EVALUATION WITH DATASET ACTIONS OR RANDOM ACTIONS
                    eval_rollout_result = {}
                    image_stats = {}
                    if not train:
                        world_model.eval()

                        @torch.no_grad
                        def val_rollout(
                            video_features,
                            action_features,
                            proprio_features,
                            pred_video_features,
                            pred_proprio_features,
                            gt_obs,
                            gt_state,
                            prefix_rollout_result=None,
                            val_rollout_steps=5,
                            ctxt_window=None,
                        ):
                            """
                            gt_obs:
                                visual: (B, T, C, H, W)
                            """
                            rollout_steps = min(val_rollout_steps, video_features.shape[1] - 1)
                            with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                                prefixes = range(video_features.shape[1] - rollout_steps)
                                val_rollout_result = {}

                                # Helper function to run rollout with different action noise levels and decode heads
                                def run_rollout_and_decode(action_noise, rollout_prefix):
                                    """Run rollout with given action noise and decode with image/state heads."""
                                    stats = defaultdict(list)
                                    for t in prefixes:
                                        rollout_losses, _, last_vid_feats, last_prop_feats = world_model.rollout(
                                            video_features=video_features,
                                            pred_video_features=pred_video_features,
                                            proprio_features=proprio_features,
                                            pred_proprio_features=pred_proprio_features,
                                            action_features=action_features,
                                            action_noise=action_noise,
                                            rollout_steps=rollout_steps,
                                            rollout_stop_gradient=True,
                                            debug=action_noise == 0.0,
                                            ctxt_window=ctxt_window,
                                            mode="sequential",
                                            t=t,
                                        )
                                        # last_vid_feats: [B T V H W D]
                                        for k in rollout_losses:
                                            stats[k].append(rollout_losses[k])

                                    image_samples = None
                                    if "image_head" in world_model.heads:
                                        image_samples = world_model.heads["image_head"].decode(
                                            last_vid_feats[:, -rollout_steps - 1 :]
                                        )
                                        for h in range(1, image_samples.shape[1]):
                                            key = f"{rollout_prefix}/lpips/{h}"
                                            if prefix_rollout_result is not None:
                                                key = f"{prefix_rollout_result}/{key}"
                                            v = lpips(
                                                image_samples.squeeze(2)[:, h]
                                                .permute(0, 3, 1, 2)
                                                .to(world_model.device, dtype=torch.float32)
                                                / 255.0,
                                                gt_obs["visual"][:, h * tubelet_size_enc].to(
                                                    world_model.device, dtype=torch.float32
                                                ),
                                            ).mean()
                                            val_rollout_result[key] = v.detach().cpu().item()

                                    if "state_head" in world_model.heads:
                                        target_states = gt_state[:, ::tubelet_size_enc][:, -rollout_steps - 1 :]
                                        state_loss = world_model.heads["state_head"].compute_loss(
                                            last_vid_feats[:, -rollout_steps - 1 :],
                                            None,
                                            target_states,
                                            reduce_mean=False,
                                        )
                                        for k in state_loss:
                                            for j in range(state_loss[k].shape[1]):
                                                key = f"{rollout_prefix}/encoder_state_{k}/{j}"
                                                if prefix_rollout_result is not None:
                                                    key = f"{prefix_rollout_result}/{key}"
                                                val_rollout_result[key] = state_loss[k][:, j].mean().item()

                                    # Aggregate rollout losses
                                    stats = {k: torch.stack(stats[k]).mean(0) for k in stats}
                                    for k in stats:
                                        for j in range(len(stats[k])):
                                            key = f"{rollout_prefix}/{k}/{j+1}"
                                            if prefix_rollout_result is not None:
                                                key = f"{prefix_rollout_result}/{key}"
                                            val_rollout_result[key] = stats[k][j].item()

                                    return image_samples

                                eval_image_samples = run_rollout_and_decode(
                                    action_noise=0.0, rollout_prefix="val_rollout"
                                )

                                # Optionally decode position from ground truth visual features
                                if "state_head" in world_model.heads and data_traj_decode_gt:
                                    target_states = gt_state[:, ::tubelet_size_enc][:, -rollout_steps - 1 :]
                                    decode_gt_state_loss = world_model.heads["state_head"].compute_loss(
                                        video_features[:, -rollout_steps - 1 :], None, target_states, reduce_mean=False
                                    )
                                    for k in decode_gt_state_loss:
                                        for j in range(decode_gt_state_loss[k].shape[1]):
                                            key = f"val_rollout/decode_gt_state_{k}/{j}"
                                            if prefix_rollout_result is not None:
                                                key = f"{prefix_rollout_result}/{key}"
                                            val_rollout_result[key] = decode_gt_state_loss[k][:, j].mean().item()

                                noisy_eval_image_samples = run_rollout_and_decode(
                                    action_noise=0.05, rollout_prefix="noisy_val_rollout"
                                )

                                if "image_head" in world_model.heads:
                                    t = eval_image_samples.shape[1]
                                    b = gt_obs["visual"].shape[0]
                                    # b = gt_obs["visual"][:, ::tubelet_size_enc].shape[0]
                                    b = min(4, b)
                                    rgb_v = inverse_transform(gt_obs["visual"][:, ::tubelet_size_enc].cpu())
                                    rgb_v = (255.0 * rgb_v).clip(0.0, 255.0).to(torch.uint8)
                                    rgb_v = rearrange(rgb_v, "b t (v c) h w -> b t v h w c", c=3)
                                    rgb = torch.stack([eval_image_samples, rgb_v[:, -t:]], dim=2)[:b]
                                    rgb = (
                                        rearrange(
                                            rgb,
                                            "b t e v h w c -> (b v e h) (t w) c",
                                        )
                                        .cpu()
                                        .numpy()
                                    )
                                    rgb_noised = torch.stack([noisy_eval_image_samples, rgb_v[:, -t:]], dim=2)[:b]
                                    rgb_noised = (
                                        rearrange(
                                            rgb_noised,
                                            "b t e v h w c -> (b v e h) (t w) c",
                                        )
                                        .cpu()
                                        .numpy()
                                    )
                                    animation = torch.stack(
                                        [rgb_v[:, -t:], eval_image_samples, noisy_eval_image_samples], dim=2
                                    )[:b]
                                    animation = (
                                        rearrange(
                                            animation,
                                            "b t e v h w c -> t c (b v h) (e w)",
                                        )
                                        .cpu()
                                        .numpy()
                                    )
                                else:
                                    rgb, rgb_noised, animation = None, None, None
                            return rgb, rgb_noised, animation, val_rollout_result

                        if do_data_traj_rollout_eval:
                            rgb, rgb_noised, animation, eval_rollout_result = val_rollout(
                                video_features,
                                action_features,
                                proprio_features,
                                pred_video_features,
                                pred_proprio_features,
                                gt_obs=obs,
                                prefix_rollout_result="data_traj",
                                val_rollout_steps=cfgs_data_traj_rollout_eval.get("data_traj_eval_rollout_steps", 3),
                                ctxt_window=cfgs_data_traj_rollout_eval.get("data_traj_eval_ctxt_window", None),
                                gt_state=state,
                            )
                        if do_energy_landscape_eval:
                            gt_visual = inverse_transform(obs["visual"].cpu())
                            gt_visual = (255.0 * gt_visual).clip(0.0, 255.0).to(torch.uint8)
                            gt_proprio = preprocessor.denormalize_proprios(obs["proprio"].cpu())
                            with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                                energy_landscape = world_model.compute_energy_landscape(
                                    video_features,
                                    action_features,
                                    proprio_features,
                                    gt_visual,
                                    gt_proprio,
                                    rollout_steps=cfgs_energy_landscape_eval.get("energy_landscape_rollout_steps", 1),
                                    ctxt_window=cfgs_energy_landscape_eval.get("energy_landscape_ctxt_window", None),
                                    proprio_dim=traj_dataset.proprio_dim,
                                    action_dim=traj_dataset.action_dim,
                                    actions_per_vid_feat=actions_per_vid_feat,
                                    dataset_path=dataset_paths[0],
                                    preprocessor=preprocessor,
                                )
                            image_stats.update(
                                {
                                    "data_traj/energy_landscape": energy_landscape,
                                }
                            )
                        if "image_head" in world_model.heads:
                            if do_data_traj_rollout_eval:
                                image_stats.update(
                                    {
                                        "data_traj/image_rollouts": wandb.Image(rgb),
                                        "data_traj/image_rollouts_noisy_actions": wandb.Image(rgb_noised),
                                        "data_traj/image_animated_rollout": wandb.Video(animation, fps=6),
                                    }
                                )
                        world_model.train()
                    total_stats.update(eval_rollout_result)
                    return (
                        float(losses["loss"]),
                        losses,
                        optim_stats,
                        total_stats,
                        image_stats,
                    )

                # In train mode, image_stats is empty
                if not light_eval_only_mode:
                    obs, action, state, reward, masks_enc, masks_pred = get_batch()
                    (loss, losses, optim_stats, total_stats, image_stats), gpu_etime_ms = gpu_timer(
                        lambda: step_model(obs, action, state, reward, train=True)
                    )
                    iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
                    loss_meter.update(loss)
                    gpu_time_meter.update(gpu_etime_ms)
                    wall_time_meter.update(iter_elapsed_time_ms)
                    if train_csv_logger is None:  # Initialize the logger once
                        train_csv_logger = create_csv_logger(losses, total_stats, train=True)
                else:
                    losses = {}
                    total_stats = {}

                if itr % light_eval_freq == light_eval_freq - 1 and val_loader_iters is not None:
                    # image_stats overrides the empty image_stats from the train step at same itr
                    image_stats, eval_losses, eval_total_stats = {}, {}, {}
                    # Then, use non-distributed validation data for visualization on rank 0
                    if val_viz_rank0_loader and rank == 0:
                        viz_obs, viz_action, viz_state, viz_reward, _, _ = get_viz_batch()
                        if viz_obs is not None:
                            with torch.no_grad():
                                (_, _, _, _, viz_img_stats), _ = gpu_timer(
                                    lambda: step_model(viz_obs, viz_action, viz_state, viz_reward, train=False)
                                )
                                if light_eval_only_mode:
                                    image_stats.update(
                                        {f"light_eval_only/epoch-{epoch+1}/{k}": v for k, v in viz_img_stats.items()}
                                    )
                                else:
                                    image_stats.update(viz_img_stats)
                    # First, use distributed validation data for metrics
                    for idx in range(len(val_loader_iters)):
                        obs, action, state, reward, masks_enc, masks_pred = get_batch(train=False, idx=idx)
                        with torch.no_grad():
                            (_, val_losses, _, val_total_stats, val_img_stats), gpu_etime_ms = gpu_timer(
                                lambda: step_model(obs, action, state, reward, train=False)
                            )
                            if not light_eval_only_mode:
                                eval_losses.update({f"eval_data/load-{idx}/{k}": v for k, v in val_losses.items()})
                                eval_total_stats.update(
                                    {f"eval_data/load-{idx}/{k}": v for k, v in val_total_stats.items()}
                                )
                            # Only use distributed batch images if we don't have visualization images
                            if not val_viz_rank0_loader:
                                prefixed_stats = {f"load-{idx}/{k}": v for k, v in val_img_stats.items()}
                                if light_eval_only_mode:
                                    image_stats.update(
                                        {f"light_eval_only/epoch-{epoch+1}/{k}": v for k, v in prefixed_stats.items()}
                                    )
                                else:
                                    image_stats.update(prefixed_stats)
                    if eval_csv_logger is None:  # Initialize the logger once
                        eval_csv_logger = create_csv_logger(eval_losses, eval_total_stats, train=False)
                else:
                    eval_losses = {}
                    eval_total_stats = {}

                # -- Logging
                def log_stats():
                    trainer.log(epoch, itr, losses, total_stats, eval_losses, eval_total_stats, image_stats)
                    if not light_eval_only_mode:
                        log_values = [epoch + 1, itr, loss, gpu_etime_ms, iter_elapsed_time_ms]
                        for key in train_csv_logger_columns[5:]:
                            if key in losses:
                                value = losses[key]
                                log_values.append(value.item() if isinstance(value, torch.Tensor) else value)
                            elif key in total_stats:
                                value = total_stats[key]
                                log_values.append(value.item() if isinstance(value, torch.Tensor) else value)
                            else:
                                log_values.append(0.0)
                        train_csv_logger.log(*log_values)
                        if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                            logger.info(
                                "[%d, %5d] "
                                "[mem: %.2e] "
                                "[gpu: %.1f ms]"
                                "[wall: %.1f ms]"
                                % (
                                    epoch + 1,
                                    itr,
                                    torch.cuda.max_memory_allocated() / 1024.0**2,
                                    gpu_time_meter.avg,
                                    wall_time_meter.avg,
                                )
                            )
                    if itr % light_eval_freq == light_eval_freq - 1:
                        log_values = [epoch + 1, itr]
                        for key in eval_csv_logger_columns[2:]:
                            if key in eval_losses:
                                value = eval_losses[key]
                                log_values.append(value.item() if isinstance(value, torch.Tensor) else value)
                            elif key in eval_total_stats:
                                log_values.append(eval_total_stats[key])
                            else:
                                log_values.append(0.0)
                        eval_csv_logger.log(*log_values)

                log_stats()
                if not light_eval_only_mode:
                    assert not np.isnan(loss), "loss is nan"
            logger.info("avg. loss %.3f" % loss_meter.avg)

            # -- Save Last
            if not light_eval_only_mode:
                if epoch % checkpoint_freq == 0 or epoch == (num_epochs - 1):
                    if rank == 0:
                        save_checkpoint(epoch + 1, latest_path)
                        if save_every_freq > 0 and epoch % save_every_freq == 0:
                            save_every_file = pref_tag + f"e{epoch}.{latest_format}"
                            save_every_path = os.path.join(folder, save_every_file)
                            save_checkpoint(epoch + 1, save_every_path)

            # -- Launch Planning Eval
            if not light_eval_only_mode:
                if (epoch % eval_freq == 0) or epoch == (num_epochs - 1):
                    if save_every_freq > 0:
                        checkpoint = pref_tag + f"e{epoch}.{latest_format}"
                    else:
                        checkpoint = pref_tag + f"latest.{latest_format}"
                    launch_planning_evals(
                        rank,
                        epoch + 1,
                        folder,
                        checkpoint,
                        cfgs_plan_evals,
                        cfgs_model,
                        cfgs_data,
                        cfgs_data_aug,
                        "",
                        world_model=world_model,
                        dset=val_traj_dataset,
                        preprocessor=preprocessor,
                    )
                    world_model.train()
    elif unroll_decode_eval_only_mode:
        logger.info("Launching unroll-decode evals only mode")
        checkpoint = pref_tag + f"latest.{latest_format}"
        launch_unroll_decode_eval(
            rank,
            start_epoch,
            folder,
            checkpoint,
            cfgs_unroll_decode_evals,
            cfgs_model,
            cfgs_data,
            cfgs_data_aug,
            world_model=world_model,
            dset=val_traj_dataset,
            preprocessor=preprocessor,
        )
    else:
        logger.info("Skipping training loop due to plan_only_eval_mode being enabled.")
        checkpoint = pref_tag + f"latest.{latest_format}"
        launch_planning_evals(
            rank,
            start_epoch,
            folder,
            checkpoint,
            cfgs_plan_evals,
            cfgs_model,
            cfgs_data,
            cfgs_data_aug,
            "-plan-only",
            world_model=world_model,
            dset=val_traj_dataset,
            preprocessor=preprocessor,
        )


def launch_planning_evals(
    rank,
    epoch,
    folder,
    checkpoint,
    cfgs_plan_evals,
    cfgs_model,
    cfgs_data,
    cfgs_data_aug,
    tag_suffix,
    world_model=None,
    dset=None,
    preprocessor=None,
):
    eval_cfg_paths = cfgs_plan_evals.get("eval_cfg_paths", None)
    eval_nodes = cfgs_plan_evals.get("nodes", None)
    eval_episodes = cfgs_plan_evals.get("eval_episodes", None)
    eval_low_pri = cfgs_plan_evals.get("low_pri", True)
    separate = cfgs_plan_evals.get("separate", True)
    override_cfgs_data = cfgs_plan_evals.get("override_cfgs_data", True)
    override_datasets = cfgs_plan_evals.get("override_datasets", True)
    # task_specification
    evals_obs = cfgs_plan_evals.get("obs", None)
    # planner
    evals_alpha = cfgs_plan_evals.get("alpha", None)
    max_episode_steps = cfgs_plan_evals.get("max_episode_steps", None)
    num_act_stepped = cfgs_plan_evals.get("num_act_stepped", None)
    horizon = cfgs_plan_evals.get("horizon", None)
    evals_decode = cfgs_plan_evals.get("decode", None)
    sum_all_diffs = cfgs_plan_evals.get("sum_all_diffs", None)
    if eval_cfg_paths is not None:
        eval_nodes, eval_tasks_per_node, args_eval, eval_cpus_per_task = build_plan_eval_args(
            app_name="vjepa_wm",
            folder=folder,
            checkpoint=checkpoint,
            eval_cfg_paths=eval_cfg_paths,
            cfgs_model=cfgs_model,
            cfgs_data=cfgs_data,
            cfgs_data_aug=cfgs_data_aug,
            override_cfgs_data=override_cfgs_data,
            override_datasets=override_datasets,
            tag=f"epoch-{epoch}{tag_suffix}",
            evals_decode=evals_decode,
            sum_all_diffs=sum_all_diffs,
            evals_obs=evals_obs,
            evals_alpha=evals_alpha,
            eval_nodes=eval_nodes,
            eval_episodes=eval_episodes,
            max_episode_steps=max_episode_steps,
            num_act_stepped=num_act_stepped,
            horizon=horizon,
            wrapper_kwargs=cfgs_plan_evals.get("wrapper_kwargs", {}),
        )

        # # # #####################
        # # Uncomment for debug: save online eval cfgs
        if rank == 0:
            dir = "configs/basile/cw/dump_online_evals/vjepa_wm"
            yaml_paths = [
                os.path.join(dir, "dset_ng_L2.yaml"),
                os.path.join(dir, "dset_cem_L2.yaml"),
                os.path.join(dir, "dset_ng_L1.yaml"),
                os.path.join(dir, "dset_cem_L1.yaml"),
                # # # # rand state
                os.path.join(dir, "rand_ng_L2.yaml"),
                os.path.join(dir, "rand_cem_L2.yaml"),
                os.path.join(dir, "rand_ng_L1.yaml"),
                os.path.join(dir, "rand_cem_L1.yaml"),
                # itr
                os.path.join(dir, "itr_ng_L2.yaml"),
                os.path.join(dir, "itr_cem_L2.yaml"),
                os.path.join(dir, "itr_ng_L1.yaml"),
                os.path.join(dir, "itr_cem_L1.yaml"),
            ]
            os.makedirs(dir, exist_ok=True)
            for i, args in enumerate(args_eval):
                dump_yaml(args_eval[i], yaml_paths[i])
            breakpoint()
        # # #     # #####################

        for i, cfg in enumerate(args_eval):
            args_eval[i] = convert_to_dict_recursive(args_eval[i])

        if separate:
            if rank == 0:
                account, partition, qos = slurm_account_partition_and_qos(low_pri=eval_low_pri)
                logger.info(f"Launching online evals with {account=}, {partition=}, {qos=}")
                with submitit.helpers.clean_env():
                    launch_evals(
                        args_for_evals=args_eval,
                        nodes=eval_nodes,
                        tasks_per_node=eval_tasks_per_node,
                        submitit_folder=os.path.join(folder, "submitit-evals"),
                        account=account,
                        partition=partition,
                        qos=qos,
                        cpus_per_task=eval_cpus_per_task,
                        delay_seconds=5,
                    )
                logger.info(f"Launched online evals from templates {eval_cfg_paths}")
        else:
            from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import EncPredWM
            from evals.simu_env_planning.eval import main_distributed_episodes_eval as gc_main_dist

            world_model.eval()
            for i, cfg in tqdm(enumerate(args_eval)):
                eval_tag = cfg.get("tag", None)
                pretrain_folder = cfg.get("folder", None)
                folder = os.path.join(pretrain_folder, "simu_env_planning/")
                if eval_tag is not None:
                    folder = os.path.join(folder, eval_tag)
                cfg["frameskip"] = cfg["model_kwargs"]["data"]["custom"]["frameskip"]
                cfg["work_dir"] = folder
                model = EncPredWM(
                    world_model,
                    action_dim=world_model.action_dim,
                    preprocessor=preprocessor,
                    ctxt_window=cfg["model_kwargs"]["wrapper_kwargs"]["ctxt_window"],
                )
                with torch.no_grad():
                    gc_main_dist(cfg, model=model, dset=dset, preprocessor=preprocessor, rank=rank)


def launch_unroll_decode_eval(
    rank,
    epoch,
    folder,
    checkpoint,
    cfgs_unroll_decode_evals,
    cfgs_model,
    cfgs_data,
    cfgs_data_aug,
    world_model=None,
    dset=None,
    preprocessor=None,
):
    """
    Launch unroll decode evaluations.
    """
    do_unroll_decode_eval = cfgs_unroll_decode_evals.get("do_unroll_decode_eval", True)
    specific_video = cfgs_unroll_decode_evals.get("specific_video", None)
    specific_video_path = cfgs_unroll_decode_evals.get("specific_video_path", None)
    play_in_reverse = cfgs_unroll_decode_evals.get("play_in_reverse", False)
    obs = cfgs_unroll_decode_evals.get("obs", "rgb")
    save_decoding_only = cfgs_unroll_decode_evals.get("save_decoding_only", False)
    repeat_hardcode_act = cfgs_unroll_decode_evals.get("repeat_hardcode_act", 5)

    if do_unroll_decode_eval:
        eval_nodes, eval_tasks_per_node, args_eval = build_unroll_decode_eval_args(
            app_name="vjepa_wm",
            folder=folder,
            checkpoint=checkpoint,
            cfgs_model=cfgs_model,
            cfgs_data=cfgs_data,
            cfgs_data_aug=cfgs_data_aug,
            tag=f"epoch-{epoch}",
            specific_video=specific_video,
            specific_video_path=specific_video_path,
            play_in_reverse=play_in_reverse,
            obs=obs,
            save_decoding_only=save_decoding_only,
            repeat_hardcode_act=repeat_hardcode_act,
            wrapper_kwargs=cfgs_unroll_decode_evals.get("wrapper_kwargs", {}),
        )

        # # # #####################
        # # # # Uncomment for debug: save online eval cfgs
        dump_dir = "configs/basile/cw/dump_online_evals/vjepa_wm/unroll_decode"
        os.makedirs(dump_dir, exist_ok=True)
        yaml_paths = []
        for i, cfg in enumerate(args_eval):
            yaml_path = os.path.join(dump_dir, f"{i}.yaml")
            yaml_paths.append(yaml_path)
            dump_yaml(cfg, yaml_path)
        logger.info(f"Dumped {len(yaml_paths)} unroll_decode eval configs to {dump_dir}")
        # # # #####################
