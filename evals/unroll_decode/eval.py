# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Unroll Decode Evaluation (DROID data only)

This evaluation generates counterfactual decodings by hardcoding custom actions
(e.g., open/close gripper + move up) to produce and compare different prediction scenarios.

The hardcoded actions are defined in the `create_counterfactual_actions` function and can be
modified to test different action scenarios.

NOTE: This evaluation is designed to work only with DROID data.
"""

import importlib
import logging
import os

import lpips as lpips_lib
import numpy as np
import torch
import torch.multiprocessing as mp
import yaml
from einops import rearrange
from tensordict.tensordict import TensorDict

from evals.utils import log_media_local, make_datasets, prepare_obs
from src.utils.logging import CSVLogger

# Set CUDA device for distributed training
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

from src.utils.distributed import init_distributed

logging.basicConfig()
logger = logging.getLogger()

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


def create_counterfactual_actions(actions, repeat_hardcode_act):
    """
    Create hardcoded counterfactual actions for evaluation.

    This function overrides the input actions with custom hardcoded values
    to generate counterfactual predictions. Modify this function to test
    different action scenarios.

    Args:
        actions: Original actions tensor of shape (T, B, A)
        repeat_hardcode_act: Number of times to repeat the action sequence

    Returns:
        Modified actions tensor with hardcoded values
    """
    # Override all actions to zero
    actions[:, :, :] = 0.0

    # Gripper action (last dimension):
    # - Positive value (e.g., 0.75) = close gripper
    # - Negative value (e.g., -0.75) = open gripper
    actions[:, :, -1] = -0.75  # Open gripper

    # Vertical motion (z-axis, typically index 2):
    actions[:, :, 2] = 0.05  # Move up

    # Repeat the action sequence
    actions = actions.repeat(repeat_hardcode_act, 1, 1)

    return actions


def init_module(
    folder,
    checkpoint,
    module_name,
    model_kwargs,
    device,
    cfgs_data=None,
    wrapper_kwargs=None,
    action_dim=None,
    proprio_dim=None,
    preprocessor=None,
):
    """Build (frozen) model and initialize from pretrained checkpoint."""
    model = importlib.import_module(f"{module_name}").init_module(
        folder=folder,
        checkpoint=checkpoint,
        model_kwargs=model_kwargs,
        device=device,
        action_dim=action_dim,
        proprio_dim=proprio_dim,
        preprocessor=preprocessor,
        cfgs_data=cfgs_data,
        wrapper_kwargs=wrapper_kwargs,
    )
    return model


def main(args_eval, resume_preempt=False):
    """
    Main evaluation function for unroll decode evaluation.

    Args:
        args_eval: Evaluation configuration dictionary containing:
            - folder: Output folder for results
            - tag: Evaluation tag for organizing outputs
            - specific_video: Whether to use a specific video file
            - specific_video_path: Path to specific video file (npz format)
            - play_in_reverse: Whether to reverse the video sequence
            - obs: Observation type ("rgb" or "rgb_state")
            - save_decoding_only: Whether to only save decoded predictions
            - repeat_hardcode_act: Number of times to repeat hardcoded actions
            - model_kwargs: Model configuration including checkpoint path
    """
    # Parse config
    eval_tag = args_eval.get("tag", None)
    pretrain_folder = args_eval.get("folder", None)

    model_kwargs = args_eval.get("model_kwargs")
    module_name = model_kwargs.get("module_name")
    cfgs_data = model_kwargs.get("data", {})
    cfgs_data_aug = model_kwargs.get("data_aug", {})
    wrapper_kwargs = model_kwargs.get("wrapper_kwargs", {})
    pretrain_kwargs = model_kwargs.get("pretrain_kwargs", {})

    specific_video = args_eval.get("specific_video", False)
    specific_video_path = args_eval.get("specific_video_path", "evals/unroll_decode/demo_data.npz")
    play_in_reverse = args_eval.get("play_in_reverse", False)
    expected_obs = args_eval.get("obs", "rgb")
    save_decoding_only = args_eval.get("save_decoding_only", False)
    repeat_hardcode_act = args_eval.get("repeat_hardcode_act", 5)

    # Setup output folder
    folder = os.path.join(pretrain_folder, "unroll_decode/")
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    os.makedirs(folder, exist_ok=True)

    # Save config
    yaml_file_path = os.path.join(folder, "args_eval.yaml")
    with open(yaml_file_path, "w") as yaml_file:
        yaml.dump(args_eval, yaml_file, default_flow_style=False)
    logger.info(f"Saved args_eval to {yaml_file_path}")

    # Initialize distributed
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
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # Initialize dataset and preprocessor
    dset, preprocessor = make_datasets(cfgs_data, cfgs_data_aug, world_size, rank, filter_first_episodes=2)

    # Initialize model
    if importlib.util.find_spec(module_name) is None:
        module_name = "app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds"
    logger.info(f"Module found: {module_name}")

    checkpoint = model_kwargs.get("checkpoint")
    wm = init_module(
        folder=pretrain_folder,
        checkpoint=checkpoint,
        module_name=module_name,
        model_kwargs=pretrain_kwargs,
        wrapper_kwargs=wrapper_kwargs,
        cfgs_data=cfgs_data,
        device=device,
        action_dim=dset.action_dim,
        proprio_dim=dset.proprio_dim,
        preprocessor=preprocessor,
    )
    wm.eval()
    logger.info("Loaded world model")

    # Initialize LPIPS once (like in train.py)
    lpips = lpips_lib.LPIPS(net="vgg").eval().to(device)

    # Initialize dataloader
    batch_size = cfgs_data.get("loader", {}).get("batch_size", 4)
    loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    logger.info("Initialized dataloader")

    # CSV logger setup
    eval_csv_logger = None
    eval_csv_logger_columns = None

    def create_csv_logger(total_stats):
        nonlocal eval_csv_logger_columns
        csv_log_file = os.path.join(folder, f"log_eval_r{rank}.csv")
        excluded_keys = ["image_rollouts", "image_rollouts_noisy_actions", "image_animated_rollout"]
        all_keys = {key for key in total_stats.keys() if key not in excluded_keys}
        sorted_keys = sorted(all_keys)
        new_columns = [("%.5f", key) for key in sorted_keys]
        eval_csv_logger_columns = ["itr"] + sorted_keys
        return CSVLogger(csv_log_file, ("%d", "itr"), *new_columns)

    # Determine whether to compute proprio loss based on wrapper_kwargs.proprio_mode
    proprio_mode = wrapper_kwargs.get("proprio_mode", "predict_proprio")
    use_proprio_loss = proprio_mode == "predict_proprio"

    @torch.no_grad()
    def val_rollout(obs, actions):
        """
        Perform validation rollout with hardcoded counterfactual actions.

        Args:
            obs: Dictionary with 'visual' (B, T, C, H, W) and 'proprio' (B, T, P) tensors
            actions: Action tensor of shape (B, T, A)

        Returns:
            rgb: Decoded rollout image grid
            rgb_noised: Decoded rollout with noisy actions
            animation: Animated rollout video frames
            val_rollout_result: Dictionary of evaluation metrics
        """
        B, tau, A = actions.shape
        actions = actions.reshape(B, -1, wm.action_dim)
        actions = rearrange(actions, "b t a -> t b a")

        # Apply counterfactual actions
        actions = create_counterfactual_actions(actions, repeat_hardcode_act)
        T, B, A = actions.shape

        # Prepare observations
        clips = obs["visual"]
        proprio = obs["proprio"]
        clips = preprocessor.inverse_transform(clips.cpu())
        clips = 255.0 * clips
        clips = clips.clip(0.0, 255.0)

        # Pad clips if needed
        if clips.shape[1] < T - 1:
            last_frame = clips[:, -1:]
            num_repeats = T - clips.shape[1]
            clips = torch.cat([clips, last_frame.repeat(1, num_repeats, 1, 1, 1)], dim=1)

        # Encode context and ground truth
        td_ctxt = TensorDict({"visual": clips[:, :1], "proprio": proprio[:, :1]})
        z_ctxt = wm.encode(prepare_obs(expected_obs, td_ctxt))  # [B 1 V H W D], [B 1 P]
        td_gt = TensorDict({"visual": clips, "proprio": proprio})
        z_gt = wm.encode(prepare_obs(expected_obs, td_gt))  # [B T V H W D], [B T P]
        z_gt_visual = z_gt["visual"] if expected_obs == "rgb_state" else z_gt  # B T V H W D
        z_gt_proprio = z_gt["proprio"] if expected_obs == "rgb_state" and use_proprio_loss else None

        # Discard result of last action since we have no target for it
        if T >= clips.shape[1]:
            actions = actions[:-1]

        noise_actions = torch.randn_like(actions) * 0.1

        # Unroll predictions
        predicted_encs = wm.unroll(z_ctxt, act_suffix=actions)
        predicted_encs_visual = predicted_encs["visual"] if expected_obs == "rgb_state" else predicted_encs
        predicted_encs_proprio = (
            predicted_encs["proprio"] if expected_obs == "rgb_state" and use_proprio_loss else None
        )

        predicted_noise_encs = wm.unroll(z_ctxt, act_suffix=noise_actions)
        predicted_noise_encs_visual = (
            predicted_noise_encs["visual"] if expected_obs == "rgb_state" else predicted_noise_encs
        )
        predicted_noise_encs_proprio = (
            predicted_noise_encs["proprio"] if expected_obs == "rgb_state" and use_proprio_loss else None
        )

        # Compute losses
        val_rollout_result = {}
        for h in range(1, len(predicted_encs_visual)):
            pred_proprio_h = (
                predicted_encs_proprio.transpose(1, 0)[:, h : h + 1] if predicted_encs_proprio is not None else None
            )
            gt_proprio_h = z_gt_proprio[:, h : h + 1] if z_gt_proprio is not None else None
            losses = wm.model.compute_loss(
                predicted_encs_visual.transpose(1, 0)[:, h : h + 1],
                pred_proprio_h,
                z_gt_visual[:, h : h + 1],
                gt_proprio_h,
                shift=0,
                reduce_mean=True,
            )
            noisy_pred_proprio_h = (
                predicted_noise_encs_proprio.transpose(1, 0)[:, h : h + 1]
                if predicted_noise_encs_proprio is not None
                else None
            )
            noisy_losses = wm.model.compute_loss(
                predicted_noise_encs_visual.transpose(1, 0)[:, h : h + 1],
                noisy_pred_proprio_h,
                z_gt_visual[:, h : h + 1],
                gt_proprio_h,
                shift=0,
                reduce_mean=True,
            )
            for k, v in losses.items():
                val_rollout_result[f"val_rollout/{k}/{h}"] = v.detach().cpu().item()
                val_rollout_result[f"noisy_val_rollout/{k}/{h}"] = noisy_losses[k].detach().cpu().item()

        # Decode images if image head is available
        rgb, rgb_noised, animation = None, None, None
        if "image_head" in wm.heads:
            eval_image_samples = torch.from_numpy(wm.decode_unroll(predicted_encs, batch=True)).unsqueeze(2)
            noisy_eval_image_samples = torch.from_numpy(wm.decode_unroll(predicted_noise_encs, batch=True)).unsqueeze(
                2
            )

            # Compute LPIPS scores (using pre-initialized lpips)
            for h in range(1, len(predicted_encs_visual)):
                v = lpips(
                    eval_image_samples.squeeze(2)[:, h].permute(0, 3, 1, 2).to(wm.device, dtype=torch.float32) / 255.0,
                    clips[:, h].to(wm.device, dtype=torch.float32) / 255.0,
                ).mean()
                noisy_v = lpips(
                    noisy_eval_image_samples.squeeze(2)[:, h].permute(0, 3, 1, 2).to(wm.device, dtype=torch.float32)
                    / 255.0,
                    clips[:, h].to(wm.device, dtype=torch.float32) / 255.0,
                ).mean()
                val_rollout_result[f"val_rollout/lpips/{h}"] = v.detach().cpu().item()
                val_rollout_result[f"noisy_val_rollout/lpips/{h}"] = noisy_v.detach().cpu().item()

            # Create visualizations
            t = eval_image_samples.shape[1]
            b = min(4, clips.shape[0])
            rgb_v = rearrange(clips, "b t c h w -> b t 1 h w c")

            if save_decoding_only:
                rgb = torch.stack([eval_image_samples], dim=2)[:b]
            else:
                rgb = torch.stack([eval_image_samples, rgb_v[:, -t:]], dim=2)[:b]
            rgb = rearrange(rgb, "b t e v h w c -> (b v e h) (t w) c").cpu().numpy()

            rgb_noised = torch.stack([noisy_eval_image_samples, rgb_v[:, -t:]], dim=2)[:b]
            rgb_noised = rearrange(rgb_noised, "b t e v h w c -> (b v e h) (t w) c").cpu().numpy()

            animation = torch.stack([rgb_v[:, -t:], eval_image_samples, noisy_eval_image_samples], dim=2)[:b]
            animation = rearrange(animation, "b t e v h w c -> t c (b v h) (e w)").cpu().numpy()

        return rgb, rgb_noised, animation, val_rollout_result

    # Run evaluation
    if specific_video:
        from app.plan_common.datasets.droid_dset import poses_to_diffs

        data = np.load(specific_video_path)
        itr = 0

        # Load and prepare video - handle different npz file formats
        if "observations" in data.keys():
            # Standard format: observations (B, T, H, W, C), states (B, T, P)
            visual, proprio = data["observations"].copy(), data["states"].copy()
        elif "image_sequence" in data.keys():
            # Demo format: start_image, image_sequence, goal_image, start_pose, goal_pose
            visual = data["image_sequence"][None]
            # visual = np.concatenate([data["start_image"][None], data["image_sequence"], data["goal_image"][None]], axis=0)[None]
            T = visual.shape[1]
            alphas = np.linspace(0, 1, T)[:, None]
            proprio = (data["start_pose"] * (1 - alphas) + data["goal_pose"] * alphas)[None]
        else:
            raise ValueError(f"Unknown npz file format. Available keys: {list(data.keys())}")

        if play_in_reverse:
            visual, proprio = visual[:, ::-1].copy(), proprio[:, ::-1].copy()

        obs = {
            "visual": preprocessor.transform(torch.tensor(visual).permute(0, 1, 4, 2, 3).float() / 255.0).to(device),
            "proprio": torch.tensor(proprio).float().to(device),
        }
        action = (
            torch.tensor(poses_to_diffs(obs["proprio"].squeeze().cpu())).unsqueeze(0).to(device, dtype=torch.float32)
        )

        rgb, rgb_noised, animation, eval_rollout_result = val_rollout(obs, action)

        # Log results
        image_stats = {}
        if "image_head" in wm.heads:
            image_stats.update(
                {
                    "image_rollouts": rgb,
                    "image_rollouts_noisy_actions": rgb_noised,
                    "image_animated_rollout": animation,
                }
            )
        log_media_local(image_stats, folder=folder, step=itr)

        if eval_csv_logger is None:
            eval_csv_logger = create_csv_logger(eval_rollout_result)
        log_values = [itr] + [eval_rollout_result.get(key, 0.0) for key in eval_csv_logger_columns[1:]]
        eval_csv_logger.log(*log_values)
    else:
        for itr, batch in enumerate(loader):
            obs, action, state, reward = batch
            for k in obs.keys():
                obs[k] = obs[k].to(device, dtype=torch.float32)
            action = action.to(device, dtype=torch.float32)

            rgb, rgb_noised, animation, eval_rollout_result = val_rollout(obs, action)

            # Log results
            image_stats = {}
            if "image_head" in wm.heads:
                image_stats.update(
                    {
                        "image_rollouts": rgb,
                        "image_rollouts_noisy_actions": rgb_noised,
                        "image_animated_rollout": animation,
                    }
                )
            log_media_local(image_stats, folder=folder, step=itr)

            if eval_csv_logger is None:
                eval_csv_logger = create_csv_logger(eval_rollout_result)
            log_values = [itr] + [eval_rollout_result.get(key, 0.0) for key in eval_csv_logger_columns[1:]]
            eval_csv_logger.log(*log_values)
