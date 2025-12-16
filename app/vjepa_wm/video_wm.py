import logging
import os
import sys
from collections import defaultdict
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from tensordict import TensorDict
from tqdm import tqdm

from src.utils.logging import adamw_logger, grad_logger

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


class VideoWM(nn.Module):
    """
    A latent world model that has at least an encoder and a predictor. It should unrollable
    on a sequence of actions and latent context with self.unroll().
    Goal of this class is to keep action encoder, proprio encoder, predictor separate
    """

    def __init__(
        self,
        # Core modules
        encoder,
        predictor,
        action_encoder,
        proprio_encoder,
        # Model architecture parameters
        enc_type="vjepa",
        pred_type="dino_wm",
        grid_size=16,
        tubelet_size_enc=2,
        img_size=256,
        # Input dimensions
        action_dim=None,
        proprio_dim=None,
        # Action & proprio parameters
        action_tokens=1,
        proprio_tokens=1,
        use_action=True,
        use_proprio=True,
        action_skip=2,
        frameskip=1,
        action_conditioning="token",
        proprio_encoding="feature",
        # Encoder behavior flags
        action_encoder_inpred=False,
        proprio_encoder_inpred=False,
        batchify_video=False,
        dup_image=False,
        normalize_reps=False,
        # Rollout behavior
        proprio_rollout_mode="predict_proprio",
        # Optimization parameters
        device=None,
        scaler=None,
        optimizer=None,
        clip_grad=None,
        mixed_precision=None,
        use_radamw=None,
        # Loss configuration
        cfgs_loss=None,
        # Additional modules
        heads=None,
    ):
        super().__init__()
        # Core modules
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.proprio_encoder = proprio_encoder
        # Model architecture parameters
        self.enc_type = enc_type
        self.pred_type = pred_type
        self.grid_size = grid_size
        self.tubelet_size_enc = tubelet_size_enc
        self.img_size = img_size
        # Input dimensions
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        # Action & proprio parameters
        self.action_tokens = action_tokens
        self.proprio_tokens = proprio_tokens
        self.use_action = use_action
        self.use_proprio = use_proprio  # Note: this means pred receives proprio as input
        # In the case of VJ2AC pred, it does not mean it also outputs proprio.
        # Hence, we have a flag proprio_loss in compute_loss
        self.action_skip = action_skip
        self.frameskip = frameskip
        self.action_conditioning = action_conditioning
        self.proprio_encoding = proprio_encoding
        # Encoder behavior flags
        self.action_encoder_inpred = action_encoder_inpred
        self.proprio_encoder_inpred = proprio_encoder_inpred
        self.batchify_video = batchify_video
        self.dup_image = dup_image
        self.normalize_reps = normalize_reps
        # Rollout behavior
        self.proprio_rollout_mode = proprio_rollout_mode
        # Validation
        if self.enc_type == "dino":
            assert self.batchify_video == True, "batchify_video must be True for dino"
        # Device
        self.device = device
        # Optimization parameters
        self.scaler = scaler
        self.optimizer = optimizer
        self.clip_grad = clip_grad
        self.mixed_precision = mixed_precision
        self.use_radamw = use_radamw
        # Loss configuration
        self.cfgs_loss = cfgs_loss
        self.loss_func = "base"
        if cfgs_loss is not None:
            logger.info(
                f"Initialized loss with:\n"
                f'  cos_loss_weight: {cfgs_loss.get("cos_loss_weight", None)}\n'
                f'  l1_loss_weight: {cfgs_loss.get("l1_loss_weight", None)}\n'
                f'  l2_loss_weight: {cfgs_loss.get("l2_loss_weight", None)}\n'
                f'  smooth_l1_loss_weight: {cfgs_loss.get("smooth_l1_loss_weight", None)}'
            )
            if "func" in cfgs_loss.keys():
                self.loss_func = cfgs_loss["func"]
            self.proprio_loss = cfgs_loss.get("proprio_loss", True)
        # Additional modules
        if heads is None:
            heads = []
        self.heads = heads

    def encode_obs(self, obs):
        """
        Encodes an observation into its latent representation.
        This implementation either takes a dict (or Tensordict) with keys
        "visual" and "proprio", or just a visual tensor.
        Input visual obs: (b t c h w)
        Possible combinations:
            - self.enc_type == 'dino':
                - self.dup_image = False, self.batchify_video = True
            - self.enc_type == 'vjepa':
                - self.dup_image = False, self.batchify_video = False -> treat V-JEPA as video encoder
                - self.dup_image = True, self.batchify_video = False -> treat V-JEPA as vid enc but more compatible with frameskip
                - self.dup_image = True, self.batchify_video = True -> treat V-JEPA as frame encoder
        """
        if isinstance(obs, dict):
            visual = obs["visual"]
            proprio = obs.get("proprio")  # Use get() to avoid KeyError if 'proprio' is not present
        else:
            raise ValueError("Input must be a dictionary with keys 'visual' and 'proprio' ")
        b, t, c, h, w = visual.shape
        if self.batchify_video:
            # image encoder flattens the time dimension as batch dimension
            visual = rearrange(visual, "b t ... -> (b t) ...")
        if self.dup_image:
            if not self.batchify_video:
                visual = visual.repeat_interleave(2, dim=1)  # b, (2 t), c, h, w
            else:
                visual = visual.unsqueeze(2).repeat(1, 1, 2, 1, 1)  # b c 2 h w
        if self.enc_type == "dino":  # no duplication needed
            visual_embs = self.encoder(visual)
            visual_embs = rearrange(
                visual_embs, "(b t) (h w) d -> b t 1 h w d", b=b, h=self.grid_size, w=self.grid_size
            )
        elif self.enc_type == "vjepa":
            if not self.batchify_video:
                visual = rearrange(visual, "b t c h w -> b c t h w ")
            visual_embs = self.encoder(visual)
            if self.batchify_video:
                visual_embs = rearrange(
                    visual_embs, "(b t) (h w) d -> b t 1 h w d", b=b, t=t, h=self.grid_size, w=self.grid_size
                )
            else:
                visual_embs = rearrange(visual_embs, "b (t h w) d -> b t 1 h w d", h=self.grid_size, w=self.grid_size)
        else:
            raise ValueError("enc_type must be 'dino' or 'vjepa' ")
        if self.normalize_reps:
            visual_embs = F.layer_norm(visual_embs, (visual_embs.size(-1),))
        if self.use_proprio and proprio is not None:
            proprio_emb = self.encode_proprio(proprio)
        else:
            proprio_emb = None
        return TensorDict({"visual": visual_embs, "proprio": proprio_emb})

    def encode_act(self, a):
        """
        Input a: (b, num_frames, frameskip * action_dim)
            num_frames must ne multiple of tubelet_size_enc
        Output: (b, num_frames // tubelet_size_enc, 1, action_emb_dim * tubelet_size_enc)
        else:
            Input: a: (b, num_frames * frameskip // action_skip, frameskip * action_dim)
                where num_frames * frameskip // action_skip can be = num_frames // tubelet_size_enc,
                so that num_frames * frameskip // action_skip = grid_depth.
        the input time dimension is variable, but the constant is that
            Output: (b, grid_depth, 1, model_action_dim)
            where grid_depth is the same for proprio and video features.
        """
        B, T, D = a.shape
        a = a.reshape(B, -1, self.action_dim)
        if self.action_encoder_inpred:
            return a
        else:
            action = self.action_encoder(a)
        if self.action_conditioning == "feature":
            action = repeat(action, "b t 1 a -> b t f a", f=self.grid_size**2)
        return action

    def encode_proprio(self, proprio):
        """
        Input: (b, num_frames, proprio_dim)
            num_frames must ne multiple of tubelet_size_enc
        Output: (b, num_frames // tubelet_size_enc, 1, proprio_emb_dim * tubelet_size_enc)
        """
        B, T, D = proprio.shape
        proprio = proprio.reshape(B, -1, self.proprio_dim)
        if self.proprio_encoder_inpred:
            return proprio
        else:
            proprio = self.proprio_encoder(proprio)
        if self.proprio_encoding == "feature":
            proprio = repeat(proprio, "b t 1 a -> b t f a", f=self.grid_size**2)
        return proprio

    def encode(self, obs, a):
        """
        Output:
            video_features: (b tau v h w d)
            proprio_features: (b tau 1 d)
            action_features: (b tau 1 d)
        where tau = num_frames // tubelet_size_enc = grid_depth
        """
        encoded_obs = self.encode_obs(obs)
        video_features = encoded_obs["visual"]
        proprio_features = encoded_obs["proprio"]
        if self.use_action:
            action_features = self.encode_act(a)
        else:
            action_features = None
        return video_features, proprio_features, action_features

    def forward(
        self,
        obs,
        a,
        ctxt_window=None,
    ):
        """
        Args:
            obs: observation containing "visual" and "proprio"
            a (B, T, A): actions
            s (B, T, S): states
            use_action_encoder (bool):
            use_proprio (bool):
            use_proprio (bool):
            world_size (int):
            stats (dict):
        """
        video_features, proprio_features, action_features = self.encode(obs, a)
        pred_video_features, pred_action_features, pred_proprio_features = self.forward_pred(
            video_features,
            action_features,
            proprio_features,
            ctxt_window=ctxt_window,
        )

        return (
            pred_video_features,
            pred_action_features,
            pred_proprio_features,
            video_features,
            proprio_features,
            action_features,
        )

    def forward_pred(
        self,
        video_features,
        action_features,
        proprio_features,
        ctxt_window=None,
        debug=False,
    ):
        """
        Forward pass through the predictor.

        Args:
            video_features: (B, tau, V, H, W, D) visual features
            action_features: (B, T, A) if action_encoder_inpred else (B, T, action_tokens, D)
            proprio_features: (B, T, P) if proprio_encoder_inpred else (B, T, proprio_tokens, D)

        Returns:
            pred_video_features: (B, T, V, H, W, D)
            pred_action_features: (B, T, 1, D) or (B, T, H*W, D) depending on conditioning
            pred_proprio_features: (B, T, 1, D) or (B, T, H*W, D) depending on encoding
        """
        B, tau, V, H, W, D = video_features.shape  # N = p**2 * num_frames / tubelet_size_enc
        if self.action_encoder_inpred:
            B, T, A = action_features.shape
        else:
            B, T, action_tokens, D = action_features.shape
        if self.pred_type == "dino_wm":
            video_features = rearrange(
                video_features, "b t v h w d -> b t (v h w) d", h=self.grid_size, w=self.grid_size
            )
            features = self.concat_obs_act(video_features, action_features, proprio_features)
            features = rearrange(features, "b t p d -> b (t p) d")
            pred_features = self.predictor(features)
            pred_features = rearrange(pred_features, "b (t p) d -> b t p d", t=T)
            pred_video_features, pred_action_features = (
                pred_features[:, :, :, : -action_features.shape[-1]],
                pred_features[:, :, :, -action_features.shape[-1] :],
            )
            if proprio_features is not None:
                pred_video_features, pred_proprio_features = (
                    pred_video_features[:, :, :, : -proprio_features.shape[-1]],
                    pred_video_features[:, :, :, -proprio_features.shape[-1] :],
                )
            else:
                pred_proprio_features = None
            pred_video_features = rearrange(
                pred_video_features, "b t (v h w) d -> b t v h w d", h=self.grid_size, w=self.grid_size
            )
        elif self.pred_type == "vjepa2_ac":
            pred_video_features, pred_action_features, pred_proprio_features = self.predictor(
                video_features.flatten(1, 4),  # (b, tau * v * h * w, d)
                action_features,
                proprio_features if proprio_features is not None else None,
            )
            pred_video_features = rearrange(
                pred_video_features, "b (t v h w) d -> b t v h w d", h=self.grid_size, w=self.grid_size, v=1
            )
        elif self.pred_type == "AdaLN":
            pred_video_features, pred_action_features, pred_proprio_features = self.predictor(
                video_features,
                action_features,
                proprio_features,
            )
            pred_video_features = rearrange(
                pred_video_features, "b t (v h w) d -> b t v h w d", h=self.grid_size, w=self.grid_size, v=1
            )
        else:
            raise ValueError(f"self.pred_type should be in ['dino_wm', 'vjepa2_ac', 'AdaLN']")
        if self.normalize_reps:
            pred_video_features = F.layer_norm(pred_video_features, (pred_video_features.size(-1),))
        return pred_video_features, pred_action_features, pred_proprio_features

    def concat_obs_act(self, z_vis, act_emb, proprio_emb):
        """
        input :  obs (dict): "visual", "proprio", (b, num_frames, 3, img_size, img_size)
        output:    z (tensor): (b, num_frames, num_patches, emb_dim)
        """
        if proprio_emb is not None:
            z = torch.cat([z_vis, proprio_emb, act_emb], dim=3)  # (b, num_frames, num_patches, dim + action_dim)
        else:
            z = torch.cat([z_vis, act_emb], dim=3)  # (b, num_frames, num_patches, dim + action_dim)
        return z

    def compute_loss(
        self,
        pred_video_features,
        pred_proprio_features,
        video_features,
        proprio_features,
        visual_loss=True,
        shift=1,
        num_views=1,
        reduce_mean=True,
    ):
        """
        Input:
            proprio_features: [B, T, 1, D]
            video_features: [B, T, V, H, W, D]
        """
        proprio_loss = self.proprio_loss
        if not self.use_proprio or pred_proprio_features is None:
            proprio_loss = False
        B, T, V, H, W, C = pred_video_features.shape
        V = num_views
        H, W = self.grid_size, self.grid_size
        pred_video_features = pred_video_features.reshape(B, T, V * H * W, C)
        video_features = video_features.reshape(B, T, V * H * W, C)
        if shift != 0:
            visual_targets_ = video_features[:, shift:]
            visual_features_ = pred_video_features[:, :-shift]
            if proprio_loss:
                proprio_targets_ = proprio_features[:, shift:]
                proprio_features_ = pred_proprio_features[:, :-shift]
        else:
            visual_targets_ = video_features
            visual_features_ = pred_video_features
            if proprio_loss:
                proprio_targets_ = proprio_features
                proprio_features_ = pred_proprio_features
        loss = 0.0
        visual_cos_loss = -(
            visual_features_
            * visual_targets_
            / (visual_features_.norm(dim=-1, keepdim=True) * visual_targets_.norm(dim=-1, keepdim=True))
        ).sum(-1)
        visual_l1_loss = l1_(visual_features_, visual_targets_).mean(dim=-1)
        visual_l2_loss = l2_(visual_features_, visual_targets_).mean(dim=-1)
        visual_smooth_l1_loss = smooth_l1_(visual_features_, visual_targets_).mean(dim=-1)
        if proprio_loss:
            proprio_cos_loss = -(
                proprio_features_
                * proprio_targets_
                / (proprio_features_.norm(dim=-1, keepdim=True) * proprio_targets_.norm(dim=-1, keepdim=True))
            ).sum(-1)
            proprio_l1_loss = l1_(proprio_features_, proprio_targets_).mean(dim=-1)
            proprio_l2_loss = l2_(proprio_features_, proprio_targets_).mean(dim=-1)
            proprio_smooth_l1_loss = smooth_l1_(proprio_features_, proprio_targets_).mean(dim=-1)
        # Combine losses
        if visual_loss:
            loss += self.cfgs_loss["cos_loss_weight"] * visual_cos_loss
            loss += self.cfgs_loss["l1_loss_weight"] * visual_l1_loss
            loss += self.cfgs_loss["l2_loss_weight"] * visual_l2_loss
            loss += self.cfgs_loss["smooth_l1_loss_weight"] * visual_smooth_l1_loss
        if proprio_loss:
            loss += self.cfgs_loss["cos_loss_weight"] * proprio_cos_loss
            loss += self.cfgs_loss["l1_loss_weight"] * proprio_l1_loss
            loss += self.cfgs_loss["l2_loss_weight"] * proprio_l2_loss
            loss += self.cfgs_loss["smooth_l1_loss_weight"] * proprio_smooth_l1_loss
        out = {
            "loss": loss.mean() if reduce_mean else loss,
            "visual_cos_loss": visual_cos_loss.mean() if reduce_mean else visual_cos_loss,
            "visual_l1_loss": visual_l1_loss.mean() if reduce_mean else visual_l1_loss,
            "visual_l2_loss": visual_l2_loss.mean() if reduce_mean else visual_l2_loss,
            "visual_smooth_l1_loss": visual_smooth_l1_loss.mean() if reduce_mean else visual_smooth_l1_loss,
        }
        if proprio_loss:
            out.update(
                {
                    "proprio_cos_loss": proprio_cos_loss.mean() if reduce_mean else proprio_cos_loss,
                    "proprio_l1_loss": proprio_l1_loss.mean() if reduce_mean else proprio_l1_loss,
                    "proprio_l2_loss": proprio_l2_loss.mean() if reduce_mean else proprio_l2_loss,
                    "proprio_smooth_l1_loss": proprio_smooth_l1_loss.mean() if reduce_mean else proprio_smooth_l1_loss,
                }
            )
        return out

    def rollout(
        self,
        video_features=None,
        pred_video_features=None,
        proprio_features=None,
        pred_proprio_features=None,
        action_features=None,
        # Rollout parameters
        action_noise=0.0,
        loss_weight=1.0,
        rollout_steps=None,
        rollout_stop_gradient=True,
        ctxt_window=8,
        debug=False,
        # Mode selection
        mode="sequential",  # 'sequential' or 'parallel'
        t=None,  # For sequential mode: timestep at which to cut between prefix and suffix
        gt_prob=0.0,  # For parallel mode
        prepend_gt=False,  # For parallel mode only
    ):
        """
        Rollout function that supports two modes:

        1. 'sequential': Standard sequential rollout
           - Pass full sequences (video_features, pred_video_features, etc.) and timestep t
           - The function constructs sequences internally using timestep t
           - Creates video_feature_sequence by concatenating encoder features up to t+1 with predictor feature at t
           - Rolls out using actions from t+1 onwards
           - Compares predictions to ground truth features from t+2 onwards

        2. 'parallel': Parallel rollout training predictor from its own predictions
           - Uses predictor's own predictions as context for next predictions
           - Optionally mixes ground truth with predictions (scheduled sampling)

        Args:
            video_features: Full video features from encoder [B, T, V, H, W, D]
            pred_video_features: Predicted video features from predictor [B, T, V, H, W, D]
            proprio_features: Full proprio features from encoder [B, T, 1, D] if not proprio_encoder_inpred else [B, T, proprio_dim] (optional)
            pred_proprio_features: Predicted proprio features from predictor [B, T, 1, D] (optional)
            action_features: Full action features [B, T, 1, D] if not action_encoder_inpred else [B, T, A]

            action_noise: Standard deviation of noise to add to actions (sequential mode)
            loss_weight: Weight to apply to losses
            rollout_steps: Number of steps to roll out
            rollout_stop_gradient: Whether to detach features during rollout
            ctxt_window: Context window size for predictions
            debug: Debug flag

            mode: 'sequential' or 'parallel'
            t: Timestep for sequential mode (where to cut between prefix and suffix). Required for sequential mode.
            gt_prob: Probability of using GT in parallel mode (for scheduled sampling)
            prepend_gt: Whether to prepend GT in parallel mode
        """
        loss_weight = loss_weight / (rollout_steps + 1)
        rollout_losses = defaultdict(list)
        total_rollout_loss = 0

        if mode == "sequential":
            if t is None:
                raise ValueError("timestep t must be provided for sequential mode")

            if pred_video_features is not None:
                # Standard case: concatenate encoder features with predictor features
                current_vid_history = video_features[:, : t + 1]
                next_video_step = pred_video_features[:, t : t + 1]
                vid_feats = torch.cat([current_vid_history, next_video_step], dim=1)
                act_feats = action_features[:, : t + 1]
                if proprio_features is not None:
                    if self.proprio_rollout_mode == "use_ground_truth":
                        prop_feats = proprio_features[:, : t + 2]
                    elif self.proprio_rollout_mode == "predict_proprio":
                        current_proprio_history = proprio_features[:, : t + 1]
                        next_proprio_step = pred_proprio_features[:, t : t + 1]
                        prop_feats = torch.cat([current_proprio_history, next_proprio_step], dim=1)
                    else:
                        raise ValueError("proprio_rollout_mode must be 'use_ground_truth' or 'predict_proprio' ")
                else:
                    prop_feats = None
                act_feats_suffix = action_features[:, t + 1 :]
                vid_feats_suffix = video_features[:, t + 2 :]
                prop_feats_suffix = proprio_features[:, t + 2 :] if proprio_features is not None else None
            else:
                # Starting from scratch: no predictor features to concatenate
                vid_feats = video_features[:, : t + 1]
                act_feats = action_features[:, :t]
                prop_feats = proprio_features[:, : t + 1] if proprio_features is not None else None
                act_feats_suffix = action_features[:, t:]
                vid_feats_suffix = video_features[:, t + 1 :]
                prop_feats_suffix = proprio_features[:, t + 1 :] if proprio_features is not None else None

        elif mode == "parallel":
            if pred_video_features is not None:
                shift = 2
                # Standard case: initialize from 1st-order predictions
                next_vid_feats = pred_video_features[:, :-1]
                act_feats = action_features[:, :-1]
                if self.proprio_rollout_mode == "use_ground_truth":
                    next_prop_feats = proprio_features[:, :-1] if proprio_features is not None else None
                elif self.proprio_rollout_mode == "predict_proprio":
                    next_prop_feats = pred_proprio_features[:, :-1] if proprio_features is not None else None
                else:
                    raise ValueError("proprio_rollout_mode must be 'use_ground_truth' or 'predict_proprio' ")
                out_vid_feats = video_features[:, :1]
                out_prop_feats = proprio_features[:, :1] if proprio_features is not None else None
            else:
                shift = 1
                # Starting from scratch: need to compute first prediction in first iteration
                next_vid_feats = video_features.clone()
                act_feats = action_features.clone()
                next_prop_feats = proprio_features.clone() if proprio_features is not None else None
                # Initialize output sequences with first encoder features
                out_vid_feats = video_features[:, :1]
                out_prop_feats = proprio_features[:, :1] if proprio_features is not None else None

        for h in range(rollout_steps):
            if mode == "parallel":
                # Prune latest predictions from previous step
                vid_feats = next_vid_feats[:, :-1]
                act_feats = act_feats[:, :-1]
                if proprio_features is not None:
                    prop_feats = next_prop_feats[:, :-1]

                # Optionally prepend ground truth
                if prepend_gt:
                    vid_feats = torch.cat([video_features[:, :1], vid_feats], dim=1)
                    if proprio_features is not None:
                        prop_feats = torch.cat([proprio_features[:, :1], prop_feats], dim=1)
                    act_feats = torch.cat([action_features[:, :1], act_feats], dim=1)

                # Apply scheduled sampling to context
                if gt_prob > 0:
                    start_idx = 1 if prepend_gt else 0
                    if start_idx < vid_feats.shape[1]:
                        seq_len = vid_feats.shape[1] - start_idx

                        # Create sampling mask
                        mask = torch.rand(vid_feats.shape[0], seq_len, device=vid_feats.device) < gt_prob
                        mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

                        # Mix GT and predicted features
                        gt_timesteps = video_features[:, start_idx : start_idx + seq_len]
                        mixed_feats = torch.where(
                            mask_expanded, gt_timesteps, vid_feats[:, start_idx : start_idx + seq_len]
                        )
                        vid_feats = torch.cat([vid_feats[:, :start_idx], mixed_feats], dim=1)

                        if proprio_features is not None:
                            gt_prop_timesteps = proprio_features[:, start_idx : start_idx + seq_len]
                            prop_mixed_feats = torch.where(
                                mask.unsqueeze(-1).unsqueeze(-1),
                                gt_prop_timesteps,
                                prop_feats[:, start_idx : start_idx + seq_len],
                            )
                            prop_feats = torch.cat([prop_feats[:, :start_idx], prop_mixed_feats], dim=1)
            else:
                # Sequential mode: append new action
                new_act_feats = act_feats_suffix[:, h : h + 1]
                if action_noise > 0:
                    new_act_feats = (
                        torch.randn(*new_act_feats.shape).to(new_act_feats.device).to(new_act_feats.dtype)
                        * action_noise
                    )
                act_feats = torch.cat([act_feats, new_act_feats], dim=1)

            # Forward prediction with optional context window
            next_vid_feats, next_act_feats, next_prop_feats = self.forward_pred(
                vid_feats[:, -ctxt_window:].detach() if rollout_stop_gradient else vid_feats[:, -ctxt_window:],
                act_feats[:, -ctxt_window:],
                (
                    prop_feats[:, -ctxt_window:].detach()
                    if rollout_stop_gradient and prop_feats is not None
                    else prop_feats[:, -ctxt_window:] if prop_feats is not None else prop_feats
                ),
                debug=debug,
            )

            if mode == "parallel":
                # build the output unrolled sequence from the "diagonal"
                out_vid_feats = torch.cat(
                    [out_vid_feats, next_vid_feats[:, h * prepend_gt : 1 + h * prepend_gt]], dim=1
                )
                if next_prop_feats is not None:
                    out_prop_feats = torch.cat(
                        [out_prop_feats, next_prop_feats[:, h * prepend_gt : 1 + h * prepend_gt]], dim=1
                    )
            else:
                next_vid_feat = next_vid_feats[:, -1:]
                vid_feats = torch.cat(
                    [vid_feats.detach() if rollout_stop_gradient else vid_feats, next_vid_feat], dim=1
                )
                if self.use_proprio and proprio_features is not None:
                    if self.proprio_rollout_mode == "use_ground_truth":
                        next_prop_feat = proprio_features[:, t + 2 + h : t + 2 + h + 1]
                    elif self.proprio_rollout_mode == "predict_proprio":
                        next_prop_feat = next_prop_feats[:, -1:]
                    else:
                        raise ValueError("proprio_rollout_mode must be 'use_ground_truth' or 'predict_proprio' ")
                    prop_feats = torch.cat(
                        [prop_feats.detach() if rollout_stop_gradient else prop_feats, next_prop_feat], dim=1
                    )
                else:
                    next_prop_feat = None
                    prop_feats = None

            # Loss computation
            if mode == "parallel":
                vid_targets = video_features[:, shift + h * (1 - prepend_gt) :].detach()
                prop_targets = (
                    proprio_features[:, shift + h * (1 - prepend_gt) :].detach()
                    if self.use_proprio and proprio_features is not None
                    else None
                )
                losses = self.compute_loss(next_vid_feats, next_prop_feats, vid_targets, prop_targets, shift=0)
            else:
                if vid_feats_suffix is not None:
                    vid_targets = vid_feats_suffix[:, h : h + 1].detach()
                    prop_targets = (
                        prop_feats_suffix[:, h : h + 1].detach()
                        if self.use_proprio and prop_feats_suffix is not None
                        else None
                    )
                    losses = self.compute_loss(next_vid_feat, next_prop_feat, vid_targets, prop_targets, shift=0)
                else:
                    losses = None

            if losses is not None:
                for k in losses:
                    rollout_losses[k].append(losses[k].detach() if hasattr(losses[k], "detach") else losses[k])
                loss = losses["loss"] * loss_weight
                total_rollout_loss += loss

        device = video_features.device if video_features is not None else action_features.device
        for k in rollout_losses:
            rollout_losses[k] = (
                torch.stack(rollout_losses[k])
                if rollout_losses[k] and isinstance(rollout_losses[k][0], torch.Tensor)
                else (
                    torch.tensor(rollout_losses[k], device=device)
                    if rollout_losses[k]
                    else torch.tensor(0.0, device=device)
                )
            )
        if mode == "parallel":
            # For parallel mode, return None for vid_feats and prop_feats as they're not used
            return rollout_losses, total_rollout_loss, out_vid_feats, out_prop_feats
        else:
            return rollout_losses, total_rollout_loss, vid_feats, prop_feats

    def backward(self, loss):
        """
        Copy-paste from TrainableModel class
        """
        if self.mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def optimization_step(self):
        """
        Copy-paste from TrainableModel class
        """
        self.scaler.unscale_(self.optimizer)
        if self.clip_grad > 0:
            _grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad)
            if self.use_radamw and (_grad_norm > self.clip_grad):
                logger.info(f"Gradient spike... skipping update {_grad_norm=}")
                self.optimizer.skip_step()
        if self.mixed_precision:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        if self.clip_grad > 0:
            grad_stats = grad_logger(self.named_parameters())
            grad_stats.global_norm = float(_grad_norm)
        else:
            grad_stats = None
        self.optimizer.zero_grad()

        # Detect optimizer type and call appropriate logger
        optimizer_name = type(self.optimizer).__name__
        from src.utils.logging import adamw_logger

        optim_stats = adamw_logger(self.optimizer)

        return grad_stats, optim_stats

    @torch.no_grad
    def compute_energy_landscape(
        self,
        video_features,
        action_features,
        proprio_features,
        gt_visual,
        gt_proprio,
        rollout_steps=1,
        ctxt_window=3,
        proprio_dim=4,
        action_dim=4,
        actions_per_vid_feat=1,
        dataset_path="metaworld",
        preprocessor=None,
    ):
        print(f"{actions_per_vid_feat=}")
        if action_features.ndim == 4:
            orig_B, T, _, A = action_features.shape
        elif action_features.ndim == 3:
            orig_B, T, A = action_features.shape
        else:
            raise ValueError(f"Unexpected number of dimensions: {action_features.ndim}")
        all_figures = []

        b = min(orig_B, 3)
        if "metaworld" in dataset_path.lower() or "tdmpc2" in dataset_path.lower():
            AMAX, env_name = 2.0, "mw"
        elif "pusht" in dataset_path.lower():
            AMAX, env_name = 3.0, "pt"
        elif "point_maze" in dataset_path.lower():
            AMAX, env_name = 1.0, "mz"
        elif "wall" in dataset_path.lower():
            AMAX, env_name = 1.0, "wall"
        elif "droid" in dataset_path.lower():
            AMAX, env_name = 0.1, "droid"
        N1, N2 = 40, 40
        for batch_idx in range(b):
            DIMS = {"dx": 0, "dy": 1, "dz": 2}
            NAME1 = "dx"  # plot x axis
            NAME2 = "dy"  # plot y axis
            DIM1 = DIMS[NAME1]
            DIM2 = DIMS[NAME2]

            d1 = np.linspace(-AMAX, AMAX, N1)
            d2 = np.linspace(-AMAX, AMAX, N2)
            dxdz_pairs = np.array(np.meshgrid(d1, d2)).T.reshape(-1, 2)

            proc_B = 16
            dxdz_pairs_batched = torch.tensor(dxdz_pairs.reshape(-1, proc_B, 2))
            cc = np.zeros((N1, N2))

            vid_feat = video_features[batch_idx : batch_idx + 1]
            prop_feat = proprio_features[batch_idx : batch_idx + 1] if self.use_proprio else None
            act_feat = action_features[batch_idx : batch_idx + 1]

            visual_obs = gt_visual[batch_idx].cpu().numpy()
            proprio_obs = gt_proprio[batch_idx].cpu().numpy()
            if env_name == "mw":
                proprio_obs = 100 * proprio_obs
            ctxt_idx = np.random.randint(0, T - rollout_steps)
            ctxt_frames = visual_obs[
                ctxt_idx * self.tubelet_size_enc : (ctxt_idx + 1) * self.tubelet_size_enc
            ].transpose(
                0, 2, 3, 1
            )  # T C H W
            ctxt_prop = proprio_obs[ctxt_idx * self.tubelet_size_enc : (ctxt_idx + 1) * self.tubelet_size_enc]
            goal_vid_feat = vid_feat[:, ctxt_idx + rollout_steps : ctxt_idx + rollout_steps + 1].repeat(
                proc_B, 1, 1, 1, 1, 1
            )
            # Handle goal_prop_feat repeat robustly based on dimensionality
            if self.use_proprio:
                goal_slice = prop_feat[:, ctxt_idx + rollout_steps : ctxt_idx + rollout_steps + 1]
                goal_prop_feat = goal_slice.repeat(proc_B, *([1] * (goal_slice.ndim - 1)))
            else:
                goal_prop_feat = None
            goal_frames = visual_obs[
                (ctxt_idx + rollout_steps)
                * self.tubelet_size_enc : (ctxt_idx + rollout_steps + 1)
                * self.tubelet_size_enc
            ].transpose(0, 2, 3, 1)
            goal_prop = proprio_obs[
                (ctxt_idx + rollout_steps)
                * self.tubelet_size_enc : (ctxt_idx + rollout_steps + 1)
                * self.tubelet_size_enc
            ]
            true_delta_proprio = (goal_prop[-1][:proprio_dim] - ctxt_prop[0][:proprio_dim]) / (
                actions_per_vid_feat * rollout_steps
            )

            pbar = tqdm(dxdz_pairs_batched)
            for batch_pairs in pbar:
                batch_actions = torch.zeros(proc_B, rollout_steps * actions_per_vid_feat, action_dim)
                batch_actions[:, :, 0] = batch_pairs[:, 0].unsqueeze(1)  # / rollout_steps
                batch_actions[:, :, 1] = batch_pairs[:, 1].unsqueeze(1)  # / rollout_steps

                batch_actions = preprocessor.normalize_actions(batch_actions).to(self.device, dtype=torch.float32)
                encoded_batch_actions = self.encode_act(batch_actions)
                # Create context actions up to ctxt_idx to match what rollout expects
                # rollout will slice action_features[:, :t+1] for sequential mode
                encoded_act_context = (
                    act_feat[:, : ctxt_idx + 1].repeat(proc_B, 1, 1, 1)
                    if action_features.ndim == 4
                    else act_feat[:, : ctxt_idx + 1].repeat(proc_B, 1, 1)
                )
                # Handle proprio_features repeat robustly based on dimensionality
                repeated_prop_feat = (
                    prop_feat.repeat(proc_B, *([1] * (prop_feat.ndim - 1))) if self.use_proprio else None
                )

                rollout_losses, total_rollout_loss, final_vid_feats, final_prop_feats = self.rollout(
                    video_features=vid_feat.repeat(proc_B, 1, 1, 1, 1, 1),
                    proprio_features=repeated_prop_feat,
                    action_features=torch.cat([encoded_act_context, encoded_batch_actions], dim=1),
                    t=ctxt_idx,
                    rollout_steps=rollout_steps,
                    ctxt_window=ctxt_window,
                )
                costs = self.compute_loss(
                    final_vid_feats[:, -1:],
                    final_prop_feats[:, -1:] if self.use_proprio else None,
                    goal_vid_feat,
                    goal_prop_feat,
                    shift=0,
                    reduce_mean=False,
                )["loss"]
                costs = costs.mean(dim=tuple(range(1, costs.ndim)))
                for ib in range(proc_B):
                    cost = costs[ib].item()
                    ix = np.where(np.isclose(d1, batch_pairs[ib, 0].item()))[0][0]
                    iz = np.where(np.isclose(d2, batch_pairs[ib, 1].item()))[0][0]
                    cc[ix, iz] = cost
            fig = plot_energy_landscape(
                cc, d1, d2, ctxt_frames[0], goal_frames[0], ctxt_prop, goal_prop, true_delta_proprio, env_name=env_name
            )
            all_figures.append(fig)
        # Stack all images vertically
        if all_figures:
            final_fig = plt.figure(figsize=(24, 8 * b))
            for i, fig in enumerate(all_figures):
                fig.canvas.draw()
                img = np.array(fig.canvas.renderer._renderer)
                plt.subplot(len(all_figures), 1, i + 1)
                plt.imshow(img)
                plt.axis("off")
            plt.tight_layout()
        else:
            final_fig = plt.figure(figsize=(10, 10))
            plt.text(
                0.5, 0.5, "No energy landscapes generated", horizontalalignment="center", verticalalignment="center"
            )
            plt.axis("off")
        return final_fig


def plot_energy_landscape(
    cc, d1, d2, ctxt_frames, goal_frames, ctxt_prop, goal_prop, true_delta_proprio, env_name="mw"
):
    fig, axs = plt.subplots(1, 3, figsize=(24, 8))

    # Context Frames
    axs[0].imshow(ctxt_frames)
    axs[0].set_title(f"Context Frames \nProprio: {ctxt_prop[0]}")
    axs[0].axis("off")

    # Goal Frames
    axs[1].imshow(goal_frames)
    axs[1].set_title(f"Goal Frame \nProprio: {goal_prop[-1]}")
    axs[1].axis("off")

    # Energy Landscape
    if env_name == "pt":
        im = axs[2].imshow(cc.T, extent=(d1[0], d1[-1], d2[0], d2[-1]), origin="lower", cmap="viridis")
        axs[2].invert_yaxis()
    else:
        im = axs[2].imshow(cc.T, extent=(d1[0], d1[-1], d2[0], d2[-1]), origin="lower", cmap="viridis")
    axs[2].scatter(true_delta_proprio[0], true_delta_proprio[1], c="r", marker="o", s=100)
    plt.colorbar(im, ax=axs[2])
    axs[2].set_xlabel("dx")
    axs[2].set_ylabel("dy")
    axs[2].set_title(f"Energy Landscape\nΔXYZ: {true_delta_proprio}")

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    return fig


def cos(a, b):
    B = a.shape[0]
    a = a.view(B, -1)
    b = b.view(B, -1)
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    cosine_similarities = (a * b).sum(dim=-1)
    average_cosine_similarity = cosine_similarities.mean(dim=0)
    return average_cosine_similarity


l2 = torch.nn.MSELoss()
l1 = torch.nn.L1Loss()
smooth_l1 = torch.nn.SmoothL1Loss()

l2_ = torch.nn.MSELoss(reduction="none")
l1_ = torch.nn.L1Loss(reduction="none")
smooth_l1_ = torch.nn.SmoothL1Loss(reduction="none")
