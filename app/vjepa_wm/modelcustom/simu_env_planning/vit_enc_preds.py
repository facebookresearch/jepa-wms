# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tensordict.tensordict import TensorDict

from app.plan_common.datasets.droid_dset import compute_new_pose
from app.plan_common.models.wm_heads import WorldModelPoseReadoutHead, WorldModelViTImageHead
from app.vjepa_wm.utils import (
    clean_state_dict,
    fetch_checkpoint,
    init_video_model,
    load_checkpoint_state_dict,
)
from app.vjepa_wm.video_wm import VideoWM

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def init_module(
    folder,
    checkpoint,
    model_kwargs,
    device,
    action_dim,
    proprio_dim,
    preprocessor,
    cfgs_data=None,
    wrapper_kwargs=None,
    **kwargs,
):
    """Initialize EncPredWM model with pretrained weights and decoders.
    Args:
        folder (str or Path): Directory containing checkpoint file (only used for local paths).
        checkpoint (str): Checkpoint source - can be either:
            - A filename to load from folder (e.g., "jepa-latest.pth.tar")
            - A full URL to download from (e.g., "https://dl.fbaipublicfiles.com/jepa-wms/...")
        model_kwargs (dict): Model configuration with encoder, predictor, heads, and data configs.
        device (torch.device): Device to load model on.
        action_dim (int): Action dimension from the dataset.
        proprio_dim (int): Proprioception dimension from the dataset.
        preprocessor: Preprocessor with transform, inverse_transform, normalize methods.
        cfgs_data (dict): Data configuration with img_size, custom settings, etc.
        wrapper_kwargs (dict): Additional kwargs for EncPredWM wrapper.
        **kwargs: Additional arguments (unused).
    Returns:
        EncPredWM: Wrapped VideoWM model ready for encoding and prediction.
    """
    img_size = cfgs_data.get("img_size", 256)
    frameskip = cfgs_data.get("custom").get("frameskip", 1)
    action_skip = cfgs_data.get("custom").get("action_skip", 1)
    state_skip = cfgs_data.get("custom").get("state_skip", 1)

    # Extract nested configs
    cfgs_heads = model_kwargs.get("heads_cfg", {})
    heads_architectures = cfgs_heads.get("architectures", {})
    pretrain_dec_path = cfgs_heads.get("pretrain_dec_path", None)
    new_path_heads = cfgs_heads.get("new_path_heads", {})

    cfgs_wm_encoding = model_kwargs.get("wm_encoding", {})

    # Compute derived values
    action_tokens = model_kwargs.get("action_encoder", {}).get("action_tokens", 1)
    proprio_tokens = model_kwargs.get("proprio_encoder", {}).get("proprio_tokens", 1)
    action_emb_dim = model_kwargs.get("action_encoder", {}).get("action_emb_dim", 0)
    proprio_emb_dim = model_kwargs.get("proprio_encoder", {}).get("proprio_emb_dim", 0)
    use_proprio = proprio_tokens > 0 or proprio_emb_dim > 0
    use_action = action_tokens > 0 or action_emb_dim > 0
    tubelet_size_enc = model_kwargs.get("tubelet_size_enc", 2)

    # Compute model dimensions
    if use_action:
        model_action_dim = action_dim * tubelet_size_enc * frameskip // action_skip
    else:
        model_action_dim = None
    if use_proprio:
        model_proprio_dim = proprio_dim * tubelet_size_enc // state_skip
    else:
        model_proprio_dim = None

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
        "data",
        "data_aug",
    ]
    init_model_kwargs = {k: v for k, v in model_kwargs.items() if k not in excluded_keys}
    if "visual_encoder" in model_kwargs:
        init_model_kwargs.update(model_kwargs["visual_encoder"])
    if "action_encoder" in model_kwargs:
        init_model_kwargs.update(model_kwargs["action_encoder"])
    if "proprio_encoder" in model_kwargs:
        init_model_kwargs.update(model_kwargs["proprio_encoder"])
    if "predictor" in model_kwargs:
        init_model_kwargs.update(model_kwargs["predictor"])
    init_model_kwargs.update(
        {
            "device": device,
            "img_size": img_size,
            "action_dim": model_action_dim,
            "proprio_dim": model_proprio_dim,
            "cfgs_attn_pattern": model_kwargs.get("attn", None),
            "use_proprio": use_proprio,
            "use_action": use_action,
            "use_activation_checkpointing": False,
        }
    )

    predictor, encoder, action_encoder, proprio_encoder = init_video_model(**init_model_kwargs)

    heads = {}
    if pretrain_dec_path:
        if "image_head" in heads_architectures:
            image_head_type = heads_architectures["image_head"]["kind"]
            if image_head_type is not None and image_head_type.lower() != "none":
                if image_head_type == "vit":
                    decoder = WorldModelViTImageHead(
                        head_config=dict(heads_architectures["image_head"]["config"]),
                        inverse_transform=preprocessor.inverse_transform,
                        device=device,
                    )
                heads["image_head"] = decoder
        if "state_head" in heads_architectures:
            state_decoder = WorldModelPoseReadoutHead(
                head_config=dict(heads_architectures["state_head"]["config"]), device=device
            )
            heads["state_head"] = state_decoder

    is_url = isinstance(checkpoint, str) and checkpoint.startswith(("http://", "https://"))
    if is_url:
        checkpoint_source = checkpoint
    else:
        checkpoint_source = Path(folder) / checkpoint

    checkpoint_data = fetch_checkpoint(checkpoint_source, device="cpu")

    (
        predictor,
        action_encoder,
        proprio_encoder,
        _,
        _,
        _,
    ) = load_checkpoint_state_dict(
        checkpoint=checkpoint_data,
        predictor=predictor,
        action_encoder=action_encoder,
        proprio_encoder=proprio_encoder,
    )
    del checkpoint_data

    # Load heads from pretrain_dec_path (separate head checkpoint files)
    if pretrain_dec_path:
        for name, head in heads.items():
            if new_path_heads.get(name, True):
                head_path = pretrain_dec_path[name].removesuffix(".pth.tar") + "_" + name + ".pth.tar"
                head.load_checkpoint(head_path)
                logger.info(f"loaded pretrained head named {name}")
            else:
                head_checkpoint = torch.load(pretrain_dec_path[name], map_location=torch.device("cpu"))
                epoch = head_checkpoint["epoch"]
                pretrained_dict = clean_state_dict(head_checkpoint[name])
                msg = head.model.load_state_dict(pretrained_dict, strict=False)
                logger.info(f"loaded pretrained head named {name} from epoch {epoch} with msg: {msg}")
                del head_checkpoint

    if wrapper_kwargs is None:
        wrapper_kwargs = {}
    proprio_mode = wrapper_kwargs.get("proprio_mode", "predict_proprio")
    proprio_loss = proprio_mode == "predict_proprio"

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
        # From model_kwargs (pass directly from config)
        "action_tokens": action_tokens,
        "proprio_tokens": proprio_tokens,
        "grid_size": model_kwargs.get("grid_size", 14),
        "tubelet_size_enc": tubelet_size_enc,
        "action_conditioning": model_kwargs.get("action_conditioning", "token"),
        "proprio_encoding": model_kwargs.get("proprio_encoding", "feature"),
        "enc_type": model_kwargs.get("visual_encoder", {}).get("enc_type", "vjepa"),
        "pred_type": model_kwargs.get("predictor", {}).get("pred_type", "dino_wm"),
        "action_encoder_inpred": model_kwargs.get("action_encoder", {}).get("action_encoder_inpred", False),
        "proprio_encoder_inpred": model_kwargs.get("proprio_encoder", {}).get("proprio_encoder_inpred", False),
        **cfgs_wm_encoding,
        # From cfgs_data
        "action_skip": action_skip,
        "frameskip": frameskip,
        "img_size": img_size,
        # Heads
        "heads": heads,
        # Loss config (proprio_loss derived from wrapper_kwargs.proprio_mode)
        "cfgs_loss": {
            "proprio_loss": proprio_loss,
            "cos_loss_weight": 0.0,
            "l1_loss_weight": 0.0,
            "l2_loss_weight": 1.0,
            "smooth_l1_loss_weight": 0.0,
        },
    }
    model = VideoWM(**wm_kwargs)
    model.eval()
    model = EncPredWM(
        model,
        action_dim=model_action_dim,
        preprocessor=preprocessor,
        **wrapper_kwargs,
    )
    return model


class EncPredWM(nn.Module):
    """Wrapper around VideoWM for encoding, prediction unrolling, and decoding.
    Provides interfaces for encoding raw observations into latent space, unrolling
    predictions conditioned on actions, and decoding predictions back to visual/state space.
    """

    def __init__(
        self,
        model,
        action_dim,
        preprocessor,
        ctxt_window=2,
        proprio_mode="predict_proprio",
    ):
        """Args:
        proprio_mode (str): Mode for proprio handling. Options:
        - "predict_proprio": Use predictor to predict proprio features (default)
        - "compute_new_pose": Use compute_new_pose() to compute proprio from actions
        """
        super().__init__()
        self.model = model
        self.heads = model.heads
        self.device = self.model.device
        self.action_dim = action_dim
        self.tubelet_size_enc = self.model.tubelet_size_enc
        self.encode_proprio = self.model.encode_proprio
        self.encode_obs = self.model.encode_obs
        self.preprocessor = preprocessor
        self.grid_size = self.model.grid_size
        self.action_skip = self.model.action_skip
        self.normalize_reps = self.model.normalize_reps
        self.enc_type = self.model.enc_type
        # wrapper_kwargs
        self.ctxt_window = ctxt_window
        self.proprio_mode = proprio_mode

    def unroll(self, z_ctxt, act_suffix=None, debug=False):
        """Autoregressively predict latent features forward in time using actions.

        Starts from context features and iteratively predicts next timestep using
        action conditioning. Maintains a sliding window of ctxt_window frames for prediction.

        Args:
            z_ctxt (TensorDict or Tensor): Context latent features.
                If TensorDict: keys "visual" [B, tau, V, H, W, D] and "proprio" [B, tau, proprio_tokens, D].
                If Tensor: visual features only [B, tau, V, H, W, D].
            act_suffix (Tensor): Action sequence [T, B, A] where A matches predictor's expected action dim.
            debug (bool): Enable debug mode in forward_pred.

        Returns:
            TensorDict or Tensor: Predicted latent features [T+tau, B, V, H, W, D].
                Returns same type as z_ctxt input.
        """
        T, B, A = act_suffix.shape
        if isinstance(z_ctxt, TensorDict) or isinstance(z_ctxt, dict):
            vid_feats_prefix = z_ctxt["visual"].expand(act_suffix.shape[1], *z_ctxt["visual"].shape[1:])
            prop_feats_prefix = z_ctxt["proprio"].expand(act_suffix.shape[1], *z_ctxt["proprio"].shape[1:])
        elif isinstance(z_ctxt, torch.Tensor):
            vid_feats_prefix = z_ctxt.expand(act_suffix.shape[1], *z_ctxt.shape[1:])
            prop_feats_prefix = None
        vid_feats = vid_feats_prefix
        prop_feats = prop_feats_prefix
        # CAUSE OF THE BUG: act_suffix = rearrange(act_suffix, "t b (act_suffix tube) -> b (t tube) act_suffix", tube=self.tubelet_size_enc)
        act_suffix = rearrange(act_suffix, "t b ... -> b t ...")
        act_feats_suffix = self.model.encode_act(act_suffix)  # (b t a) or (b t 1 d) if action_encoder_inpred=False
        for h in range(T):
            new_act_feats = act_feats_suffix[:, h : h + 1]
            if h == 0:
                act_feats = new_act_feats
            else:
                act_feats = torch.cat([act_feats, new_act_feats], dim=1)
            pred_video_features, _, pred_proprio_features = self.model.forward_pred(
                vid_feats[:, -self.ctxt_window :],
                act_feats[:, -self.ctxt_window :],
                prop_feats[:, -self.ctxt_window :] if prop_feats is not None else None,
                debug=debug,
            )
            next_vid_feat = pred_video_features[:, -1:]
            # self.normalize_reps already done in self.model.forward_pred()
            if prop_feats is not None:
                if self.proprio_mode == "compute_new_pose":
                    # Use compute_new_pose to compute proprio from actions
                    # act_suffix has raw actions in shape [B, T, A]
                    next_prop_feat = compute_new_pose(prop_feats[:, -1:], act_suffix[:, h : h + 1])
                elif self.proprio_mode == "predict_proprio":
                    # Use predictor to predict proprio
                    next_prop_feat = pred_proprio_features[:, -1:]
                else:
                    raise ValueError(
                        f"Invalid mode: {self.proprio_mode}. Must be 'predict_proprio' or 'compute_new_pose'"
                    )
            vid_feats = torch.cat([vid_feats, next_vid_feat], dim=1)
            if prop_feats is not None:
                prop_feats = torch.cat([prop_feats, next_prop_feat], dim=1)
        if isinstance(z_ctxt, TensorDict):
            vid_feats = rearrange(vid_feats, "b t ... -> t b ...")
            prop_feats = rearrange(prop_feats, "b t ... -> t b ...")
            return TensorDict({"visual": vid_feats, "proprio": prop_feats})
        elif isinstance(z_ctxt, torch.Tensor):
            vid_feats = rearrange(vid_feats, "b t ... -> t b ...")
            return vid_feats

    @torch.no_grad()
    def encode(self, obs, act=True):
        """Encode raw simulator observations into latent representations.

        Handles preprocessing (normalization, transforms) and encoding in a single pass
        to minimize CPU-GPU transfers. Supports both visual-only and multimodal inputs.

        Args:
            obs (TensorDict, dict, or Tensor): Raw observations from simulator.
                If dict/TensorDict: keys "visual" [B, T, C, H, W] and "proprio" [B, T, P].
                If Tensor: visual observations only [B, T, C, H, W].
            act (bool): Unused legacy parameter.

        Returns:
            TensorDict or Tensor: Latent features [B, T, V, H, W, D].
                Returns TensorDict with "visual" and "proprio" keys if input is dict, else Tensor.
        """
        if isinstance(obs, TensorDict) or isinstance(obs, dict):
            visual = obs["visual"]
            trans_proprio = self.preprocessor.normalize_proprios(obs["proprio"].cpu()).to(
                self.model.device, dtype=torch.float32, non_blocking=True
            )
            proprio_emb = self.encode_proprio(trans_proprio)
        elif isinstance(obs, torch.Tensor):
            visual = obs
        else:
            raise ValueError("Input must be a dictionary with key 'visual' or a tensor")
        b, t, c, h, w = visual.shape  # b t c h w
        visual = visual.to(self.model.device, non_blocking=True, dtype=torch.float32)
        trans_visual = visual / 255.0  # instead of calling preprocessor.preprocess_obs_visual()
        # same transform as train time transform part of dataloader
        trans_visual = self.preprocessor.transform(trans_visual)
        if self.model.batchify_video:
            trans_visual = rearrange(trans_visual, "b t ... -> (b t) ...")
        if self.model.dup_image:
            if not self.model.batchify_video:
                trans_visual = trans_visual.repeat_interleave(2, dim=1)  # b t c h w -> b 2*t c h w
            else:
                trans_visual = trans_visual.unsqueeze(2).repeat(1, 1, 2, 1, 1)  # b c h w -> b c 2 h w
                # vjepa expects (b c t h w), so no rearrange needed below
        else:
            # if we feed t=1 to a model expecting at least t=2, need to duplicate
            # self.tubelet_size_enc==1 for self.enc_type == "dino"
            if self.enc_type == "vjepa":
                trans_visual = trans_visual.repeat(1, self.tubelet_size_enc, 1, 1, 1)  # b 1 c h w -> b t c h w
        if self.enc_type == "dino":
            visual_embs = self.model.encoder(trans_visual)
            visual_embs = rearrange(
                visual_embs, "(b t) (h w) d -> b t 1 h w d", b=b, h=self.grid_size, w=self.grid_size
            )
        elif self.enc_type == "vjepa":
            if not self.model.batchify_video:
                trans_visual = rearrange(trans_visual, "b t c h w -> b c t h w ")
            visual_embs = self.model.encoder(trans_visual)
            if self.model.batchify_video:
                visual_embs = rearrange(
                    visual_embs, "(b t) (h w) d -> b t 1 h w d", b=b, t=t, h=self.grid_size, w=self.grid_size
                )
            else:
                visual_embs = rearrange(visual_embs, "b (t h w) d -> b t 1 h w d", h=self.grid_size, w=self.grid_size)
        if self.normalize_reps:
            visual_embs = F.layer_norm(visual_embs, (visual_embs.size(-1),))
        if isinstance(obs, TensorDict) or isinstance(obs, dict):
            return TensorDict({"visual": visual_embs, "proprio": proprio_emb}, device=visual_embs.device)
        elif isinstance(obs, torch.Tensor):
            return visual_embs

    @torch.no_grad()
    def decode_unroll(self, predicted_encs, batch=False):
        """Decode predicted latent features back to visual observations.

        Uses the image_head decoder to reconstruct RGB frames from latent predictions.

        Args:
            predicted_encs (TensorDict, dict, or Tensor): Predicted latent features.
                If dict/TensorDict: key "visual" [T, B, V, H, W, D].
                If Tensor: visual features [T, B, V, H, W, D].
            batch (bool): If True, return batch dimension [B, T, H, W, 3], else [T, H, W, 3].

        Returns:
            ndarray: Decoded RGB frames as uint8 in [0, 255].
                Shape [B, T, H, W, 3] if batch=True, else [T, H, W, 3].
        """
        if isinstance(predicted_encs, TensorDict) or isinstance(predicted_encs, dict):
            visual_feat_preds = predicted_encs["visual"]
            proprio_feat_preds = predicted_encs["proprio"]
        elif isinstance(predicted_encs, torch.Tensor):
            visual_feat_preds = predicted_encs
        if "image_head" in self.model.heads:
            visual_feat_preds = rearrange(visual_feat_preds, "t b v h w c -> b t v h w c ")
            eval_image_samples = self.model.heads["image_head"].decode(visual_feat_preds)
            if batch:
                pred_frames = eval_image_samples[:, :, 0].cpu().numpy()
            else:
                pred_frames = eval_image_samples[0, :, 0].cpu().numpy()
            return pred_frames
        # TODO: if proprio decoder heads, also return its decoding
