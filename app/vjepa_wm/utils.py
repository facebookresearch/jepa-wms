# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import sys

import torch

import src.models.ac_predictor as vit_ac_pred
import src.models.vision_transformer_v2 as vit_v2_open
from app.plan_common.models.AdaLN_vit import vit_predictor_AdaLN
from app.plan_common.models.dino import DinoEncoder
from app.plan_common.models.prop_embedding import ProprioceptiveEmbedding
from app.plan_common.models.vit import ViTPredictor
from src.utils.adamw import AdamW as RAdamW
from src.utils.schedulers import CosineWDSchedule, WarmupCosineSchedule, WSDSchedule
from src.utils.tensors import trunc_normal_
from src.utils.yaml_utils import load_yaml

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def clean_state_dict(state_dict):
    """Remove 'module.' prefix from state_dict keys."""
    out = {k.replace("module.", ""): v for k, v in state_dict.items()}
    return out


def update_tag_pattern(pref_tag, pattern_id, value, insert_before=None):
    """
    Updates a tag pattern in pref_tag string.
    If pattern already exists (like _H123), replaces it with new value.
    Otherwise appends the new pattern to the tag, or inserts it before another pattern.

    Args:
        pref_tag (str): The current preference tag
        pattern_id (str): Pattern identifier (e.g., "H", "maxstp", "sum")
        value: The new value (can be a boolean or any other type)
        insert_before (str, optional): Pattern identifier to insert before (e.g., "ctxt")

    Returns:
        str: Updated preference tag
    """
    import re

    # Handle boolean values: append/insert pattern_id only if True, skip if False
    if isinstance(value, bool):
        if not value:
            return pref_tag
        new_pattern = f"_{pattern_id}"
    else:
        new_pattern = f"_{pattern_id}{value}"

    # Check if pattern already exists
    pattern = f"_{pattern_id}\\d+"
    if re.search(pattern, pref_tag):
        return re.sub(pattern, new_pattern, pref_tag)

    # Insert before specified pattern if requested
    if insert_before is not None:
        before_pattern = f"_{insert_before}\\d+"
        match = re.search(before_pattern, pref_tag)
        if match:
            # Insert new_pattern right before the matched pattern
            pos = match.start()
            return pref_tag[:pos] + new_pattern + pref_tag[pos:]

    # Otherwise append at the end
    return f"{pref_tag}{new_pattern}"


def build_unroll_decode_eval_args(
    app_name,
    folder,
    checkpoint,
    cfgs_model,
    cfgs_data,
    cfgs_data_aug,
    cfgs_unroll_decode_evals,
    tag,
    eval_nodes=1,
    eval_tasks_per_node=1,
):
    """
    Builds evaluation arguments for unroll decode evaluations.

    This evaluation allows hardcoding custom actions to generate counterfactual decodings,
    e.g., comparing "open gripper + move up" vs "close gripper + move up" predictions.

    Args:
        app_name (str): Name of the application.
        folder (str): Folder for logging outputs.
        checkpoint (str): Checkpoint filename to load.
        cfgs_model (dict): Model configuration.
        cfgs_data (dict): Data configuration.
        cfgs_data_aug (dict): Data augmentation configuration.
        cfgs_unroll_decode_evals (dict): Unroll decode eval configuration containing:
            - specific_video (bool): Use a specific video file instead of dataset samples
            - specific_video_path (str): Path to specific video file (npz format)
            - play_in_reverse (bool): Reverse the video sequence
            - obs (str): Observation type - "rgb" or "rgb_state"
            - save_decoding_only (bool): Only save decoded predictions
            - repeat_hardcode_act (int): Number of times to repeat hardcoded actions
            - wrapper_kwargs (dict): Model wrapper configuration (same as evals.wrapper_kwargs)
                - ctxt_window (int): Context window size for the model wrapper
                - proprio_mode (str): Proprioception mode (e.g., "compute_new_pose")
        tag (str): Tag for the evaluation.
        eval_nodes (int): Number of nodes for evaluation.
        eval_tasks_per_node (int): Number of tasks per node.

    Returns:
        list: List containing a single eval config dict.
    """
    # Extract config values with defaults
    specific_video = cfgs_unroll_decode_evals.get("specific_video", False)
    specific_video_path = cfgs_unroll_decode_evals.get("specific_video_path", None)
    play_in_reverse = cfgs_unroll_decode_evals.get("play_in_reverse", False)
    obs = cfgs_unroll_decode_evals.get("obs", "rgb")
    save_decoding_only = cfgs_unroll_decode_evals.get("save_decoding_only", False)
    repeat_hardcode_act = cfgs_unroll_decode_evals.get("repeat_hardcode_act", 5)
    wrapper_kwargs = cfgs_unroll_decode_evals.get("wrapper_kwargs", {"ctxt_window": 2})

    pref_tag = "decode_enc"
    unroll_decode_cfg = {
        "eval_name": "unroll_decode",
        "nodes": eval_nodes,
        "tasks_per_node": eval_tasks_per_node,
        "folder": folder,
        "tag": f"{pref_tag}/{tag}",
        "specific_video": specific_video,
        "specific_video_path": specific_video_path,
        "play_in_reverse": play_in_reverse,
        "obs": obs,
        "save_decoding_only": save_decoding_only,
        "repeat_hardcode_act": repeat_hardcode_act,
        "model_kwargs": {
            "module_name": f"app.{app_name}.modelcustom.simu_env_planning.vit_enc_preds",
            "checkpoint": checkpoint,
            "pretrain_kwargs": cfgs_model,
            "data": cfgs_data,
            "data_aug": cfgs_data_aug,
            "wrapper_kwargs": wrapper_kwargs,
        },
    }

    return [unroll_decode_cfg]


def _to_list(value):
    """Convert a value to a list for cartesian product. None becomes [None], lists stay as-is."""
    if value is None:
        return [None]
    elif isinstance(value, list):
        return value
    else:
        return [value]


def build_plan_eval_args(
    app_name,
    folder,
    checkpoint,
    eval_cfg_paths,
    cfgs_model,
    cfgs_data,
    cfgs_data_aug,
    tag,
    evals_decode=True,
    sum_all_diffs=None,
    evals_obs=None,
    evals_alpha=None,
    eval_nodes=None,
    eval_tasks_per_node=None,
    eval_episodes=None,
    max_episode_steps=None,
    num_act_stepped=None,
    horizon=None,
    goal_H=None,
    num_elites=None,
    override_cfgs_data=True,
    override_datasets=True,
    wrapper_kwargs={},
    checkpoint_folder=None,
):
    """
    Builds evaluation arguments for online planning evaluations.

    Args:
        app_name (str): Name of the application.
        folder (str): Folder for logging outputs.
        eval_cfg_paths (list): List of evaluation configuration paths.
        cfgs_model (dict): Model configuration.
        cfgs_data (dict): Data configuration.
        tag (str): Tag for the evaluation.
        evals_alpha: Single value or list of alpha values.
        horizon: Single value or list of horizon values.
        num_act_stepped: Single value or list of num_act_stepped values.
        goal_H: Single value or list of goal_H values.
        num_elites: Single value or list of num_elites values.

        If any of these parameters is a list, a cartesian product is created
        across all list parameters and eval_cfg_paths. For example, with
        2 eval_cfg_paths and alpha=[0.1, 0.2] and horizon=[5, 10], this will
        generate 2 * 2 * 2 = 8 eval configs.

    Returns:
        tuple: (eval_nodes, eval_tasks_per_node, args_eval, cpus_per_task)
    """
    import copy
    import itertools

    # Convert sweep parameters to lists for cartesian product
    alpha_values = _to_list(evals_alpha)
    horizon_values = _to_list(horizon)
    num_act_stepped_values = _to_list(num_act_stepped)
    goal_H_values = _to_list(goal_H)
    num_elites_values = _to_list(num_elites)

    args_eval = []
    for eval_cfg_path in eval_cfg_paths:
        planning_cfg_template = load_yaml(eval_cfg_path)

        _nodes = planning_cfg_template.get("nodes", 1)
        _tasks = planning_cfg_template.get("tasks_per_node", 8)
        _cpus = planning_cfg_template.get("cpus_per_task", 12)
        eval_nodes = _nodes if eval_nodes is None else eval_nodes
        eval_tasks_per_node = _tasks if eval_tasks_per_node is None else eval_tasks_per_node

        # Cartesian product over all sweep parameters
        for alpha_val, horizon_val, nas_val, goal_H_val, num_elites_val in itertools.product(
            alpha_values, horizon_values, num_act_stepped_values, goal_H_values, num_elites_values
        ):
            planning_cfg = copy.deepcopy(planning_cfg_template)

            # Update the pretrain_kwargs field without overwriting it completely
            # Useful to keep ctxt_window or other args added in pretrain_kwargs
            planning_cfg["nodes"] = eval_nodes
            planning_cfg["tasks_per_node"] = eval_tasks_per_node
            model_kwargs = planning_cfg.get("model_kwargs", {})
            model_kwargs["module_name"] = f"app.{app_name}.modelcustom.simu_env_planning.vit_enc_preds"
            model_kwargs["checkpoint"] = checkpoint
            model_kwargs["pretrain_kwargs"].update(cfgs_model)  # Merge cfgs_model into pretrain_kwargs
            # take the needed keys from the planning cfg before overriding
            # Set to False for eval on Robocasa from DROID model
            if not override_datasets:  # take datasets from the planning_cfg, overwrite cfgs_data from training
                cfgs_data["custom"]["filter_tasks"] = planning_cfg["model_kwargs"]["data"]["custom"].get("filter_tasks")
                cfgs_data["datasets"] = planning_cfg["model_kwargs"]["data"].get("datasets", [])
                cfgs_data["validation"]["val_datasets"] = planning_cfg["model_kwargs"]["data"]["validation"].get(
                    "val_datasets", []
                )
                cfgs_data["custom"]["split_ratio"] = planning_cfg["model_kwargs"]["data"]["custom"].get(
                    "split_ratio", 1.0
                )
                cfgs_data["custom"]["custom_teleop_dset"] = planning_cfg["model_kwargs"]["data"]["custom"].get(
                    "custom_teleop_dset", None
                )
                cfgs_data["custom"]["frameskip"] = planning_cfg["model_kwargs"]["data"]["custom"].get("frameskip", 1)
                cfgs_data["custom"]["action_skip"] = planning_cfg["model_kwargs"]["data"]["custom"].get("action_skip", 1)
                cfgs_data["validation"]["val_dataset_camera_views"] = planning_cfg["model_kwargs"]["data"][
                    "validation"
                ].get("val_dataset_camera_views", None)
                cfgs_data["custom"]["num_hist"] = planning_cfg["model_kwargs"]["data"]["custom"].get("num_hist", 0)
                cfgs_data["custom"]["num_pred"] = planning_cfg["model_kwargs"]["data"]["custom"].get("num_pred", 0)
            if override_cfgs_data:  # always True
                model_kwargs["data"] = cfgs_data  # Override with cfgs_data to pretrain_kwargs
            model_kwargs["data"]["img_size"] = cfgs_data.get("img_size", 256)  # Ensure img_size is set in data
            model_kwargs["data_aug"] = cfgs_data_aug  # Override with cfgs_data_aug to pretrain_kwargs
            planning_cfg["model_kwargs"] = model_kwargs
            planning_cfg["folder"] = folder
            planning_cfg["checkpoint_folder"] = checkpoint_folder

            # To have one template working for both DINO and VJEPA WMs
            planning_cfg["task_specification"]["num_frames"] = cfgs_model["tubelet_size_enc"]
            planning_cfg["task_specification"]["num_proprios"] = cfgs_model["tubelet_size_enc"]

            planning_cfg["planner"]["decode_each_iteration"] = evals_decode
            if "img_size" in cfgs_data.keys():
                planning_cfg["task_specification"]["img_size"] = cfgs_data["img_size"]
            if evals_obs is not None:
                planning_cfg["task_specification"]["obs"] = evals_obs
            if alpha_val is not None:
                planning_cfg["planner"]["planning_objective"]["alpha"] = alpha_val

            pref_tag = f"online_{planning_cfg['tag']}_r{planning_cfg['task_specification']['img_size']}_alpha{planning_cfg['planner']['planning_objective']['alpha']}"
            if max_episode_steps is not None:
                planning_cfg["task_specification"]["max_episode_steps"] = max_episode_steps
                pref_tag = update_tag_pattern(pref_tag, "maxstp", max_episode_steps)
            if nas_val is not None:
                planning_cfg["planner"]["num_act_stepped"] = nas_val
                pref_tag = update_tag_pattern(pref_tag, "nas", nas_val)
            if sum_all_diffs is not None:
                planning_cfg["planner"]["planning_objective"]["sum_all_diffs"] = sum_all_diffs
                pref_tag = update_tag_pattern(pref_tag, "sum", sum_all_diffs, insert_before="ctxt")
            ctxt_window = wrapper_kwargs.get("ctxt_window")
            if ctxt_window is not None:
                planning_cfg["model_kwargs"]["wrapper_kwargs"]["ctxt_window"] = ctxt_window
                pref_tag = update_tag_pattern(pref_tag, "ctxt", ctxt_window)
            for k, v in wrapper_kwargs.items():
                if k != "ctxt_window":
                    planning_cfg["model_kwargs"]["wrapper_kwargs"][k] = v
            if horizon_val is not None:
                planning_cfg["planner"]["horizon"] = horizon_val
                pref_tag = update_tag_pattern(pref_tag, "H", horizon_val)
            if num_elites_val is not None:
                planning_cfg["planner"]["num_elites"] = num_elites_val
            if goal_H_val is not None:
                planning_cfg["task_specification"]["goal_H"] = goal_H_val
                pref_tag = update_tag_pattern(pref_tag, "gH", goal_H_val)
            if eval_episodes is not None:
                planning_cfg["meta"]["eval_episodes"] = eval_episodes
                pref_tag = update_tag_pattern(pref_tag, "ep", eval_episodes)
            if planning_cfg["planner"]["decode_each_iteration"]:
                pref_tag += "_decode"
            planning_cfg["tag"] = f"{pref_tag}/{tag}"
            args_eval.append(planning_cfg)

    return eval_nodes, eval_tasks_per_node, args_eval, _cpus


def fetch_checkpoint(source, device="cpu"):
    """Fetch checkpoint data from either a URL or local file path.

    Args:
        source (str or Path): Either a URL (starting with 'http://' or 'https://') or a local file path.
        device (str or torch.device): Device to map tensors to. Default is 'cpu'.

    Returns:
        dict: Checkpoint data containing model state dicts and metadata.

    Raises:
        Exception: If checkpoint cannot be loaded from the source.
    """
    from torch.hub import load_state_dict_from_url

    if isinstance(source, str) and source.startswith(("http://", "https://")):
        logger.info(f"Downloading checkpoint from URL: {source}")
        checkpoint = load_state_dict_from_url(source, map_location=torch.device(device), progress=True)
    else:
        logger.info(f"Loading checkpoint from local path: {source}")
        try:
            checkpoint = torch.load(source, map_location=torch.device(device))
        except Exception as e:
            logger.info(f"Encountered exception when loading checkpoint: {e}")
            raise

    return checkpoint


def load_checkpoint_state_dict(
    checkpoint,
    predictor=None,
    action_encoder=None,
    proprio_encoder=None,
    opt=None,
    scaler=None,
    load_act_enc=True,
    load_prop_enc=True,
    load_opt_scale_epoch=True,
):
    """Load state dicts from checkpoint data onto model modules.

    This function only loads weights from the checkpoint dict itself (predictor, action_encoder,
    proprio_encoder, optimizer, scaler). For loading heads from separate files, use load_heads_checkpoint().

    Args:
        checkpoint (dict): Checkpoint data containing state dicts (from fetch_checkpoint).
        predictor: Predictor module to load weights into.
        action_encoder: Action encoder module to load weights into.
        proprio_encoder: Proprio encoder module to load weights into.
        opt: Optimizer to load state into.
        scaler: GradScaler to load state into.
        load_act_enc (bool): Whether to load action encoder weights.
        load_prop_enc (bool): Whether to load proprio encoder weights.
        load_opt_scale_epoch (bool): Whether to load optimizer and scaler state.

    Returns:
        tuple: (predictor, action_encoder, proprio_encoder, opt, scaler, epoch)
    """
    epoch = checkpoint.get("epoch", -1)

    # -- loading predictor
    if predictor is not None and checkpoint.get("predictor") is not None:
        pretrained_dict = clean_state_dict(checkpoint["predictor"])

        # Handle naming convention differences for backward compatibility
        key_mapping = {
            "state_encoder.weight": "proprio_encoder.weight",
            "state_encoder.bias": "proprio_encoder.bias",
        }
        pretrained_dict = {key_mapping.get(k, k): v for k, v in pretrained_dict.items()}
        msg = predictor.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained predictor from epoch {epoch} with msg: {msg}")

    # -- loading action encoder
    if load_act_enc and action_encoder and checkpoint.get("action_encoder") is not None:
        pretrained_dict = clean_state_dict(checkpoint["action_encoder"])
        msg = action_encoder.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained action encoder from epoch {epoch} with msg: {msg}")

    # -- loading proprio encoder
    if load_prop_enc and proprio_encoder and checkpoint.get("proprio_encoder") is not None:
        pretrained_dict = clean_state_dict(checkpoint["proprio_encoder"])
        msg = proprio_encoder.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained proprio encoder from epoch {epoch} with msg: {msg}")

    # -- loading optimizer
    if load_opt_scale_epoch and opt is not None:
        try:
            opt.load_state_dict(checkpoint["opt"])
            logger.info(f"loaded optimizers from epoch {epoch}")
        except KeyError:
            logger.warning("Optimizer state not found in checkpoint, skipping optimizer load.")
        except ValueError as e:
            logger.warning(
                f"Failed to load optimizer state due to parameter group mismatch: {e}\n"
                f"This is likely due to model architecture changes (e.g., removed extrinsics_encoder). "
                f"Skipping optimizer state load. Training will resume with fresh optimizer state."
            )
        if scaler is not None:
            try:
                scaler.load_state_dict(checkpoint["scaler"])
                logger.info(f"loaded scaler from epoch {epoch}")
            except KeyError:
                logger.warning("Scaler state not found in checkpoint, skipping scaler load.")

    return (
        predictor,
        action_encoder,
        proprio_encoder,
        opt,
        scaler,
        epoch,
    )


def load_heads_checkpoint(heads, heads_base_path):
    """Load head checkpoints from separate files.

    Head checkpoints are stored in files named: {heads_base_path}_<head_name>.pth.tar

    Args:
        heads (dict): Dictionary of head modules to load weights into.
        heads_base_path (str or Path): Base path for head checkpoint files.
            Head files are expected at: {heads_base_path}_<head_name>.pth.tar
            (e.g., if heads_base_path="/path/to/checkpoint.pth.tar" and head name is "image_head",
            will load from "/path/to/checkpoint_image_head.pth.tar")

    Returns:
        tuple: (heads, head_epoch) where head_epoch is the epoch from the last loaded head.
    """
    if not heads:
        return heads, 0

    head_epoch = 0
    for name, head in heads.items():
        head_path = str(heads_base_path).removesuffix(".pth.tar") + "_" + name + ".pth.tar"
        head_epoch = head.load_checkpoint(head_path)
        logger.info(f"loaded pretrained head named {name} from epoch {head_epoch}")

    return heads, head_epoch


def load_checkpoint(
    r_path,
    predictor,
    action_encoder,
    proprio_encoder,
    heads,
    opt,
    scaler,
    load_act_enc=True,
    load_prop_enc=True,
    load_heads=True,
    load_opt_scale_epoch=True,
    load_stats=True,
    train_predictor=True,
    train_heads=False,
):
    """Load checkpoint from local file path and apply to model modules.

    Convenience wrapper that combines fetch_checkpoint(), load_checkpoint_state_dict(),
    and optionally load_heads_checkpoint().

    Args:
        r_path (str or Path): Local file path to checkpoint.
        predictor: Predictor module to load weights into.
        action_encoder: Action encoder module to load weights into.
        proprio_encoder: Proprio encoder module to load weights into.
        heads (dict): Dictionary of head modules to load weights into.
        opt: Optimizer to load state into.
        scaler: GradScaler to load state into.
        load_act_enc (bool): Whether to load action encoder weights.
        load_prop_enc (bool): Whether to load proprio encoder weights.
        load_heads (bool): Whether to load head weights from separate files.
        load_opt_scale_epoch (bool): Whether to load optimizer and scaler state.
        load_stats (bool): Unused, kept for backward compatibility.
        train_predictor (bool): If True, return predictor epoch; used for epoch tracking.
        train_heads (bool): If True, return head epoch instead of predictor epoch.

    Returns:
        tuple: (predictor, action_encoder, proprio_encoder, heads, opt, scaler, epoch)
    """
    checkpoint = fetch_checkpoint(r_path, device="cpu")

    (
        predictor,
        action_encoder,
        proprio_encoder,
        opt,
        scaler,
        epoch,
    ) = load_checkpoint_state_dict(
        checkpoint=checkpoint,
        predictor=predictor,
        action_encoder=action_encoder,
        proprio_encoder=proprio_encoder,
        opt=opt,
        scaler=scaler,
        load_act_enc=load_act_enc,
        load_prop_enc=load_prop_enc,
        load_opt_scale_epoch=load_opt_scale_epoch,
    )

    # Load heads from separate files if requested
    if load_heads and heads:
        heads, head_epoch = load_heads_checkpoint(heads, heads_base_path=r_path)
        if train_heads:
            epoch = head_epoch

    logger.info(f"read-path: {r_path}")
    del checkpoint

    return (
        predictor,
        action_encoder,
        proprio_encoder,
        heads,
        opt,
        scaler,
        epoch,
    )


def init_video_model(
    device,
    # Image and frame parameters
    img_size=256,
    num_frames_pred=8,
    # Encoder configuration
    enc_type="vjepa",
    enc_version="v1",
    enc_name="vit_large",
    pretrain_enc_path=None,
    pretrain_enc_ckpt_key="target_encoder",
    enc_use_rope=False,
    use_sdpa_enc=True,
    uniform_power=True,
    num_frames_enc=16,
    # Predictor architecture
    pred_type="dino_wm",
    pred_depth=6,
    pred_embed_dim=384,
    embed_dim=1024,
    pred_num_heads=16,
    pred_use_extrinsics=False,
    tubelet_size=1,
    use_rope=False,
    use_SiLU=False,
    use_sdpa=False,
    cfgs_attn_pattern=None,
    use_activation_checkpointing=False,
    init_scale_factor_adaln=10,
    # Action configuration
    action_dim=7,
    action_conditioning="token",
    action_tokens=1,
    action_emb_dim=0,
    action_encoder_inpred=False,
    act_mlp=False,
    use_action=True,
    # Proprioception configuration
    proprio_dim=7,
    proprio_encoding="feature",
    proprio_tokens=1,
    proprio_emb_dim=0,
    proprio_encoder_inpred=False,
    prop_mlp=False,
    use_proprio=True,
    # discard other kwargs
    **kwargs,
):

    local_window_time = -1
    local_window_h = -1
    local_window_w = -1
    if cfgs_attn_pattern is not None:
        local_window_time = cfgs_attn_pattern.get("local_window_time", -1)
        local_window_h = cfgs_attn_pattern.get("local_window_h", -1)
        local_window_w = cfgs_attn_pattern.get("local_window_w", -1)
    local_window = (local_window_time, local_window_h, local_window_w)
    if use_action:
        if action_conditioning == "token":
            assert action_tokens > 0
        elif action_conditioning == "feature":
            assert action_emb_dim > 0
    if use_proprio:
        if proprio_encoding == "token":
            assert proprio_tokens > 0
        elif proprio_encoding == "feature":
            assert proprio_emb_dim > 0
    if enc_type == "vjepa":
        if enc_version == "v1_open":
            import src.models.vision_transformer as vit_open

            # No option to make this encoder causal, it is used as a frame encoder (batchify video)
            # for planning, so no need for causality
            encoder = vit_open.__dict__[enc_name](
                num_frames=num_frames_enc,
                img_size=img_size,
                tubelet_size=2,
                uniform_power=uniform_power,
                use_sdpa=use_sdpa_enc,
                use_activation_checkpointing=False,
            ).to(device)
            for p in encoder.parameters():
                p.requires_grad = False
            encoder = encoder.eval()
            state_dict = torch.load(pretrain_enc_path, map_location="cpu")
            try:
                epoch = state_dict["epoch"]
            except:
                logger.info("No epoch in checkpoint, setting default epoch -1")
                epoch = -1
            state_dict[pretrain_enc_ckpt_key] = {
                k.replace("module.backbone.", ""): v for k, v in state_dict[pretrain_enc_ckpt_key].items()
            }
            msg = encoder.load_state_dict(state_dict[pretrain_enc_ckpt_key], strict=False)
            logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")
        elif enc_version == "v2_open":
            # Here, there is an option to make this encoder causal. If VideoWM.batchify_video==True, having or not causal mask
            # should not change anything, as we just truncate the mask to 1 frame embedding.
            encoder = vit_v2_open.__dict__[enc_name](
                num_frames=num_frames_enc,
                img_size=img_size,
                tubelet_size=2,
                uniform_power=uniform_power,
                use_sdpa=use_sdpa_enc,
                use_activation_checkpointing=False,
                use_rope=enc_use_rope,
                use_silu=False,
                wide_silu=True,
                local_window=local_window,
            ).to(device)
            for p in encoder.parameters():
                p.requires_grad = False
            encoder = encoder.eval()
            state_dict = torch.load(pretrain_enc_path, map_location="cpu")
            try:
                epoch = state_dict["epoch"]
            except:
                logger.info("No epoch in checkpoint, setting default epoch -1")
                epoch = -1
            state_dict[pretrain_enc_ckpt_key] = {
                k.replace("module.backbone.", ""): v for k, v in state_dict[pretrain_enc_ckpt_key].items()
            }
            msg = encoder.load_state_dict(state_dict[pretrain_enc_ckpt_key], strict=False)
            logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")
    elif enc_type == "dino":
        encoder = DinoEncoder(
            name=enc_version,
            feature_key="x_norm_patchtokens",
        ).to(device)
        for p in encoder.parameters():
            p.requires_grad = False
        encoder = encoder.eval()
    logger.info(f"Encoder: {encoder}")
    assert (
        img_size % encoder.patch_size == 0
    ), f"Image size {img_size} should be divisible by encoder patch size {encoder.patch_size}"

    if pred_type == "none":
        # No predictor, action encoder, or proprio encoder
        logger.info("Using no predictor (pred_type=none)")
        predictor = None
        action_encoder = None
        proprio_encoder = None
        return predictor, encoder, action_encoder, proprio_encoder
    elif pred_type == "dino_wm":
        # CAREFUL: num_patches defined here is essential: it determines both the block size of the causal attn mask
        # and the positional embedding
        # only works with action_conditioning == 'feature', proprio_encoding == 'feature' and action_encoder_inpred == False
        assert action_encoder_inpred == False
        assert proprio_encoder_inpred == False
        assert action_conditioning == "feature" and proprio_encoding == "feature"
        logger.info(f"Using DINO WM predictor with num_patches {int(img_size / encoder.patch_size) ** 2}")
        concat_dim = 1
        predictor = ViTPredictor(
            depth=pred_depth,
            heads=pred_num_heads,
            mlp_dim=2048,
            dropout=0.1,
            num_patches=int(img_size / encoder.patch_size) ** 2,
            num_frames=num_frames_pred,
            dim=pred_embed_dim + (proprio_emb_dim * 1 + action_emb_dim * 1) * (concat_dim),
            use_sdpa=use_sdpa,
        ).to(device)
    elif pred_type == "vjepa2_ac":
        # works with both action_encoder_inpred False or True, with action_conditioning in [’feature’, ‘token’],
        # and with proprio_encoding in [’feature’, ‘token’]. The action_conditioning decides of the proprio_encoding
        assert action_conditioning == proprio_encoding or proprio_encoding == None
        predictor = vit_ac_pred.__dict__["vit_ac_predictor"](
            img_size=img_size,
            patch_size=encoder.patch_size,
            num_frames=num_frames_pred,
            tubelet_size=tubelet_size,
            uniform_power=uniform_power,
            embed_dim=embed_dim,
            predictor_embed_dim=pred_embed_dim,
            action_dim=action_dim,
            proprio_dim=proprio_dim,
            action_emb_dim=action_emb_dim,
            proprio_emb_dim=proprio_emb_dim,
            proprio_encoder_inpred=proprio_encoder_inpred,
            action_encoder_inpred=action_encoder_inpred,
            action_conditioning=action_conditioning,
            proprio_tokens=proprio_tokens,
            depth=pred_depth,
            is_frame_causal=True,
            num_heads=encoder.num_heads if pred_num_heads is None else pred_num_heads,
            use_rope=use_rope,
            use_sdpa=True,
            use_silu=use_SiLU,
            wide_silu=True,
            use_extrinsics=pred_use_extrinsics,
            use_activation_checkpointing=use_activation_checkpointing,
        ).to(device)
    elif pred_type == "AdaLN":
        # works with action_encoder_inpred False or True, with proprio_encoding in [’feature’, ‘token’]
        assert action_conditioning == "token"
        assert proprio_encoder_inpred == False
        predictor = vit_predictor_AdaLN(
            img_size=img_size,
            patch_size=encoder.patch_size,
            num_frames=num_frames_pred,
            tubelet_size=tubelet_size,
            embed_dim=embed_dim,
            predictor_embed_dim=pred_embed_dim,
            depth=pred_depth,
            num_heads=pred_num_heads,
            use_silu=use_SiLU,
            use_rope=use_rope,
            local_window=local_window,
            use_activation_checkpointing=use_activation_checkpointing,
            action_dim=action_dim,
            proprio_dim=proprio_dim,
            act_mlp=act_mlp,
            prop_mlp=prop_mlp,
            proprio_encoder_inpred=proprio_encoder_inpred,
            action_encoder_inpred=action_encoder_inpred,
            proprio_encoding=proprio_encoding,
            proprio_emb_dim=proprio_emb_dim,
            proprio_tokens=proprio_tokens,
            init_scale_factor_adaln=init_scale_factor_adaln,
        ).to(device)
    enc_params = sum(p.numel() for p in encoder.parameters())
    pred_params = sum(p.numel() for p in predictor.parameters())
    logger.info(f"🧠 Encoder: {type(encoder).__name__} ({enc_params:,} params, frozen={not any(p.requires_grad for p in encoder.parameters())})")
    logger.info(f"🔮 Predictor: {type(predictor).__name__} ({pred_params:,} params)")
    if (action_tokens > 0 or action_emb_dim > 0) and not action_encoder_inpred:
        # Determine the correct output dimension for the action encoder
        if action_conditioning == "token":
            action_encoder_output_dim = predictor.predictor_total_embed_dim if pred_type == "AdaLN" else embed_dim
        elif action_conditioning == "feature":
            action_encoder_output_dim = action_emb_dim
        else:
            raise ValueError(f"Unknown action_conditioning: {action_conditioning}")
        action_encoder = ProprioceptiveEmbedding(
            num_frames=num_frames_pred,
            tubelet_size=1,
            tokens_per_step=action_tokens if action_conditioning == "token" else 1,
            in_chans=action_dim,
            embed_dim=action_encoder_output_dim,
            shift_input=False,
            mlp_dims=[action_encoder_output_dim],
            use_mlp=act_mlp,
        ).to(device)
        logger.info(f"Action encoder output dim: {action_encoder_output_dim}")
    else:
        action_encoder = None
    # Proprioceptive encoder should always be outside of the predictor, so that VideoWM.encode_proprio() really outputs a feature
    if not proprio_encoder_inpred and (proprio_tokens > 0 or proprio_emb_dim > 0):
        proprio_encoder = ProprioceptiveEmbedding(
            num_frames=num_frames_pred,
            tubelet_size=1,
            tokens_per_step=proprio_tokens if proprio_encoding == "token" else 1,
            in_chans=proprio_dim,
            embed_dim=embed_dim if proprio_encoding == "token" else proprio_emb_dim,
            shift_input=False,
            mlp_dims=[embed_dim],
            use_mlp=prop_mlp,
        ).to(device)
    else:
        proprio_encoder = None
    return predictor, encoder, action_encoder, proprio_encoder


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.constant_(m.bias, 0)
        torch.nn.init.constant_(m.weight, 1.0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_opt(
    predictor,
    action_encoder,
    proprio_encoder=None,
    encoder=None,
    iterations_per_epoch=1000,
    start_lr=0.0,
    ref_lr=1e-3,
    warmup=2,
    num_epochs=90,
    freeze_encoder=True,
    use_radamw=False,
    weight_decay=1e-6,
    final_weight_decay=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    ipe_scale=1.25,
    betas=(0.9, 0.999),
    eps=1e-8,
    use_wsd_schedule=False,
    anneal_steps=None,
    **kwargs,
):
    """
    Initialize optimizer and learning rate scheduler.

    Args:
        use_wsd_schedule: If True, use WSDSchedule (Warmup-Stable-Decay) instead of WarmupCosineSchedule.
        anneal_steps: Number of steps for the decay phase in WSDSchedule. If None, defaults to
                      warmup_steps to be backwards compatible (approximately similar behavior).
                      Only used when use_wsd_schedule=True.
    """
    param_groups = []
    param_groups += [
        {
            "params": (
                p
                for n, p in predictor.named_parameters()
                if ("bias" not in n) and (len(p.shape) != 1) and p.requires_grad
            )
        },
    ]
    param_groups += [
        {
            "params": (
                p for n, p in predictor.named_parameters() if ("bias" in n) or (len(p.shape) == 1) and p.requires_grad
            ),
            "WD_exclude": True,
            "weight_decay": 0,
        },
    ]
    if action_encoder is not None:
        param_groups += [
            {
                "params": (
                    p
                    for n, p in action_encoder.named_parameters()
                    if ("bias" not in n) and (len(p.shape) != 1) and p.requires_grad
                )
            }
        ]
        param_groups += [
            {
                "params": (
                    p
                    for n, p in action_encoder.named_parameters()
                    if ("bias" in n) or (len(p.shape) == 1) and p.requires_grad
                ),
                "WD_exclude": True,
                "weight_decay": 0,
            },
        ]
    if proprio_encoder is not None:
        param_groups += [
            {
                "params": (
                    p
                    for n, p in proprio_encoder.named_parameters()
                    if ("bias" not in n) and (len(p.shape) != 1) and p.requires_grad
                )
            }
        ]
        param_groups += [
            {
                "params": (
                    p
                    for n, p in proprio_encoder.named_parameters()
                    if ("bias" in n) or (len(p.shape) == 1) and p.requires_grad
                ),
                "WD_exclude": True,
                "weight_decay": 0,
            },
        ]
    if encoder is not None and not freeze_encoder:
        param_groups += [
            {
                "params": (
                    p
                    for n, p in encoder.named_parameters()
                    if ("bias" not in n) and (len(p.shape) != 1) and p.requires_grad
                )
            }
        ]
        param_groups += [
            {
                "params": (
                    p
                    for n, p in encoder.named_parameters()
                    if ("bias" in n) or (len(p.shape) == 1) and p.requires_grad
                ),
                "WD_exclude": True,
                "weight_decay": 0,
            },
        ]

    if use_radamw:
        logger.info("Using Rescaled-AdamW")
        optimizer = RAdamW(param_groups, betas=betas, eps=eps)
    else:
        logger.info("Using AdamW")
        optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)

    warmup_steps = int(warmup * iterations_per_epoch)
    T_max = int(ipe_scale * num_epochs * iterations_per_epoch)
    if use_wsd_schedule:
        # Use WSDSchedule (Warmup-Stable-Decay)
        # Default anneal_steps to warmup_steps
        if anneal_steps is None:
            anneal_steps = warmup_steps
        logger.info(f"Using WSDSchedule with warmup_steps={warmup_steps}, anneal_steps={anneal_steps}, T_max={T_max}")
        scheduler = WSDSchedule(
            optimizer,
            warmup_steps=warmup_steps,
            anneal_steps=anneal_steps,
            T_max=T_max,
            start_lr=start_lr,
            ref_lr=ref_lr,
            final_lr=final_lr,
        )
    else:
        # Use WarmupCosineSchedule (default for backwards compatibility)
        logger.info(f"Using WarmupCosineSchedule with warmup_steps={warmup_steps}, T_max={T_max}")
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=warmup_steps,
            start_lr=start_lr,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=T_max,
        )

    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=weight_decay,
        final_wd=final_weight_decay,
        T_max=T_max,
    )
    scaler = torch.amp.GradScaler("cuda") if mixed_precision else None
    return optimizer, scaler, scheduler, wd_scheduler
