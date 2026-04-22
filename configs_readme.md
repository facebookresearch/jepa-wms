# Configuration Reference — JEPA World Model Evaluation

This document describes every parameter in the evaluation config YAML files (e.g., `configs/evals/simu_env_planning/pt/jepa-wm/*.yaml`). These configs drive the **simulated-environment goal-conditioned planning** evaluation pipeline.

---

## Table of Contents

- [Top-level / SLURM Launch Parameters](#top-level--slurm-launch-parameters)
- [meta](#meta)
- [distributed](#distributed)
- [logging](#logging)
- [model_kwargs](#model_kwargs)
  - [model_kwargs.data](#model_kwargsddata)
  - [model_kwargs.data_aug](#model_kwargsddata_aug)
  - [model_kwargs.pretrain_kwargs](#model_kwargspretrain_kwargs)
    - [pretrain_kwargs.attn](#pretrain_kwargsattn)
    - [pretrain_kwargs.heads_cfg](#pretrain_kwargsheads_cfg)
    - [pretrain_kwargs.predictor](#pretrain_kwargspredictor)
    - [pretrain_kwargs.action_encoder](#pretrain_kwargsaction_encoder)
    - [pretrain_kwargs.proprio_encoder](#pretrain_kwargsproprio_encoder)
    - [pretrain_kwargs.visual_encoder](#pretrain_kwargsvisual_encoder)
    - [pretrain_kwargs.rollout_cfg](#pretrain_kwargsrollout_cfg)
    - [pretrain_kwargs.wm_encoding](#pretrain_kwargswm_encoding)
  - [model_kwargs.wrapper_kwargs](#model_kwargswrapper_kwargs)
- [planner](#planner)
  - [planner.planning_objective](#plannerplanning_objective)
- [task_specification](#task_specification)
  - [task_specification.env](#task_specificationenv)
- [use_fsdp](#use_fsdp)

---

## Top-level / SLURM Launch Parameters

These parameters configure the SLURM job submission via [submitit](https://github.com/facebookincubator/submitit). They are consumed by `evals/main_distributed.py`.

| Parameter | Type | Example | Description |
|---|---|---|---|
| `nodes` | `int` | `1` | Number of SLURM nodes to allocate. Passed to the submitit executor as `nodes=`. |
| `tasks_per_node` | `int` | `8` | Number of GPU tasks per node. Also sets `gpus_per_node` in submitit. Equals the number of GPUs used per node. |
| `cpus_per_task` | `int` | `16` | Number of CPU cores allocated per GPU task. Default fallback in code: `32`. |
| `mem_per_gpu` | `str` | `"32G"` | Memory allocation per GPU for the SLURM job. Default fallback: `"220G"`. |
| `copy_code` | `bool` | `false` | If `true`, copies the entire code directory to the output folder before launching (for reproducibility). |
| `folder` | `str` | `/home/.../jepa-wms` | Root directory path. Used to construct output paths: `<folder>/simu_env_planning/<tag>/`. |
| `tag` | `str` | `online_gc_zeroshot/...` | Sub-path appended to `<folder>/simu_env_planning/` to form the working directory for all outputs (videos, CSVs, plots). |
| `eval_name` | `str` | `simu_env_planning` | Determines which eval module to invoke. `evals/scaffold.py` uses this to `importlib.import_module(f"evals.{eval_name}.eval")` and call its `main()`. |

---

## `meta`

Global meta-parameters for the evaluation run. Read in `evals/simu_env_planning/eval.py`.

| Parameter | Type | Example | Description |
|---|---|---|---|
| `quick_debug` | `bool` | `false` | Debug mode override. When `true`, forces `eval_episodes=1`, `iterations=2`, `num_samples=2`, `num_elites=2`, and `tqdm_silent=false` for rapid iteration. |
| `seed` | `int` | `1` | Global random seed for `random`, `numpy`, `torch`, and `torch.cuda`. Also used to derive per-rank local seeds and per-episode seeds. |
| `eval_episodes` | `int` | `96` | Number of evaluation episodes **per task**. Total episodes = `eval_episodes × len(tasks)`, distributed across GPUs when `distribute_multitask_eval=true`. |

---

## `distributed`

Controls multi-GPU episode distribution. Read in `evals/simu_env_planning/eval.py`.

| Parameter | Type | Example | Description |
|---|---|---|---|
| `distribute_multitask_eval` | `bool` | `true` | When `true`, evaluation episodes are partitioned across GPUs (each rank gets a disjoint subset). When `false`, all ranks evaluate all episodes but only rank 0 logs results. |
| `local_rng_samplers` | `bool` | `true` | When `true`, each rank has its own local `torch.Generator` seeded with `local_seed` (passed to the planner) but does **not** overwrite global RNGs. When `false`, calls `set_seed(local_seed)` globally, ensuring fully independent per-rank randomness. |
| `seed_shift` | `str\|int` | `"horizon_1000"` | Determines the seed offset between ranks. `"horizon_1000"` → shift = `horizon × 1000` (e.g., `6000`). Can also be a numeric value. Each rank's local seed = `meta.seed + rank × seed_shift`. |

---

## `logging`

Controls output and visualization logging. Used in the `Logger` class (`evals/simu_env_planning/planning/common/gc_logger.py`) and the evaluation loop.

| Parameter | Type | Example | Description |
|---|---|---|---|
| `exp_name` | `str` | `"gc_zeroshot_dist"` | Experiment name, printed during initialization. Used for display purposes. |
| `save_csv` | `bool` | `true` | When `true`, writes aggregated eval metrics (reward, success rate, distances, etc.) to `eval.csv` in the working directory. Per-task CSVs are also created for multi-task evals. |
| `tqdm_silent` | `bool` | `false` | When `true`, suppresses tqdm progress bars during episode rollouts. Passed as `disable=` to `tqdm()`. |
| `optional_plots` | `bool` | `true` | When `true`, generates extra visualizations: init/goal frame PDFs, expert/agent videos, decoded plan frames over CEM iterations, action comparison plots, loss plots, and distance analyses. |

---

## `model_kwargs`

Top-level model configuration section. Passed to the model's `init_module()` function.

| Parameter | Type | Example | Description |
|---|---|---|---|
| `module_name` | `str` | `app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds` | Python module path for model initialization. Must contain an `init_module()` function. Loaded via `importlib.import_module(module_name)`. |
| `checkpoint` | `str` | `/path/to/jepa_wm_pusht.pth.tar` | Path to the pretrained model checkpoint (or a URL). Loaded by `fetch_checkpoint()` and used to initialize predictor, action encoder, and proprio encoder weights. |

### `model_kwargs.data`

Data configuration used to build the evaluation dataset via `make_datasets()`.

| Parameter | Type | Example | Description |
|---|---|---|---|
| `img_size` | `int` | `224` | Image resolution (height = width) for rendering and model input. Used for transforms and encoder grid computation (`img_size / patch_size`). |
| `datasets` | `list[str]` | `["PushT"]` | List of dataset names. Resolved to file paths via `get_dataset_paths()`. The validation trajectory dataset provides goal states (when `goal_source="dset"`) and action normalization statistics (`action_mean`, `action_std`). |
| `loader.batch_size` | `int` | `8` | DataLoader batch size. Primarily a training parameter carried over for dataset construction; during online planning eval, the dataset is not batched by a loader. |
| `custom.frameskip` | `int` | `5` | Number of simulator steps per world-model forward pass. Defines temporal resolution: one WM prediction corresponds to `frameskip` low-level environment steps. Used to compute `action_ratio`, `max_episode_steps = frameskip × goal_H`, and model action dimension. |

### `model_kwargs.data_aug`

| Parameter | Type | Example | Description |
|---|---|---|---|
| `normalize` | `list[list[float]]` | `[[0.485,0.456,0.406],[0.229,0.224,0.225]]` | ImageNet-standard normalization `[mean, std]` per RGB channel. Applied as a `torchvision.transforms.Normalize` to images before encoding; its inverse is used for decoding/visualization. |

### `model_kwargs.pretrain_kwargs`

Architecture and training configuration for the JEPA world model. Passed to `init_video_model()` in `app/vjepa_wm/utils.py`.

| Parameter | Type | Example | Description |
|---|---|---|---|
| `grid_size` | `int` | `16` | Spatial grid dimension of the encoder output. For DINOv2 ViT-S/14 with 224px input: `224 / 14 = 16`. Each frame is represented as a `(grid_size × grid_size)` grid of patch tokens. |
| `tubelet_size_enc` | `int` | `1` | Number of consecutive frames grouped into one temporal token by the encoder. With `1`, each frame is encoded independently. Affects action/proprio dimension: `model_action_dim = action_dim × tubelet_size_enc × frameskip / action_skip`. |
| `use_activation_checkpointing` | `bool` | `false` | Whether to use gradient checkpointing in the predictor to save VRAM at the cost of compute. Forced to `false` during eval. |
| `action_conditioning` | `str` | `"token"` | How actions are injected into the predictor. `"token"`: actions encoded as separate tokens appended to the sequence. `"feature"`: action embeddings are broadcast to all spatial patches. |
| `proprio_encoding` | `str` | `"feature"` | How proprioception is injected into the predictor. `"token"`: as separate tokens. `"feature"`: broadcast to all `grid_size²` spatial patches via `repeat(..., f=grid_size²)`. |
| `num_frames_pred` | `int` | `4` | Maximum number of frames the predictor can handle in its positional embedding and causal attention mask. Sets the predictor's temporal capacity. |

#### `pretrain_kwargs.attn`

Attention pattern configuration for the predictor.

| Parameter | Type | Example | Description |
|---|---|---|---|
| `local_window_time` | `int` | `3` | Temporal window size for local attention. `-1` = full (global) attention in time. A value of `3` means each token can attend to 3 timesteps. |
| `local_window_h` | `int` | `-1` | Spatial height window for local attention. `-1` = full attention along H. |
| `local_window_w` | `int` | `-1` | Spatial width window for local attention. `-1` = full attention along W. |

#### `pretrain_kwargs.heads_cfg`

| Parameter | Type | Example | Description |
|---|---|---|---|
| `architectures` | `dict` | `{}` | Dictionary of decoder head configurations. Can define `"image_head"` (ViT image decoder for pixel reconstruction) and `"state_head"` (pose readout). Empty `{}` means no decoder heads — the model operates purely in latent space. |

#### `pretrain_kwargs.predictor`

Predictor (dynamics model) architecture parameters.

| Parameter | Type | Example | Description |
|---|---|---|---|
| `pred_type` | `str` | `"AdaLN"` | Predictor architecture. `"AdaLN"`: Adaptive Layer Norm predictor (actions modulate via AdaLN). `"dino_wm"`: DinoWM-style concat + causal ViT. `"vjepa2_ac"`: V-JEPA2 action-conditioned predictor. |
| `pred_depth` | `int` | `6` | Number of transformer blocks in the predictor. |
| `pred_embed_dim` | `int` | `384` | Embedding dimension of the predictor's transformer. |
| `pred_num_heads` | `int` | `16` | Number of attention heads in the predictor. |
| `use_rope` | `bool` | `true` | Whether to use Rotary Position Embeddings (RoPE) in the predictor. |
| `uniform_power` | `bool` | `true` | Whether RoPE uses uniform power distribution across dimensions. |
| `tubelet_size` | `int` | `1` | Tubelet size for the predictor's positional embeddings and token grouping. |
| `use_SiLU` | `bool` | `false` | Whether to use SiLU activation (instead of GELU) in the predictor's MLP blocks. |
| `pred_use_extrinsics` | `bool` | `false` | Whether the predictor uses extrinsic camera parameters. Only relevant for `pred_type="vjepa2_ac"`. |
| `act_pred_projector` | `bool` | `false` | Config-level flag for action prediction heads (primarily a training-time parameter). |

#### `pretrain_kwargs.action_encoder`

Configures how actions are encoded before being fed to the predictor.

| Parameter | Type | Example | Description |
|---|---|---|---|
| `action_tokens` | `int` | `1` | Number of tokens per timestep to represent actions. Used when `action_conditioning="token"`. Determines whether actions are used: `use_action = action_tokens > 0 or action_emb_dim > 0`. |
| `action_emb_dim` | `int` | `0` | Embedding dimension for action features when `action_conditioning="feature"`. When `0`, the "token" pathway is used instead. |
| `action_mlp` | `bool` | `false` | Whether the action encoder uses an MLP (non-linear) or simple linear projection. Passed as `use_mlp=` to `ProprioceptiveEmbedding`. |
| `action_encoder_inpred` | `bool` | `true` | When `true`, raw actions are passed directly into the predictor (which has its own internal action encoder). When `false`, actions are first encoded by an external `ProprioceptiveEmbedding` module before entering the predictor. |

#### `pretrain_kwargs.proprio_encoder`

Configures how proprioceptive state information is encoded.

| Parameter | Type | Example | Description |
|---|---|---|---|
| `proprio_tokens` | `int` | `0` | Number of tokens per timestep for proprioception. `0` means no proprio tokens, but `proprio_emb_dim > 0` can still enable proprio via the feature pathway. `use_proprio = proprio_tokens > 0 or proprio_emb_dim > 0`. |
| `proprio_emb_dim` | `int` | `16` | Embedding dimension for proprio features (when `proprio_encoding="feature"`). The feature is broadcast to all `grid_size²` spatial patches. |
| `prop_mlp` | `bool` | `false` | Whether the proprio encoder uses an MLP or linear projection. |
| `proprio_encoder_inpred` | `bool` | `false` | When `true`, raw proprio features are passed into the predictor directly. When `false`, proprio is first projected through an external `ProprioceptiveEmbedding` module. |

#### `pretrain_kwargs.visual_encoder`

Configures the visual (image/video) encoder backbone.

| Parameter | Type | Example | Description |
|---|---|---|---|
| `enc_type` | `str` | `"dino"` | Encoder type. `"dino"`: frozen DINOv2 image encoder (processes each frame independently). `"vjepa"`: V-JEPA video encoder (can process multiple frames as a video clip). |
| `enc_version` | `str` | `"dinov2_vits14"` | Specific encoder model name/version. For DINOv2, passed to `DinoEncoder(name=...)` and loaded from torch hub. For V-JEPA, options include `"v1_open"`, `"v2_open"`. |
| `embed_dim` | `int` | `384` | Encoder output embedding dimension. For ViT-S/14: 384. Sets the predictor input projection dimension. |
| `pretrain_enc_ckpt_key` | `str` | `"target_encoder"` | Key in the checkpoint dict containing the encoder weights. Only relevant for `enc_type="vjepa"`. |
| `pretrain_enc_path` | `str\|null` | `null` | Path to a separate pretrained encoder checkpoint. `null` means the encoder is loaded from `enc_version` (e.g., torch hub for DINO). Only needed for V-JEPA encoders. |
| `enc_name` | `str\|null` | `null` | Model architecture name (e.g., `"vit_large"`). Used only for V-JEPA encoders with `enc_version` in `["v1_open", "v2_open"]`. |
| `num_frames_enc` | `int\|null` | `null` | Number of frames the encoder expects. Only for V-JEPA encoders. For DINO with `batchify_video=true`, frames are processed individually, so this is irrelevant. |
| `enc_use_rope` | `bool\|null` | `null` | Whether the encoder uses RoPE. Only for V-JEPA v2_open encoders. |
| `use_sdpa_enc` | `bool\|null` | `null` | Whether the encoder uses PyTorch Scaled Dot-Product Attention (SDPA). Only for V-JEPA encoders. |

#### `pretrain_kwargs.rollout_cfg`

Training-time rollout configuration. Stored for reproducibility but **not actively used during evaluation** — the eval planner performs its own unrolling via `EncPredWM.unroll()`.

| Parameter | Type | Example | Description |
|---|---|---|---|
| `rollout_steps` | `int` | `2` | Number of autoregressive rollout steps during training. |
| `do_parallel_rollout` | `bool` | `false` | Whether to use parallel rollout mode during training (predict all steps simultaneously). |
| `do_sequential_rollout` | `bool` | `true` | Whether to use sequential (autoregressive) rollout mode during training. |
| `rollout_stop_gradient` | `bool` | `true` | Whether to detach features between rollout steps (prevents gradient flow through the full rollout chain). |
| `sampling_scheduler` | `dict` | `{type: linear, start: 0.0, end: 0.0}` | Scheduled sampling config for mixing ground-truth vs. predicted features during training. `start/end=0.0` means always use predicted features (no GT mixing). |
| `train_rollout_prefixes` | `str` | `"random"` | How to choose the starting timestep for rollouts during training. `"random"`: random prefix. `"first"`: always start from time 0. `"all"`: rollout from every timestep. |
| `ctxt_window_train_rollout` | `int` | `3` | Context window size used during training rollouts. |
| `prepend_gt` | `bool` | `false` | Whether to prepend ground-truth features in parallel rollout mode. |

#### `pretrain_kwargs.wm_encoding`

Controls how the world model processes encoded representations.

| Parameter | Type | Example | Description |
|---|---|---|---|
| `normalize_reps` | `bool` | `false` | When `true`, applies `F.layer_norm()` to visual embeddings after encoding and after prediction. |
| `dup_image` | `bool` | `false` | When `true`, duplicates each frame along the time dimension (for V-JEPA compatibility when `tubelet_size_enc=2` but input has single frames). |
| `batchify_video` | `bool` | `true` | When `true`, flattens the time dimension into the batch dimension before encoding, treating the encoder as a frame-by-frame image encoder. **Required `true` for `enc_type="dino"`**. |

### `model_kwargs.wrapper_kwargs`

Parameters for the `EncPredWM` wrapper around the base `VideoWM`.

| Parameter | Type | Example | Description |
|---|---|---|---|
| `ctxt_window` | `int` | `2` | Number of recent timesteps (visual + action + proprio features) used as context when autoregressively unrolling in `EncPredWM.unroll()`. At each rollout step, only the last `ctxt_window` features are fed to `forward_pred()`. Controls the trade-off between memory/compute and temporal context for predictions. |

---

## `planner`

Configures the planning optimizer that searches for optimal action sequences in the world model's latent space. Read in `evals/simu_env_planning/planning/gc_agent.py`.

| Parameter | Type | Example | Description |
|---|---|---|---|
| `planner_name` | `str` | `"cem"` | Planning algorithm. Options: `"cem"` (Cross-Entropy Method), `"mppi"` (Model Predictive Path Integral), `"nevergrad"` (Nevergrad black-box optimization), `"gd"` (gradient descent), `"adam"` (Adam optimizer). |
| `optimizer_name` | `str\|null` | `null` | Nevergrad optimizer variant (e.g., `"NgIohTuned"`, `"CMA"`, `"PSO"`). Only used when `planner_name="nevergrad"`. Default: `"NgIohTuned"`. |
| `horizon` | `int` | `6` | Planning horizon — number of future action steps to optimize over. Each CEM iteration samples action sequences of length `min(horizon, steps_left)`. |
| `num_samples` | `int` | `300` | Number of candidate action sequences sampled per CEM iteration. More samples = better exploration but slower. Must be ≥ `num_elites`. |
| `num_elites` | `int` | `10` | Number of top-performing samples used to update the CEM distribution at each iteration. Mean and std are computed from only these elite samples. |
| `iterations` | `int` | `30` | Number of CEM optimization iterations per planning call. Each iteration: sample → evaluate → select elites → update distribution. |
| `var_scale` | `float` | `1.0` | Initial standard deviation scale for the action sampling distribution. CEM initializes `std = var_scale × ones(...)`. Higher values = more exploration initially. |
| `num_act_stepped` | `int` | `6` | Number of action steps actually executed from the optimized plan before re-planning. The planner returns `mean[:num_act_stepped]`. With `num_act_stepped = horizon`, the full plan is executed each time. |
| `repeat_actskip` | `bool` | `false` | When `true`, each planned action is repeated `action_skip` times in the environment. When `false`, actions are not repeated and `action_ratio = frameskip / action_skip`. |
| `decode_each_iteration` | `bool` | `false` | When `true`, decodes predicted latent states to pixel space at every CEM iteration (requires an image decoder head in `heads_cfg`). Used for visualization: generates GIF animations of the plan evolving over iterations. |
| `distribute_planner` | `bool` | `false` | When `true`, distributes CEM sampling across multiple GPUs via `FullGatherLayer` — each GPU evaluates a subset of samples and results are gathered. Typically forced to `false` since episodes are distributed across ranks instead. |

### `planner.planning_objective`

Defines the cost function minimized by the planner. Implemented in `evals/simu_env_planning/planning/planning/objectives.py`.

| Parameter | Type | Example | Description |
|---|---|---|---|
| `objective_type` | `str` | `"L2"` | Distance metric for planning. `"L2"`: mean squared error between predicted and goal representations. `"L1"`: mean absolute error. `"repr_sim"`: negative cosine similarity. |
| `alpha` | `float` | `0.1` | Weight for the proprioceptive component of the planning loss. Total loss = `loss_visual + alpha × loss_proprio`. E.g., `0.1` means proprio distance contributes 10% relative to visual distance. Only applies when both visual and proprio encodings are present. |
| `sum_all_diffs` | `bool` | `false` | When `false`, only the **last** predicted timestep's distance to goal is used (standard goal-conditioned planning). When `true`, distances at **all** predicted timesteps are summed, encouraging the model to approach the goal at every step along the horizon. |

---

## `task_specification`

Configures the evaluation task and simulation environment. Read in `evals/simu_env_planning/eval.py` and `evals/simu_env_planning/envs/init.py`.

| Parameter | Type | Example | Description |
|---|---|---|---|
| `task` | `str` | `"pusht-base"` | Task identifier string. Prefixes map to environments: `"pusht-"` → PushT, `"mw-"` → MetaWorld, `"wall-"` → Wall, `"maze-"` → PointMaze, `"robocasa-"` → RoboCasa, `"droid-"` → DROID. Also checked against `TASK_SET` dict for multi-task support. |
| `obs` | `str` | `"rgb_state"` | Observation type. `"rgb"`: visual only. `"rgb_state"`: visual + proprioceptive. `"state"`: proprioceptive only. Controls how observations are prepared and whether `PixelWrapper` includes proprio data. |
| `obs_concat_channels` | `bool` | `false` | When `true`, multiple frames are concatenated along the channel dimension → shape `(num_frames×3, H, W)`. When `false`, frames are stacked as separate timesteps → shape `(num_frames, 3, H, W)`. |
| `goal_source` | `str` | `"dset"` | Where goal observations come from. `"dset"`: sampled from the training dataset. `"expert"`: generated by a scripted expert policy (MetaWorld only). `"random_state"`: randomly sampled environment states. `"random_action"`: dataset trajectory but with random noise actions. |
| `succ_def` | `str` | `"simu"` | Success criterion. `"simu"`: uses the simulator's built-in success signal (`infos["success"]`). For PushT with `goal_source="dset"`, uses `env.eval_state()` (position diff < 20, angle diff < $\pi/9$). |
| `done_at_succ` | `bool` | `false` | When `true`, the episode terminates immediately upon success. When `false`, the episode continues for the full `max_episode_steps`. |
| `goal_H` | `int` | `6` | Goal horizon — temporal distance (in WM steps) between the initial state and goal state. Used to compute `max_episode_steps = frameskip × goal_H`. Also determines the trajectory segment length sampled from the dataset: `frameskip × goal_H + 1`. |
| `num_frames` | `int` | `1` | Number of observation frames stacked in the visual observation. The `PixelWrapper` maintains a deque of `num_frames` rendered frames. With `1`, each observation is a single RGB image. |
| `num_proprios` | `int` | `1` | Number of proprioceptive measurements stacked. The `PixelWrapper` maintains a deque of `num_proprios` proprio vectors. With `1`, each observation has a single proprio vector. |
| `img_size` | `int` | `224` | Render resolution for the environment. `PixelWrapper` renders at `(img_size, img_size)` pixels. Must match the model's expected input size. |

### `task_specification.env`

Environment-specific parameters passed to the environment constructor.

| Parameter | Type | Example | Description |
|---|---|---|---|
| `with_target` | `bool` | `true` | **(PushT)** When `true`, renders the green target region (T-shape overlay) in the observation. |
| `with_velocity` | `bool` | `true` | **(PushT)** When `true`, the state vector includes agent velocities → 7D: `[agent_x, agent_y, T_x, T_y, angle, agent_vx, agent_vy]`. When `false`, state is 5D (no velocities). |
| `freeze_rand_vec` | `bool` | `false` | When `false`, each episode gets a different random initialization vector. Primarily relevant for MetaWorld environments where `rand_vec` controls task randomization. For PushT, randomization comes from the episode seed. |

---

## `use_fsdp`

| Parameter | Type | Example | Description |
|---|---|---|---|
| `use_fsdp` | `bool` | `false` | Whether to use Fully Sharded Data Parallel (FSDP) for model sharding across GPUs. Typically `false` during evaluation since the model is frozen and inference-only. |

---

## Environment Variables

Config YAML files use `${VAR_NAME}` placeholders that are expanded at load time by `src/utils/yaml_utils.py`. You can set these in your shell profile, or create a `.env` file in the repo root (gitignored, auto-loaded).

Copy `.env.template` to `.env` and fill in your paths:

```bash
cp .env.template .env
```

| Variable | Purpose |
|---|---|
| `JEPAWM_LOGS` | Root directory for training and eval logs/outputs |
| `JEPAWM_DSET` | Root directory for datasets (PushT, MetaWorld, etc.) |
| `JEPAWM_CKPT` | Root directory for model checkpoints |
| `JEPAWM_OSSCKPT` | Root directory for open-source pretrained encoder checkpoints (V-JEPA, DINOv2) |

---

## Config Organization

| Directory | Tracked | Purpose |
|---|---|---|
| `configs/vjepa_wm/` | Yes | Base training configs |
| `configs/evals/` | Yes | Base evaluation configs |
| `configs/online_plan_evals/` | Yes | Online planning eval configs |
| `configs/local/` | No (gitignored) | Personal experiment configs |
| `app/*/local/` | No (gitignored) | App-specific personal configs |
| `evals/*/local/` | No (gitignored) | Eval-specific personal configs |
| `*/dump_online_evals/*` | No (gitignored) | Auto-generated eval configs |

---

## Config-Driven Feature Selection

All experiment features coexist on one branch and are toggled via config:

| Feature | Config key | Default | Effect |
|---|---|---|---|
| Token Merging (ToMe) | `predictor.tome_r` | `0` (off) | Set >0 to merge tokens |
| CLS vs full-patch | `visual_encoder.feature_key` | `x_norm_patchtokens` | Set to `x_norm_clstoken` for CLS |
| Precomputed features | `visual_encoder.enc_type` | `dino` | Set to `precomputed` for cached features |
| Planner | `planner.planner_name` | `cem` | `cem`, `grasp`, `cem_gd`, `gd`, `adam`, `mppi`, `nevergrad` |
| Wandb (training) | `logging.wandb.project` | `vjepa_wm` | Wandb project for training runs |
| Wandb (eval) | `logging.wandb.project` | `jepa_wm_eval` | Wandb project for eval runs |

---

## Run Tracking

Every training and eval run generates a unique run ID (`YYYYMMDD_HHMMSS_hexhash`) and creates an isolated output directory:

```
${JEPAWM_LOGS}/experiment_name/
  runs/
    20260422_143052_a3f1b2/
      config_resolved.yaml    # Full expanded config snapshot
      run_meta.json           # Git commit, hostname, timestamp
      log_r0.csv              # Training metrics (training only)
      eval.csv                # Eval metrics
    latest -> runs/20260422_143052_a3f1b2
  jepa-latest.pth.tar         # Checkpoints (experiment-level, shared across runs)
```
