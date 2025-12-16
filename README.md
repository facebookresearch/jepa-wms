<h1 align="center">
    <p>🌍 <b>JEPA-WMs</b></p>
</h1>

<h2 align="center">
    <p><i>What drives success in physical planning with <br> Joint-Embedding Predictive World Models?</i></p>
</h2>

<div align="center" style="line-height: 1;">
  <a href="https://github.com/facebookresearch/jepa-wms" target="_blank" style="margin: 2px;"><img alt="Github" src="https://img.shields.io/badge/Github-facebookresearch/jepa--wms-black?logo=github" style="display: inline-block; vertical-align: middle;"/></a>
  <a href="https://huggingface.co/datasets/facebook/jepa-wms" target="_blank" style="margin: 2px;"><img alt="HuggingFace" src="https://img.shields.io/badge/🤗%20HuggingFace-facebook/jepa--wms-ffc107" style="display: inline-block; vertical-align: middle;"/></a>
  <a href="https://arxiv.org/abs/XXXX.XXXXX" target="_blank" style="margin: 2px;"><img alt="ArXiv" src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b5212f?logo=arxiv" style="display: inline-block; vertical-align: middle;"/></a>
</div>

<br>

<p align="center">
  <b><a href="https://ai.facebook.com/research/">Meta AI Research, FAIR</a></b>
</p>

<p align="center">
  <a href="https://x.com/BasileTerv987">Basile Terver</a>,
  Tsung-Yen Yang,
  Jean Ponce,
  Adrien Bardes,
  Yann LeCun
</p>

<p align="center">
  PyTorch implementation, data and pretrained models for <b>JEPA-WMs</b>.
</p>

<p align="center">
  <img src="assets/train_plan_schema.png" alt="JEPA-WMs diagram" width="800">
</p>

---

## 🎯 Pretrained Models

We provide pretrained [JEPA-WM](https://arxiv.org/abs/xxxx.XXXX), [DINO-WM](https://arxiv.org/abs/2411.04983) and [V-JEPA-2-AC(fixed)](https://arxiv.org/abs/2506.09985) baseline models for various environments.

### JEPA-WM Models

| Environment | Resolution | Encoder | Pred. Depth | Weights |
|-------------|------------|---------|-------------|---------|
| DROID & RoboCasa | 256×256 | DINOv3 ViT-S/16 | 6 | [download](https://dl.fbaipublicfiles.com/jepa-wms/jepa_wm_droid.pth) |
| Metaworld | 256×256 | DINOv3 ViT-S/16 | 6 | [download](https://dl.fbaipublicfiles.com/jepa-wms/jepa_wm_metaworld.pth) |
| Push-T | 224×224 | DINOv2 ViT-S/14 | 6 | [download](https://dl.fbaipublicfiles.com/jepa-wms/jepa_wm_pusht.pth) |
| PointMaze | 224×224 | DINOv2 ViT-S/14 | 6 | [download](https://dl.fbaipublicfiles.com/jepa-wms/jepa_wm_pointmaze.pth) |
| Wall | 224×224 | DINOv2 ViT-S/14 | 6 | [download](https://dl.fbaipublicfiles.com/jepa-wms/jepa_wm_wall.pth) |

### DINO-WM Baseline Models

| Environment | Resolution | Encoder | Pred. Depth | Weights |
|-------------|------------|---------|-------------|---------|
| DROID & RoboCasa  | 224×224 | DINOv2 ViT-S/14 | 6 | [download](https://dl.fbaipublicfiles.com/jepa-wms/dino_wm_droid.pth) |
| Metaworld | 224×224 | DINOv2 ViT-S/14 | 6 | [download](https://dl.fbaipublicfiles.com/jepa-wms/dino_wm_metaworld.pth) |
| Push-T | 224×224 | DINOv2 ViT-S/14 | 6 | [download](https://dl.fbaipublicfiles.com/jepa-wms/dino_wm_pusht.pth) |
| PointMaze | 224×224 | DINOv2 ViT-S/14 | 6 | [download](https://dl.fbaipublicfiles.com/jepa-wms/dino_wm_pointmaze.pth) |
| Wall | 224×224 | DINOv2 ViT-S/14 | 6 | [download](https://dl.fbaipublicfiles.com/jepa-wms/dino_wm_wall.pth) |

### V-JEPA-2-AC(fixed) Baseline Model

| Environment | Resolution | Encoder | Pred. Depth | Weights |
|-------------|------------|---------|-------------|---------|
| DROID & RoboCasa | 256×256 | V-JEPA-2 ViT-G/16 | 6 | [download](https://dl.fbaipublicfiles.com/jepa-wms/vjepa2_ac_droid.pth) |

<details>
<summary><b>🔌 Loading Models with PyTorch Hub</b></summary>

```python
import torch

# Load our best pretrained JEPA-WMs
model, preprocessor = torch.hub.load('facebookresearch/jepa-wms', 'jepa_wm_droid')
model, preprocessor = torch.hub.load('facebookresearch/jepa-wms', 'jepa_wm_metaworld')
model, preprocessor = torch.hub.load('facebookresearch/jepa-wms', 'jepa_wm_pusht')
model, preprocessor = torch.hub.load('facebookresearch/jepa-wms', 'jepa_wm_pointmaze')
model, preprocessor = torch.hub.load('facebookresearch/jepa-wms', 'jepa_wm_wall')

# Load reproduced DINO-WM baseline models
model, preprocessor = torch.hub.load('facebookresearch/jepa-wms', 'dino_wm_droid')
model, preprocessor = torch.hub.load('facebookresearch/jepa-wms', 'dino_wm_metaworld')
model, preprocessor = torch.hub.load('facebookresearch/jepa-wms', 'dino_wm_pusht')
model, preprocessor = torch.hub.load('facebookresearch/jepa-wms', 'dino_wm_pointmaze')
model, preprocessor = torch.hub.load('facebookresearch/jepa-wms', 'dino_wm_wall')

# Load fixed V-JEPA-2-AC baseline model
model, preprocessor = torch.hub.load('facebookresearch/jepa-wms', 'vjepa2_ac_droid')
```

</details>

---

## 🚀 Getting Started

### Installation

We use **conda** for system dependencies (FFmpeg) and **uv** for fast Python package management.

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create conda environment with FFmpeg
conda create -n jepa-wms python=3.10 ffmpeg=7 -c conda-forge -y
conda activate jepa-wms

# 3. Clone and install
git clone git@github.com:facebookresearch/jepa-wms.git
cd jepa-wms
uv pip install -e .

# 4. Verify installation
python -c "import torchcodec; print('✓ torchcodec works')"
```

### ⚙️ Configuration

Set these environment variables in your `~/.bashrc` or `~/.zshrc`:

```bash
export DATASET_ROOT=/path/to/your/datasets
export CHECKPOINT_ROOT=/path/to/your/checkpoints
export JEPA_WM_HOME=/path/to/your/workspace
export PRETRAINED_CKPT_ROOT=/path/to/your/pretrained_encoders  # Optional
```

Then run:
```bash
source ~/.bashrc && cd $JEPA_WM_HOME/jepa-wms && python setup_macros.py && conda activate jepa-wms
```

<details>
<summary><b>📁 Repository structure under JEPA_WM_HOME</b></summary>

```
$JEPA_WM_HOME/
├── jepa-wms/          # This repository
├── dinov3/            # DINOv3 repository (optional)
├── robocasa/          # RoboCasa repository (optional)
└── robosuite/         # RoboSuite repository (optional)
```

</details>

<details>
<summary><b>🧠 Downloading pretrained encoders</b></summary>

Organize your pretrained checkpoints in `$PRETRAINED_CKPT_ROOT`:

```
$PRETRAINED_CKPT_ROOT/
├── dinov3/                # DINOv3 checkpoints
│   ├── dinov3_vits16_pretrain_lvd1689m.pth
│   └── dinov3_vitl16_pretrain_lvd1689m-<your-hashkey>.pth
├── vjepa1_opensource/     # V-JEPA v1 checkpoints
│   └── vitl16.pth.tar
└── vjepa2_opensource/     # V-JEPA v2 checkpoints
    ├── vjepa2_vit_large.pth
    └── vjepa2_vit_giant.pth
```

Download from:
- **DINOv3**: [dinov3](https://github.com/facebookresearch/dinov3) → [ViT-S/16 distilled](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
- **V-JEPA**: [vjepa](https://github.com/facebookresearch/jepa) → [ViT-L/16](https://dl.fbaipublicfiles.com/jepa/vitl16/vitl16.pth.tar)
- **V-JEPA-2**: [vjepa2](https://github.com/facebookresearch/vjepa2) → [ViT-L/16](https://dl.fbaipublicfiles.com/vjepa2/vitl.pt) or [ViT-G/16](https://dl.fbaipublicfiles.com/vjepa2/vitg.pt)

</details>

<details>
<summary><b>🤖 MuJoCo 2.1 for PointMaze</b></summary>

Only required for PointMaze (uses `d4rl` → `mujoco-py`). Other environments use the modern `mujoco` package.

```bash
# Download MuJoCo 2.1.0
mkdir -p ~/.mujoco && cd ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xzvf mujoco210-linux-x86_64.tar.gz

# Add to ~/.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
source ~/.bashrc  # or ~/.zshrc

# Verify installation
python -c "import mujoco_py; print('mujoco-py works!')"
```

</details>

<details>
<summary><b>🏠 RoboCasa install (optional)</b></summary>

Required for RoboCasa/RoboSuite environments:

```bash
# Install RoboSuite
git clone https://github.com/Basile-Terv/robosuite.git && cd robosuite
git checkout robocasa-dev && uv pip install -e . && cd ..

# Install RoboCasa
git clone https://github.com/Basile-Terv/robocasa.git && cd robocasa
uv pip install -e .
python robocasa/scripts/download_kitchen_assets.py  # ~5GB
python robocasa/scripts/setup_macros.py && cd ..
```

</details>

---

## 📦 Downloading Data

All datasets are available on 🤗 HuggingFace: [facebook/jepa-wms](https://huggingface.co/datasets/facebook/jepa-wms)

```bash
# Download all datasets
python src/scripts/download_data.py

# Download specific dataset(s)
python src/scripts/download_data.py --dataset pusht pointmaze wall

# List available datasets
python src/scripts/download_data.py --list
```

| Dataset | Description |
|---------|-------------|
| `pusht` | Push-T environment trajectories* |
| `pointmaze` | PointMaze navigation trajectories* |
| `wall` | Wall environment trajectories* |
| `metaworld` | 42 Metaworld tasks (100 episodes each) |
| `robocasa` | RoboCasa kitchen manipulation |
| `franka` | Franka robot trajectories |

> *\* The `pusht`, `pointmaze`, and `wall` datasets are sourced from the [DINO-WM project](https://github.com/apple/ml-dino-wm) without modification. We re-host them on our HuggingFace repository for convenience.*

<details>
<summary><b>📂 Dataset directory structure</b></summary>

```
$DATASET_ROOT/
├── pusht_noise/           # Push-T dataset
├── point_maze/            # PointMaze dataset
├── wall_single/           # Wall dataset
├── Metaworld/             # Metaworld dataset
│   └── data/
│       └── train-00000-of-00001.parquet
├── robocasa/              # RoboCasa dataset
│   └── combine_all_im256.hdf5
├── franka_custom/         # Franka custom dataset
│   └── data/
│       ├── folding/
│       ├── pick/
│       └── push/
│           ├── brownboxpush_v0/
│           │   └── run_0001/
│           │       ├── episode.h5
│           │       └── trajectory.hdf5
│           └── push_various_objects/
├── DROID/                 # DROID dataset
│   └── droid_paths.csv
├── kinetics400/           # Kinetics-400 dataset (optional)
│   ├── k400_train_paths.csv
│   └── k400_val_paths.csv
├── kinetics710/           # Kinetics-710 dataset (optional)
│   ├── k710_train_paths.csv
│   └── k710_val_paths.csv
├── ssv2/                  # Something-Something-v2 dataset (optional)
│   ├── ssv2_train_paths.csv
│   └── ssv2_val_paths.csv
└── howto100m/             # HowTo100M dataset (optional)
    └── howto100m_paths.csv
```

</details>

<details>
<summary><b>🤖 DROID dataset</b></summary>

DROID requires separate download via `gsutil`:

Download the DROID dataset following the [instructions](https://droid-dataset.github.io/droid/the-droid-dataset). This requires `uv pip install gsutil`.
We only use the left camera and not the SVO cam files hence you can run the second of the two below commands to obtain the raw dataset of full-HD resolution (720 x 1280) MP4 files.
```bash
# Raw DROID dataset in stereo HD, stored as MP4 videos (8.7TB)
gsutil -m cp -r gs://gresearch/robotics/droid_raw <path_to_your_target_dir>
# Raw DROID dataset, non-stereo HD video only (5.6TB, excluding stereo video & raw SVO cam files)
gsutil -m rsync -r -x ".*SVO.*|.*stereo.*\.mp4$" "gs://gresearch/robotics/droid_raw" <path_to_your_target_dir>
```

After downloading, generate the paths CSV file required by the dataloader:
```bash
python src/scripts/generate_droid_paths.py \
    --droid_root <path_to_your_target_dir>/droid_raw/1.0.1 \
    --output_path $DATASET_ROOT/DROID/droid_paths.csv \
    --num_workers 16 \
```

This script scans the dataset directory structure in parallel and creates a CSV file listing all valid episode paths.

</details>

---

## 💡 Common Concepts

### 🐛 The `--debug` Flag

Use `--debug` with `app.main` or `evals.main` to run in **single-process mode** on the current node:

```bash
python -m app.main --fname <config.yaml> --debug
```

This is useful for:
- **Interactive debugging** with `pdb` breakpoints
- **Single-GPU runs** without distributed overhead

> ⚠️ **Don't confuse** with `meta.quick_debug` in config files, which reduces dataset size and iterations for quick sanity checks.

### 🔄 Automatic Evaluation During Training

The training script automatically launches planning evaluations every `meta.eval_freq` epochs:

1. **Config generation**: Merges your training settings with eval templates from `configs/online_plan_evals/`
2. **Job submission**: Launches eval jobs for each generated config

The `evals.separate` option controls how evals are executed:

| Value | Behavior |
|-------|----------|
| `true` *(default)* | Submit as **separate SLURM jobs** via sbatch |
| `false` | Run evals **on rank 0** of the training job |

---

## 🏋️ Training

### Quick Start

**Distributed training** (from login node):
```bash
python -m app.main_distributed --fname configs/vjepa_wm/<env>_sweep/<model>.yaml
```

**Single-GPU training** (interactive session):
```bash
python -m app.main --fname configs/vjepa_wm/<env>_sweep/<model>.yaml --debug
```

<details>
<summary><b>📋 Paper Configs</b></summary>

| Model | Environment | Config Path |
|-------|-------------|-------------|
| **JEPA-WM** | Metaworld | `mw_sweep/mw_4f_fsk5_ask1_r256_dv3vits_vjtranoaug_pred_AdaLN_ftprop_depth6_repro_2roll_save` |
| **JEPA-WM** | PointMaze | `mz_sweep/mz_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save_2n` |
| **JEPA-WM** | Push-T | `pt_sweep/pt_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save` |
| **JEPA-WM** | Wall | `wall_sweep/wall_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save_2n` |
| **JEPA-WM** | RoboCasa | `droid_final_sweep/droid_4f_fps4_r256_dv3vits_vjtranoaug_pred_AdaLN_depth6_repro_2roll_4fpcs_2n` |
| **JEPA-WM** | DROID (offline) | `droid_final_sweep/droid_4f_fps4_r256_dv3vits_vjtranoaug_pred_AdaLN_depth6_noprop_repro_2roll_4fpcs_2n` |
| **DINO-WM** | Any | `<env>_sweep/<env>_4f_fsk5_ask1_r224_pred_dino_wm_depth6_repro_1roll_save` |

All configs under `configs/vjepa_wm/`.

</details>

<details>
<summary><b>🎨 Training Decoder Heads (optional)</b></summary>

Decoder heads enable visualization and light evals (rollout decoding via `val_rollout()` in the training loop).

**Two training strategies:**
- **Cross-environment** (recommended if available): Train one head on VideoMix2M (HowTo100M + SSv2 + K400) — works across all environments
- **In-domain**: Train one head per encoder per environment on environment-specific data

```bash
# State head
python -m app.main --fname configs/vjepa_wm/<env>/step2_lpips_<env>_state_head_dinovitb_r224.yaml --debug

# Image decoder head
python -m app.main --fname configs/vjepa_wm/<env>/step2_lpips_<env>_vitbdec_dinovitbenc_224_beta0.95.yaml --debug
```

</details>

---

## 📊 Evaluation

```bash
# Single GPU
python -m evals.main --fname <config.yaml> --debug

# Distributed
python -m evals.main_distributed --fname <config.yaml> --account <account> --qos lowest --time 120

# Grid evaluation (sweep over hyperparameters or epoch checkpoints)
python -m evals.simu_env_planning.run_eval_grid --env <env> --config <config.yaml>
```

> 📓 **Visualization**: `app/plan_common/notebooks/logs_planning_joint.ipynb`

> **Full documentation**: [`evals/simu_env_planning/README.md`](evals/simu_env_planning/README.md)

<details>
<summary><b>⚙️ Eval Config Generation</b></summary>

Eval configs are auto-generated during training (see [Automatic eval during training](#automatic-eval-during-training)). To manually generate configs for an already-trained model without launching evals:

1. Set `meta.plan_only_eval_mode: true` in your training config
2. Set `evals.dump_eval_configs: true` in your training config
3. Run: `python -m app.main --fname <config.yaml> --debug`

The dump directory is automatically derived from `evals.eval_cfg_paths` (e.g., `configs/online_plan_evals/mz/...` → `configs/dump_online_evals/mz/`).

</details>

<details>
<summary><b>🔮 Unroll Decode Evaluation</b></summary>

Counterfactual decoding evaluation that generates predictions with hardcoded custom actions. This is useful for visualizing how the world model responds to specific action scenarios (e.g., "open gripper + move up" vs "close gripper + move up").

> **Note**: This evaluation is designed to work only with **DROID or franka_custom data**.

To run unroll decode evaluation, set `meta.unroll_decode_eval_only_mode: true` in your training config and configure `unroll_decode_evals`:

```yaml
meta:
  unroll_decode_eval_only_mode: true
unroll_decode_evals:
  specific_video: true  # Use a specific video file
  specific_video_path: /path/to/video.npz  # Optional: path to npz file
  play_in_reverse: false
  repeat_hardcode_act: 5  # Number of times to repeat hardcoded actions
  wrapper_kwargs:  # Same structure as evals.wrapper_kwargs
    ctxt_window: 2
```

The hardcoded actions can be customized by modifying the `create_counterfactual_actions()` function in `evals/unroll_decode/eval.py`.

</details>

---

## 📁 Code Structure

```
.
├── app                              # training loops
│   ├── vjepa_wm                     #   train world model / heads
│   ├── main_distributed.py          #   entrypoint for sbatch on slurm
│   └── main.py                      #   entrypoint for local run
├── configs                          # config files
│   ├── dump_online_evals            #   generated eval cfgs from train loop
│   ├── evals                        #   pre-generated full eval cfgs
│   ├── online_plan_evals            #   eval cfg templates to fill with train cfg
│   ├── vjepa_wm                     #   train configs
├── evals                            # evaluations
│   ├── simu_env_planning            #   planning evaluation
│   ├── main_distributed.py          #   entrypoint for distributed evals
│   └── main.py                      #   entrypoint for local evals
├── src                              # the package
│   ├── datasets                     #   VM2M datasets, loaders (optional)
│   ├── models                       #   V-JEPA1/2 model definitions
│   ├── masks                        #   masking utilities (optional)
│   └── utils                        #   shared utilities
├── tests                            # unit tests for some modules

```

## 🔧 Troubleshooting

<details>
<summary><b>🖥️ MuJoCo Rendering</b></summary>

If you encounter MuJoCo rendering errors during evaluation (especially on headless servers or clusters), you may need to configure the rendering backend by setting these environment variables before running your scripts:

```bash
# For systems with EGL support (e.g., NVIDIA GPUs with recent drivers)
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# For systems without EGL (e.g., CPU-only rendering)
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
```

**When to use each backend:**
- **EGL**: Preferred for GPU-accelerated rendering on headless servers with NVIDIA GPUs and recent drivers. Provides better performance.
- **OSMesa**: Fallback option for CPU-based rendering when EGL is not available. Slower but more compatible.

**Common error messages:**
- `"ERROR: GLEW initialization error: Missing GL version"` → Try using `osmesa` backend
- `"Cannot initialize EGL"` → Try using `osmesa` backend or check GPU drivers
- Rendering appears blank or corrupted → Verify the correct backend for your system

</details>

<details>
<summary><b>🚀 Distributed jobs</b></summary>

You cannot launch a main_distributed.py job from a GPU node if you do not clear the env variables, as is done with `with submitit.helpers.clean_env():` in `app/vjepa_wm/train.py`.

</details>

<details>
<summary><b>🔄 Updating uv.lock</b></summary>

If you encounter errors when loading checkpoints from torchhub such as `urllib.error.HTTPError: HTTP Error 503: Service Unavailable`, you should `rm uv.lock`, then recreate your uv venv with `uv sync`, activate this new env and rerun your command.

</details>

<details>
<summary><b>🐍 numba/numpy issues</b></summary>

if running into issues with numba/numpy because of the numba dependency of robocasa, run:
```
conda install -c numba numba=0.56.4 -y
```

</details>

---

## 📄 License

This project is licensed under [CC-BY-NC 4.0](LICENSE). See [THIRD-PARTY-LICENSES.md](THIRD-PARTY-LICENSES.md) for third-party components.

---

## 📚 Citing JEPA-WMs

If you find this repository useful, please consider giving a ⭐ and citing:

`TODO`
```bibtex
@article{terver2025jepawms,
  title={What drives success in physical planning with Joint-Embedding Predictive World Models?},
  author={Terver, Basile and Yang, Jimmy and Ponce, Jean and Bardes, Adrien and LeCun, Yann},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```
