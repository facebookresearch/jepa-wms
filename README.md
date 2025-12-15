# JEPA-WMs: What drives success in physical planning with Joint-Embedding Predictive World Models?
[[`ArXiv`]()] [[`BibTeX`](#citing-jepa-wms)]
`TODO`

**[Meta AI Research, FAIR](https://ai.facebook.com/research/)**

[Basile Terver](https://x.com/BasileTerv987),
Jimmy Yang,
Jean Ponce,
Adrien Bardes,
Yann Le Cun

PyTorch implementation and pretrained models for JEPA-WMs. For details, see [**What drives success in physical planning with Joint-Embedding Predictive World Models?**](). This repository contains the code and models to reproduce the paper.

![JEPA-WMs diagram](assets/train_plan_schema.png)

## Pretrained models

We provide pretrained JEPA-WM, [DINO-WM](https://arxiv.org/abs/2411.04983) and [V-JEPA-2-AC(fixed)](https://arxiv.org/abs/2506.09985) baseline models for various environments.

### JEPA-WM Models

| Environment | Resolution | Encoder | Pred. Depth | Weights |
|-------------|------------|---------|-----------------|---------|
| DROID & RoboCasa | 256×256 | DINOv3 ViT-S/16 | 6 | [download](https://dl.fbaipublicfiles.com/jepa-wms/jepa_wm_droid.pth) |
| Metaworld | 256×256 | DINOv3 ViT-S/16 | 6 | [download](https://dl.fbaipublicfiles.com/jepa-wms/jepa_wm_metaworld.pth) |
| Push-T | 224×224 | DINOv2 ViT-S/14 | 6 | [download](https://dl.fbaipublicfiles.com/jepa-wms/jepa_wm_pusht.pth) |
| PointMaze | 224×224 | DINOv2 ViT-S/14 | 6 | [download](https://dl.fbaipublicfiles.com/jepa-wms/jepa_wm_pointmaze.pth) |
| Wall | 224×224 | DINOv2 ViT-S/14 | 6 | [download](https://dl.fbaipublicfiles.com/jepa-wms/jepa_wm_wall.pth) |

### DINO-WM Baseline Models

| Environment | Resolution | Encoder | Pred. Depth | Weights |
|-------------|------------|---------|-----------------|---------|
| DROID & RoboCasa  | 224×224 | DINOv2 ViT-S/14 | 6 | [download](https://dl.fbaipublicfiles.com/jepa-wms/dino_wm_droid.pth) |
| Metaworld | 224×224 | DINOv2 ViT-S/14 | 6 | [download](https://dl.fbaipublicfiles.com/jepa-wms/dino_wm_metaworld.pth) |
| Push-T | 224×224 | DINOv2 ViT-S/14 | 6 | [download](https://dl.fbaipublicfiles.com/jepa-wms/dino_wm_pusht.pth) |
| PointMaze | 224×224 | DINOv2 ViT-S/14 | 6 | [download](https://dl.fbaipublicfiles.com/jepa-wms/dino_wm_pointmaze.pth) |
| Wall | 224×224 | DINOv2 ViT-S/14 | 6 | [download](https://dl.fbaipublicfiles.com/jepa-wms/dino_wm_wall.pth) |

### V-JEPA-2-AC(fixed) Baseline Model

| Environment | Resolution | Encoder | Pred. Depth | Weights |
|-------------|------------|---------|-----------------|---------|
| DROID & RoboCasa | 256×256 | V-JEPA-2 ViT-G/16 | 6 | [download](https://dl.fbaipublicfiles.com/jepa-wms/vjepa2_ac_droid.pth) |

### Loading Models with PyTorch Hub

You can easily load pretrained models using PyTorch Hub:

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

# Load fixed V-JEPA-2-AC baseline models
model, preprocessor = torch.hub.load('facebookresearch/jepa-wms', 'vjepa2_ac_droid')
```

## Getting Started

### Environment Setup

The environment to run the codebase is straightforward to create using [`uv`](https://github.com/astral-sh/uv). If you don't have it, you can install it using:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then clone the repository in the `JEPA_WM_HOME` (see [Path Configuration](#path-configuration)) with:
```bash
git clone git@github.com:facebookresearch/jepa-wms.git
cd jepa-wms
```

Once it is installed, you can create your environment using:
```bash
uv sync                # Install base dependencies
uv sync --extra dev    # Install base + dev dependencies
```

This environment can be activated using `. .venv/bin/activate`, and deactivated using `deactivate`. ou can also run Python scripts using the venv without activating it using `uv run script.py`.

### Path Configuration

To make the repository agnostic to specific cluster configurations, we use environment variables to configure dataset and checkpoint paths. **You must set these environment variables before running the code.**

1. **`DATASET_ROOT`**: Root directory where all datasets are stored
2. **`CHECKPOINT_ROOT`**: Root directory where checkpoints and experiment outputs will be saved
3. **`JEPA_WM_HOME`**: parent directory containing all repositories
4. **`PRETRAINED_CKPT_ROOT`**: Root directory where pretrained checkpoints are stored (e.g., DINOv3, V-JEPA, V-JEPA-2)

Add the following lines to your shell configuration file (e.g., `~/.bashrc`, `~/.zshrc`, or `~/.bash_profile`):

```bash
# JEPA-WMs Path Configuration
export DATASET_ROOT=</path/to/your/datasets>
export CHECKPOINT_ROOT=</path/to/your/checkpoints>
export JEPA_WM_HOME=</path/to/your/workspace>
export PRETRAINED_CKPT_ROOT=</path/to/your/pretrained_encoders>  # Optional
```
After adding these lines, reload your shell configuration:
```bash
source ~/.bashrc  # or ~/.zshrc, depending on your shell
```

Then generate the local `macros.py` file (used by notebooks and scripts):
```bash
cd $JEPA_WM_HOME/jepa-wms
python setup_macros.py
```

Once you set `JEPA_WM_HOME`, you will have to organize your cloned repositories as follows (see below instructions on [installing Robocasa](#optional-robocasa-install) and [Downloading pretrained encoders](#downloading-pretrained-encoders)):
```
$JEPA_WM_HOME/
├── jepa-wms/          # This repository
├── dinov3/            # DINOv3 repository
├── robocasa/          # RoboCasa repository
└── robosuite/         # RoboSuite repository
```

### Downloading pretrained encoders

Once you set `PRETRAINED_CKPT_ROOT`, organize your pretrained checkpoints as follows :

```
$PRETRAINED_CKPT_ROOT/
├── dinov3/                # DINOv3 checkpoints
│   ├── dinov3_vits16_pretrain_lvd1689m.pth
│   └── dinov3_vitl16_pretrain_lvd1689m-<your-hashkey>.pth
├── vjepa1_opensource/     # V-JEPA v1 checkpoints
│   └── vitl16.pth.tar
├── vjepa2_opensource/     # V-JEPA v2 checkpoints
│   ├── vjepa2_vit_large.pth
│   └── vjepa2_vit_giant.pth
```

To download pretrained encoders, follow the instructions from their respective repositories:

- **DINOv3**: Follow instructions at [dinov3](https://github.com/facebookresearch/dinov3) to download DINOv3 checkpoints. To reproduce our paper, you only need to download the [ViT-S/16 distilled](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/). You will receive an e-mail with downloading urls. Place the downloaded checkpoint(s) in `$PRETRAINED_CKPT_ROOT/dinov3/`. You should also `git clone git@github.com:facebookresearch/dinov3.git` inside `JEPA_WM_HOME`. Optional: If you want to use the ViT-L/16 encoder, you should replace the hashkey `pretrain_lvd1689m-<your-hashkey>.pth` with yours in `app/plan_common/models/dino.py`

- **V-JEPA**: Follow instructions at [vjepa](https://github.com/facebookresearch/jepa) to download V-JEPA v1 checkpoints. You only need the [ViT-L/16](https://dl.fbaipublicfiles.com/jepa/vitl16/vitl16.pth.tar) to reproduce our paper. Place the downloaded checkpoints in `$PRETRAINED_CKPT_ROOT/vjepa1_opensource/`.

- **V-JEPA-2**: Follow instructions at [vjepa2](https://github.com/facebookresearch/vjepa2) to download V-JEPA-2 checkpoints. You only need the [ViT-L/16](https://dl.fbaipublicfiles.com/vjepa2/vitl.pt) to reproduce our paper and need the [ViT-G/16](https://dl.fbaipublicfiles.com/vjepa2/vitg.pt) to reproduce the V-JEPA-2-AC baseline. Place the downloaded checkpoints in `$PRETRAINED_CKPT_ROOT/vjepa2_opensource/`.

### MuJoCo 2.1 for PointMaze (mujoco-py)

The PointMaze environment uses `d4rl`, which depends on `mujoco-py`, which requires a system-level MuJoCo 2.1.0 installation.
Other environments (Push-T, Wall, Metaworld, RoboCasa) use the modern `mujoco` package and do not require this setup.

1. **Download and extract MuJoCo 2.1.0**:
   ```bash
   mkdir -p ~/.mujoco
   wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -P ~/.mujoco/
   cd ~/.mujoco
   tar -xzvf mujoco210-linux-x86_64.tar.gz
   ```

2. **Set environment variables** by adding these lines to your `~/.bashrc` (or `~/.zshrc`):
   ```bash
   # MuJoCo 2.1.0 for mujoco-py
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
   # NVIDIA Library Path (if using NVIDIA GPUs)
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
   ```

3. **Reload your shell configuration**:
   ```bash
   source ~/.bashrc  # or ~/.zshrc
   ```

4. **Verify the installation**:
   ```bash
   python -c "import mujoco_py; print('mujoco-py works!')"
   ```

### Optional: Robocasa install

For robot manipulation environments (RoboCasa, RoboSuite), you need to manually install these dependencies from source, as they cannot be installed via pip alone.

1. **Install RoboSuite** (use the `robocasa-dev` branch):
   ```bash
   git clone https://github.com/Basile-Terv/robosuite.git
   cd robosuite
   git checkout robocasa-dev
   uv pip install -e .
   cd ..
   ```

2. **Install RoboCasa**:
   ```bash
   git clone https://github.com/Basile-Terv/robocasa.git
   cd robocasa
   uv pip install -e .

   # Optional: set up code formatter
   uv pip install pre-commit
   pre-commit install

   # If you encounter issues with numba/numpy, run:
   # conda install -c numba numba=0.56.4 -y
   ```

3. **Download RoboCasa assets and setup**:
   ```bash
   python robocasa/scripts/download_kitchen_assets.py   # Caution: Assets are around 5GB
   python robocasa/scripts/setup_macros.py              # Set up system variables
   cd ..
   ```

**Note**: If you're not using `uv`, replace `uv pip install` with `pip install` in the commands above.

### Downloading Data

Our experiments use datasets from multiple sources. Below are instructions for obtaining and setting up each dataset. We remind that you should have **set the environment variable** `export DATASET_ROOT=</path/to/your/datasets>`.

Once you have set `DATASET_ROOT`, you can follow the below instructions per dataset to organize your datasets as follows:

```
$DATASET_ROOT/
├── pusht_noise/           # Push-T dataset
├── point_maze/            # PointMaze dataset
├── wall_single/           # Wall dataset
├── Metaworld/             # Metaworld dataset
│   ├── h5folder
│   └── train_paths.csv
├── robocasa/              # RoboCasa dataset
│   └── combine_all_im256.hdf5
├── droid/                 # DROID dataset
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

#### Push-T, Wall, and PointMaze

For these environments, we use datasets from the [DINO-WM project](https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28).

1. **Download the datasets** from [this link](https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28).

2. **Extract the datasets**: Unzip the downloaded files.

**Note**: We do not use the deformable data from DINO-WM.

#### Metaworld

For Metaworld, we create a custom dataset by training [TD-MPC2](https://github.com/nicklashansen/tdmpc2) online agents and collecting the first 100 episodes from each of the 42 tasks considered in our experiments.

#### DROID

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

#### RoboCasa

For RoboCasa, download the trajectories from the following URL:

`TODO: Add RoboCasa dataset URL`


## Common Concepts

### The `--debug` flag

The `--debug` flag for `app.main` or `evals.main` launches a **single process on the current node**. This enables:
- Debugging with `pdb` breakpoints
- Running training/eval on a single GPU without distributed overhead

> **Note**: Don't confuse with `meta.quick_debug` in configs, which reduces dataset size and iterations for quick sanity checks.

### Automatic eval during training

Every `meta.eval_freq` epochs, the training script automatically:
1. Generates eval configs by merging training settings with templates from `configs/online_plan_evals/`
2. Launches eval jobs for each config

The `evals.separate` option controls execution:
- `true` (default): Submit as **separate distributed jobs** via sbatch
- `false`: Run **on rank 0** of the training job

## Training

### Quick Start

**Distributed training** (from login node):
```bash
python -m app.main_distributed --fname configs/vjepa_wm/<env>_sweep/<model>.yaml
```

**Single-GPU training** (interactive session):
```bash
python -m app.main --fname configs/vjepa_wm/<env>_sweep/<model>.yaml --debug
```

### Paper Configs

| Model | Environment | Config Path |
|-------|-------------|-------------|
| **JEPA-WM** | Metaworld | `mw_sweep/mw_4f_fsk5_ask1_r256_dv3vits_vjtranoaug_pred_AdaLN_ftprop_depth6_repro_2roll_save` |
| **JEPA-WM** | PointMaze | `mz_sweep/mz_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save_2n` |
| **JEPA-WM** | Push-T | `pt_sweep/pt_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save` |
| **JEPA-WM** | Wall | `wall_sweep/wall_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save_2n` |
| **JEPA-WM** | RoboCasa | `droid_final_sweep/droid_4f_fps4_r256_dv3vits_vjtranoaug_pred_AdaLN_depth6_repro_2roll_4fpcs_2n` |
| **JEPA-WM** | DROID (offline) | `droid_final_sweep/droid_4f_fps4_r256_dv3vits_vjtranoaug_pred_AdaLN_depth6_noprop_repro_2roll_4fpcs_2n` |
| **DINO-WM** | Any | `<env>_sweep/<env>_4f_fsk5_ask1_r224_pred_dino_wm_depth6_repro_1roll_save` |

All configs are under `configs/vjepa_wm/`.

### Training Heads (Optional)

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

## Evaluation

### Running Evals

**Interactive (single GPU)**:
```bash
python -m evals.main --fname <config.yaml> --debug
```

**Distributed (from login node)**:
```bash
python -m evals.main_distributed --fname <config.yaml> --account <account> --qos lowest --time 120
```

**Grid evaluation** (multiple hyperparameters):
```bash
python -m evals.simu_env_planning.run_eval_grid --env <env> --config <config.yaml>
```

### Eval Config Generation

Eval configs are auto-generated during training (see [Automatic eval during training](#automatic-eval-during-training)). To manually generate configs for an already-trained model without launching evals:

1. Set `meta.plan_only_eval_mode: true` in your training config
2. Set `evals.dump_eval_configs: true` in your training config
3. Run: `python -m app.main --fname <config.yaml> --debug`

The dump directory is automatically derived from `evals.eval_cfg_paths` (e.g., `configs/online_plan_evals/mz/...` → `configs/dump_online_evals/mz/`).

Example config snippet:
```yaml
meta:
  plan_only_eval_mode: true

evals:
  dump_eval_configs: true
  eval_cfg_paths:
    - configs/online_plan_evals/mz/ng/mz_L2_ng_sourcerandstate_H6_nas6_ctxt2.yaml
    # ... other eval config templates
```

Config naming: `<goal_source>_<planner>_<objective>.yaml`
- Goal sources: `dset`, `rand`, `itr` | Planners: `ng`, `cem` | Objectives: `L1`, `L2`

> **Note**: Set `dump_eval_configs: False` for distributed training.

### Planning Evaluation Details

Goal-conditioned trajectory optimization evaluating world models on physical planning tasks.

> **Full documentation**: [`evals/simu_env_planning/README.md`](evals/simu_env_planning/README.md)

**Visualization notebook**: `app/plan_common/notebooks/logs_planning_joint.ipynb`

### Unroll Decode Evaluation

Counterfactual decoding evaluation that generates predictions with hardcoded custom actions. This is useful for visualizing how the world model responds to specific action scenarios (e.g., "open gripper + move up" vs "close gripper + move up").

> **Note**: This evaluation is designed to work only with **DROID data**.

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

## Code Structure

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

## Troubleshooting

### MuJoCo Rendering

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

### Launching distributed jobs
And you cannot launch a main_distributed.py job from a GPU node if you do not clear the env variables, as is done with `with submitit.helpers.clean_env():` in `app/vjepa_wm/train.py`.

### Updating uv.lock
If you encounter errors when loading checkpoints from torchhub such as `urllib.error.HTTPError: HTTP Error 503: Service Unavailable`, you should `rm uv.lock`, then recreate your uv venv with `uv sync`, activate this new env and rerun your command.

## License

This project is licensed under the [CC-BY-NC 4.0 License](LICENSE) - see the LICENSE file for details.

For information about third-party components and their licenses, see [THIRD-PARTY-LICENSES.md](THIRD-PARTY-LICENSES.md).


## Citing JEPA-WMs
If you find this repository useful in your research, please consider giving a star :star: and a citation

`TODO`
```bibtex
@article{,
  title={},
  author={}
  journal={arXiv preprint arXiv},
  year={2025}
}
```
