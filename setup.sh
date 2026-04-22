#!/bin/bash
set -e

# To clone the initial commit
# git clone --depth 1 https://github.com/Planning-for-WMs/jepa-wms.git
# cd jepa-wms
# git fetch --unshallow
#git reset --hard 4ecff64
# cd ..

# To clone the latest commit in main
git clone https://github.com/Planning-for-WMs/jepa-wms.git

echo "Setting up conda env"
curl -LsSf https://astral.sh/uv/install.sh | sh
source /opt/miniforge3/etc/profile.d/conda.sh
conda create -n jepa-wms python=3.10 ffmpeg=7 -c conda-forge -y
conda activate jepa-wms
cd jepa-wms

echo "Installing uv packages"
uv pip install -e .
uv pip install -e ".[dev]"

# Reinstall PyTorch with CUDA 12.8 support (required for Blackwell GPUs, e.g. RTX 5070 Ti / sm_120)
# echo "Reinstalling PyTorch with CUDA 12.8 support"
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

python -c "import torchcodec; print('✓ torchcodec works')"

echo "Modifying global variables in ~/.bashrc"
cat >> ~/.bashrc << EOF

# ===== JEPAWM Environment Variables =====
export JEPAWM_DSET=./jepa-wms/datasets
export JEPAWM_LOGS=./jepa-wms/logs
export JEPAWM_HOME=./jepa-wms
export JEPAWM_CKPT=./jepa-wms/checkpoints
# ===== End JEPAWM Variables =====
conda activate jepa-wms
EOF

source ~/.bashrc && python setup_macros.py

# Download datasets
python -c "from huggingface_hub import login; login()"
python src/scripts/download_data.py --dataset pusht

# Download pretrained Push-T JEPA-WM checkpoint (pred depth 6)
echo "Downloading Push-T JEPA-WM checkpoint"
mkdir -p logs/pt_sweep/pt_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save
wget -q --show-progress \
    -O logs/pt_sweep/pt_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save/jepa-latest.pth.tar \
    https://dl.fbaipublicfiles.com/jepa-wms/pt_jepa-wm.pth.tar
echo "✓ Checkpoint downloaded"

# Run this to check
# python -m evals.main --fname configs/evals/simu_env_planning/pt/jepa-wm/pt_L2_cem_sourcedset_H6_nas6_ctxt2_r224_alpha0.1_quickcheck.yaml --debug
