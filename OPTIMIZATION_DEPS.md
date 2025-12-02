# Optimization Dependencies Note

The optimization libraries (nevergrad, cma, bayesian-optimization, qpsolvers) have been moved to a separate requirements file due to dependency conflicts with the base environment.

## The Conflict

- **Base environment** requires `numpy==1.23.3` and `numba==0.56.4` (which needs `numpy<1.24`)
- **Optimization libraries** require `numpy>=1.24.0`

These requirements are incompatible and cannot coexist in the same environment.

## Installation Options

### Option 1: Separate Virtual Environment (Recommended)
Create a dedicated environment for optimization work:

```bash
python -m venv venv-optimization
source venv-optimization/bin/activate  # On Windows: venv-optimization\Scripts\activate
pip install -r requirements-optimization.txt
```

### Option 2: Upgrade Base Environment (Not Recommended)
⚠️ **Warning:** This will break `numba` compatibility!

```bash
pip install --upgrade 'numpy>=1.24.0'
pip install -r requirements-optimization.txt
```

### Option 3: Use Conda Environment with Compatible Versions
Try to find compatible versions of all packages:

```bash
conda create -n jepa-opt python=3.10
conda activate jepa-opt
# Install packages with conda to handle compatibility
conda install numpy numba nevergrad cma -c conda-forge
```

## Using uv without Optimization Dependencies

The optimization dependencies have been removed from `pyproject.toml`, so you can now run:

```bash
uv sync  # Installs base dependencies only
```

To install development tools:
```bash
uv sync --extra dev
```
