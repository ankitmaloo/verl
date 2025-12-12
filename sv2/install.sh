#!/bin/bash
# sv2 standalone setup script
# Installs verl 0.6.1 with sglang backend using uv
#
# What verl[sglang] installs (from setup.py):
#   - sglang[srt,openai]==0.5.5
#   - torch==2.8.0
#   - tensordict>=0.8.0,<=0.10.0,!=0.9.0
#
# Requirements:
#   - NVIDIA GPU with CUDA >= 12.1
#   - nvidia-smi working
#   - ~20GB disk space for packages

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
PYTHON_VERSION="3.12"

echo "=== sv2 standalone setup ==="
echo ""
echo "Using verl[sglang] which pins:"
echo "  - sglang[srt,openai]==0.5.5"
echo "  - torch==2.8.0"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Step 0: Check CUDA availability
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Checking CUDA ==="

if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. CUDA driver not installed?"
    echo "       sv2 requires NVIDIA GPU with CUDA >= 12.1"
    exit 1
fi

DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "NVIDIA driver version: $DRIVER_VERSION"

# Check CUDA version from nvcc if available
if command -v nvcc &> /dev/null; then
    NVCC_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    echo "CUDA toolkit version: $NVCC_VERSION"

    CUDA_MAJOR=$(echo $NVCC_VERSION | cut -d. -f1)
    CUDA_MINOR=$(echo $NVCC_VERSION | cut -d. -f2)

    if [ "$CUDA_MAJOR" -lt 12 ] || ([ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -lt 1 ]); then
        echo "ERROR: CUDA $NVCC_VERSION detected. verl requires CUDA >= 12.1"
        exit 1
    fi
    echo "CUDA version OK"
else
    echo "WARNING: nvcc not found. PyTorch will use bundled CUDA runtime."
fi

echo ""
echo "GPU(s) detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -4
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Install uv
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Installing uv ==="

if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "uv version: $(uv --version)"

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Create virtual environment
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Creating virtual environment ==="

if [ -d "$VENV_DIR" ]; then
    echo "Existing venv found at $VENV_DIR"
    read -p "Remove and recreate? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
    else
        echo "Keeping existing venv. Activate with: source $VENV_DIR/bin/activate"
        exit 0
    fi
fi

uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
source "$VENV_DIR/bin/activate"
echo "Virtual environment created: $VENV_DIR"
echo "Python: $(python --version)"

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Install verl with sglang extra (handles torch + sglang versions)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Installing verl[sglang] ==="
echo "This installs verl + sglang + torch==2.8.0 with correct versions..."

uv pip install "verl[sglang]==0.6.1" --no-cache-dir

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Install additional dependencies for sv2
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Installing additional dependencies ==="

# torch-memory-saver for memory optimization
uv pip install torch-memory-saver --no-cache-dir

# FlashInfer for fast attention (optional but recommended)
echo "Installing FlashInfer..."
uv pip install flashinfer-python --no-cache-dir || echo "Warning: FlashInfer install failed (may need manual install)"

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Verify installation
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Verifying installation ==="

python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    print('WARNING: CUDA not available!')
"

echo ""
python -c "import verl; print(f'verl: {verl.__version__}')" || echo "Warning: verl import failed"
python -c "import sglang; print(f'sglang: {sglang.__version__}')" || echo "Warning: sglang import failed"
python -c "import hydra; print('hydra: OK')"
python -c "import ray; print(f'ray: {ray.__version__}')"

# ─────────────────────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "=== sv2 setup complete ==="
echo "=========================================="
echo ""
echo "Activate: source $VENV_DIR/bin/activate"
echo ""
echo "Usage:"
echo "  python -m sv2.main_ppo_multiturn_toolcall \\"
echo "    --config-path sv2/config --config-name sv2_multiturn \\"
echo "    data.val_files=\$DATA_DIR/test.parquet"
