#!/bin/bash
# sv2 standalone setup script
# Installs verl 0.6.1 with sglang backend using uv
#
# Installation order (for CUDA 12.4 stable):
#   1. PyTorch stable cu124 - install FIRST to set the correct CUDA version
#   2. verl[sglang] - installs all deps (uses existing torch)
#   3. sgl_kernel pre-built - works with stable torch
#
# Requirements:
#   - NVIDIA GPU with CUDA >= 12.1 (tested on GH200)
#   - nvidia-smi working
#   - ~15GB disk space for packages

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
PYTHON_VERSION="3.12"

echo "=== sv2 standalone setup (CUDA 12.4) ==="
echo ""
echo "Installation order:"
echo "  1. PyTorch stable cu124 (install first)"
echo "  2. verl[sglang] (uses existing torch)"
echo "  3. sgl_kernel pre-built wheel"
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
# Step 3: Install PyTorch with CUDA 12.4 FIRST
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Installing PyTorch with CUDA 12.4 ==="
echo "Installing stable PyTorch cu124 before other packages..."

uv pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124 \
    --no-cache-dir

# Verify CUDA is available
python -c "
import torch
if not torch.cuda.is_available():
    print('ERROR: PyTorch CUDA not available!')
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA version: {torch.version.cuda}')
    exit(1)
print(f'PyTorch {torch.__version__} with CUDA {torch.version.cuda} - OK')
print(f'GPU: {torch.cuda.get_device_name(0)}')
"

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Install verl[sglang] (will use existing torch)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Installing verl[sglang] ==="
echo "Note: Using existing PyTorch with CUDA 12.4"

# Use --no-deps for torch-related packages to avoid overwriting our CUDA torch
uv pip install "verl[sglang]==0.6.1" --no-cache-dir

# Re-verify PyTorch CUDA is still available (in case verl overwrote it)
CUDA_OK=$(python -c "import torch; print('1' if torch.cuda.is_available() else '0')")
if [ "$CUDA_OK" != "1" ]; then
    echo "WARNING: verl install may have overwritten CUDA torch. Reinstalling..."
    uv pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124 \
        --reinstall --no-cache-dir
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Install sgl_kernel (pre-built wheel works with stable torch)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Installing sgl_kernel ==="

# Uninstall any existing sgl_kernel first
uv pip uninstall sgl_kernel -y 2>/dev/null || true

# Install sgl_kernel - pre-built wheels work with stable PyTorch
uv pip install sgl_kernel --no-cache-dir

echo "sgl_kernel installed successfully"

# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Install additional dependencies
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Installing additional dependencies ==="

# torch-memory-saver for memory optimization
uv pip install torch-memory-saver --no-cache-dir

# ─────────────────────────────────────────────────────────────────────────────
# Step 7: Final verification
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Final Verification ==="

python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    print('ERROR: CUDA not available!')
    exit(1)
"

echo ""
python -c "import verl; print(f'verl: {verl.__version__}')" || echo "Warning: verl import failed"
python -c "import sglang; print(f'sglang: {sglang.__version__}')" || echo "Warning: sglang import failed"
python -c "import hydra; print('hydra: OK')"
python -c "import ray; print(f'ray: {ray.__version__}')"

# Verify sgl_kernel loads correctly
echo ""
echo "Verifying sgl_kernel..."
python -c "
import sgl_kernel
import os
pkg_dir = os.path.dirname(sgl_kernel.__file__)
print(f'sgl_kernel location: {pkg_dir}')
for item in sorted(os.listdir(pkg_dir)):
    if item.startswith('sm'):
        print(f'  Architecture: {item}')
"

# ─────────────────────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "=== sv2 setup complete (CUDA 12.4) ==="
echo "=========================================="
echo ""
echo "Activate: source $VENV_DIR/bin/activate"
echo ""
echo "Usage:"
echo "  python -m sv2.main_ppo_multiturn_toolcall \\"
echo "    --config-path sv2/config --config-name sv2_multiturn \\"
echo "    data.val_files=\$DATA_DIR/test.parquet"
