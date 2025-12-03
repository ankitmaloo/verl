#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

check_pass() { echo -e "${GREEN}✓${NC} $1"; }
check_fail() { echo -e "${RED}✗${NC} $1"; exit 1; }
check_warn() { echo -e "${YELLOW}!${NC} $1"; }

echo "=== Pre-flight checks ==="

# Python
if command -v python3 &> /dev/null; then
    PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
        check_pass "Python $PY_VERSION"
    else
        check_fail "Python 3.10+ required (found $PY_VERSION)"
    fi
else
    check_fail "Python3 not found"
fi

# uv
if command -v uv &> /dev/null; then
    check_pass "uv $(uv --version 2>/dev/null | head -1)"
else
    check_warn "uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    check_pass "GPU: $GPU_NAME (driver $DRIVER_VERSION)"
else
    check_warn "nvidia-smi not found - CPU only mode"
fi

# CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    check_pass "CUDA $CUDA_VERSION"
elif [ -d "/usr/local/cuda" ]; then
    check_pass "CUDA found at /usr/local/cuda"
else
    check_warn "CUDA not found - will use CPU or rely on PyTorch CUDA"
fi

# Git
if command -v git &> /dev/null; then
    check_pass "git $(git --version | cut -d' ' -f3)"
else
    check_fail "git not found"
fi

echo ""
echo "=== Setup ==="

# Clone verl if not present
VERL_DIR="$SCRIPT_DIR/verl"
if [ -d "$VERL_DIR" ]; then
    check_pass "verl repo exists"
else
    echo "Cloning verl..."
    git clone https://github.com/volcengine/verl.git "$VERL_DIR"
    check_pass "verl cloned"
fi

# Create venv
if [ -d ".venv" ]; then
    check_pass "venv exists"
else
    echo "Creating venv..."
    uv venv .venv
    check_pass "venv created"
fi

echo "Activating venv..."
source .venv/bin/activate

echo "Installing verl..."
uv pip install -e "$VERL_DIR[sglang]"

echo "Installing dependencies..."
uv pip install pyyaml wandb huggingface_hub datasets

# Verify imports
echo ""
echo "=== Verifying installation ==="

VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

$VENV_PYTHON -c "import torch; print(f'PyTorch {torch.__version__}')" && check_pass "torch" || check_fail "torch import failed"
$VENV_PYTHON -c "import torch; assert torch.cuda.is_available(), 'no cuda'; print(f'CUDA devices: {torch.cuda.device_count()}')" && check_pass "torch.cuda" || check_warn "CUDA not available in PyTorch"
$VENV_PYTHON -c "import verl; print('verl ok')" && check_pass "verl" || check_fail "verl import failed"
$VENV_PYTHON -c "import sglang; print('sglang ok')" && check_pass "sglang" || check_warn "sglang import failed"
$VENV_PYTHON -c "import wandb; print('wandb ok')" && check_pass "wandb" || check_fail "wandb import failed"

echo ""
echo "=== Done ==="
echo "Activate: source .venv/bin/activate"
echo "Run: python algorithm.py"
