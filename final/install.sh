#!/bin/bash
# Minimal installation script for GSM8K evaluation with VERL + SGLang
# Usage: bash install.sh

set -e

echo "=== Installing dependencies ==="

# 1. Install basic dependencies
echo "Installing datasets and tensordict..."
uv pip install datasets tensordict tqdm pyyaml

# 2. Clone and install verl with sglang
echo "Cloning verl repository..."
if [ ! -d "verl" ]; then
    git clone https://github.com/volcengine/verl.git
else
    echo "verl directory already exists, skipping clone"
fi

echo "Installing verl with sglang extras..."
uv pip install -e "./verl[sglang]"

# 3. Optional: wandb for logging (non-fatal if fails)
echo "Installing wandb (optional)..."
uv pip install wandb || echo "wandb install failed, continuing without it"

echo ""
echo "=== Installation complete ==="
echo ""
echo "Usage:"
echo "  python eval_gsm8k.py --num-samples 100 --no-wandb -v"
echo "  python chat.py"
echo "  python test.py --quick"
