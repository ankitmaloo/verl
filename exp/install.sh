#!/bin/bash
# Install script for exp/ using uv
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Creating virtual environment with uv..."
uv venv .venv

echo "Activating venv..."
source .venv/bin/activate

echo "Installing verl with sglang extras..."
uv pip install -e "$VERL_ROOT[sglang]"

echo "Installing PyYAML for config loading..."
uv pip install pyyaml

echo ""
echo "Done! To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "Then run:"
echo "  python algorithm.py"
