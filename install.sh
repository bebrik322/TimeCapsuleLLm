#!/bin/bash

uv venv -p 3.11

source .venv/bin/activate

echo "Installing Triton..."
uv pip install https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1.1/triton-3.5.1%2Brocm7.1.1.gita272dfa8-cp311-cp311-linux_x86_64.whl

echo "Installing PyTorch..."
uv pip install https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1.1/torch-2.9.1%2Brocm7.1.1.lw.git351ff442-cp311-cp311-linux_x86_64.whl

echo "Installing dependencies..."
uv pip install transformers, accelerate

echo "Setup complete. Activate with: source .venv/bin/activate"
