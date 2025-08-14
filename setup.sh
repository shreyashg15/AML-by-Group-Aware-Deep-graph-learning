#!/bin/bash
set -e

echo "=== Installing PyTorch CPU wheels first ==="
pip install torch==2.2.2+cpu torchvision==0.17.2+cpu torchaudio==2.2.2+cpu \
  --index-url https://download.pytorch.org/whl/cpu
# 1. Install torch first (CPU wheels)
pip install torch==2.2.2+cpu --index-url https://download.pytorch.org/whl/cpu

echo "=== Installing PyTorch Geometric deps from PyG wheels ==="
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 \
  -f https://data.pyg.org/whl/torch-2.2.2+cpu.html

echo "=== Installing all remaining packages from requirements.txt ==="
pip install -r requirements.txt
