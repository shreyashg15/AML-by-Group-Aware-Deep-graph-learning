#!/bin/bash
set -e

# 1. Install torch first (CPU wheels)
pip install torch==2.2.2+cpu --index-url https://download.pytorch.org/whl/cpu

# 2. Install torch-scatter, torch-sparse, torch-cluster from PyG wheels
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 \
  -f https://data.pyg.org/whl/torch-2.2.2+cpu.html

# 3. Install everything else from requirements.txt
pip install -r requirements.txt
