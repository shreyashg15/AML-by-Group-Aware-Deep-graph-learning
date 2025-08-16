#!/bin/bash
# Ensure pip is latest
pip install --upgrade pip

# Step 1: Install torch (CPU-only build for Streamlit Cloud)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install PyTorch Geometric stack
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 torch-spline-conv==1.2.2 torch-geometric==2.5.3 --no-cache-dir

# Step 3: Install everything else
pip install -r requirements.txt