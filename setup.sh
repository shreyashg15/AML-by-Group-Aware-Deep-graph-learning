#!/bin/bash
set -e

echo "Installing torch first..."
pip install torch==2.2.2+cpu torchvision==0.17.2+cpu torchaudio==2.2.2+cpu \
  --index-url https://download.pytorch.org/whl/cpu

echo "Installing PyG dependencies..."
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 \
  -f https://data.pyg.org/whl/torch-2.2.2+cpu.html

echo "Installing remaining Python dependencies..."
pip install -r requirements.txt
