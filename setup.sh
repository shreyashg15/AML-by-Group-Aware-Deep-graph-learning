#!/bin/bash
# Install Python packages
pip install torch==2.2.2+cpu torchvision==0.17.2+cpu torchaudio==2.2.2+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
