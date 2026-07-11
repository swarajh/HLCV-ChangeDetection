#!/bin/bash

# ==========================================
# PATH DEFINITIONS
# ==========================================
PROJECT_DIR="/home/hlcv_team007/HLCV-ChangeDetection/"
CONDA_PYTHON="/home/hlcv_team007/miniconda3/envs/hlcv/bin/python"
TRAIN_SCRIPT="change_detection/ssl.py"

# ==========================================
# ARGUMENTS
# ==========================================
FEATURES_PATH="features/precomputed_features.pt"  # Path to the saved .pt file
METHOD="simsiam"  # Choose between "simsiam" or "mim"
BATCH_SIZE=512  # Batch size for training
EPOCHS=200  # Number of epochs
BASE_LR=0.05  # Base learning rate
DATA_PATH="datasets/EuroSAT"  # Path to the dataset for feature extraction
MODEL_NAME="swin_tiny_patch4_window7_224"  # Backbone model name
OUTPUT_DIR="features"  # Directory to save extracted features

# ==========================================
# NAVIGATE TO PROJECT DIRECTORY
# ==========================================
cd "$PROJECT_DIR" || { echo "Failed to change directory to $PROJECT_DIR"; exit 1; }

# ==========================================
# RUN THE SCRIPT
# ==========================================
"$CONDA_PYTHON" "$TRAIN_SCRIPT" \
    --features_path "$FEATURES_PATH" \
    --method "$METHOD" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --base_lr "$BASE_LR" \
    --data_path "$DATA_PATH" \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR"