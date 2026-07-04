#!/usr/bin/env bash

# ==========================================
# EXPERIMENT CONFIGURATION
# ==========================================
MODEL_TYPE="simsiam"           # Options: "mim", "simsiam", "baseline"
BATCH_SIZE=4
EPOCHS=30

TRAIN_PATH="datasets/LEVIR-CD/train"
VAL_PATH="datasets/LEVIR-CD/val"
# ==========================================

# Path definitions
PROJECT_DIR="/home/hlcv_team007/HLCV-ChangeDetection/"
CONDA_PYTHON="/home/hlcv_team007/miniconda3/envs/hlcv/bin/python"
TRAIN_SCRIPT="change_detection/train.py"

# Navigate to the project directory
cd "$PROJECT_DIR" || { echo "Failed to change directory to $PROJECT_DIR"; exit 1; }

echo "Starting training for model: $MODEL_TYPE"
echo "Batch Size: $BATCH_SIZE | Epochs: $EPOCHS"

# Execute the training script using the variables defined above
"$CONDA_PYTHON" "$TRAIN_SCRIPT" \
    --model_type "$MODEL_TYPE" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --train_path "$TRAIN_PATH" \
    --val_path "$VAL_PATH"