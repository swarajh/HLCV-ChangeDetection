#!/usr/bin/env bash

# ==========================================
# SIMSIAM PRETRAINING CONFIGURATION
# ==========================================
# Data and Model
DATA_PATH="datasets/EuroSAT"
MODEL_NAME="swin_small_patch4_window7_224.ms_in22k"

# Hyperparameters
BATCH_SIZE=64
EPOCHS=1
BASE_LR=0.05
NUM_WORKERS=4
TO_SAVE=false

# ==========================================
# PATH DEFINITIONS
# ==========================================
PROJECT_DIR="/home/hlcv_team007/HLCV-ChangeDetection/"
CONDA_PYTHON="/home/hlcv_team007/miniconda3/envs/hlcv/bin/python"
PRETRAIN_SCRIPT="simsiam/train_simsiam.py"

# Navigate to the project directory
cd "$PROJECT_DIR" || { echo "Failed to change directory to $PROJECT_DIR"; exit 1; }

echo "=========================================="
echo "Starting SimSiam Pretraining"
echo "Model Backbone: $MODEL_NAME"
echo "Batch Size: $BATCH_SIZE | Epochs: $EPOCHS"
echo "=========================================="

# Execute the pretraining script
"$CONDA_PYTHON" "$PRETRAIN_SCRIPT" \
    --data_path "$DATA_PATH" \
    --model_name "$MODEL_NAME" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --base_lr "$BASE_LR" \
    --num_workers "$NUM_WORKERS" \
    --to_save "$TO_SAVE"