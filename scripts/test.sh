#!/usr/bin/env bash

# ==========================================
# EVALUATION CONFIGURATION
# ==========================================
MODEL_TYPE="baseline"           # Options: "mim", "simsiam", "sim", "baseline"
BATCH_SIZE=1

# Path to the test dataset
DATA_PATH="datasets/LEVIR-CD/test"

# Optional: If you want to test a specific checkpoint, set the path below.
# If left empty (""), the python script will automatically fetch the best one from the CSV!
CHECKPOINT_PATH="" 
# Example: CHECKPOINT_PATH="checkpoints/swin_baseline_20231024_120000.pth"
# ==========================================

# Path definitions
PROJECT_DIR="/home/hlcv_team007/HLCV-ChangeDetection/"
CONDA_PYTHON="/home/hlcv_team007/miniconda3/envs/hlcv/bin/python"
EVAL_SCRIPT="change_detection/evaluate.py" 

# Navigate to the project directory
cd "$PROJECT_DIR" || { echo "Failed to change directory to $PROJECT_DIR"; exit 1; }

echo "Starting evaluation for model: $MODEL_TYPE"
echo "Batch Size: $BATCH_SIZE | Data Path: $DATA_PATH"

CMD="$CONDA_PYTHON $EVAL_SCRIPT --model_type $MODEL_TYPE --data_path $DATA_PATH --batch_size $BATCH_SIZE"

# Add the checkpoint argument if it was provided
if [ -n "$CHECKPOINT_PATH" ]; then
    echo "Using specific checkpoint: $CHECKPOINT_PATH"
    CMD="$CMD --checkpoint $CHECKPOINT_PATH"
else
    echo "Auto-fetching best checkpoint from CSV..."
fi

$CMD