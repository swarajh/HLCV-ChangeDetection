#!/usr/bin/env bash

MODEL_TYPE="simsiam"           # Options: "mim", "simsiam", "sim", "baseline"

# Path to the test dataset
DATA_PATH="datasets/LEVIR-CD/test"

CHECKPOINT_PATH="checkpoints/swin_large_patch4_window7_224_simsiam_projector_20260714_074216.pth" 

MODEL_NAME="swin_large_patch4_window7_224"
# Path definitions
PROJECT_DIR="/home/hlcv_team007/HLCV-ChangeDetection/"
CONDA_PYTHON="/home/hlcv_team007/miniconda3/envs/hlcv/bin/python"
EVAL_SCRIPT="change_detection/visualize.py" 

# Navigate to the project directory
cd "$PROJECT_DIR" || { echo "Failed to change directory to $PROJECT_DIR"; exit 1; }

echo "Starting evaluation for model: $MODEL_TYPE"
echo "Batch Size: $BATCH_SIZE | Data Path: $DATA_PATH"

CMD="$CONDA_PYTHON $EVAL_SCRIPT --model_type $MODEL_TYPE --data_path $DATA_PATH --model_name $MODEL_NAME --checkpoint $CHECKPOINT_PATH"


$CMD