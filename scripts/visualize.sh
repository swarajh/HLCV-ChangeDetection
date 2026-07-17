#!/usr/bin/env bash

MODEL_TYPE="simsiam"           # Options: "mim", "simsiam", "sim", "baseline"

# Path to the test dataset
DATA_PATH="datasets/LEVIR-CD/test"

CHECKPOINT_PATH="checkpoints/vit_base_patch14_dinov2.lvd142m_simsiam_lora_20260716_154943.pth" 

MODEL_NAME="swin"
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