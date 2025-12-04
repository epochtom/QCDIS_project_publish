#!/bin/bash
# Wrapper script to run training script with custom dataset path
# Usage: ./run_training_script.sh <script_name> <dataset_path>

SCRIPT_NAME=$1
DATASET_PATH=$2

if [ -z "$SCRIPT_NAME" ] || [ -z "$DATASET_PATH" ]; then
    echo "Usage: ./run_training_script.sh <script_name> <dataset_path>"
    exit 1
fi

# Export dataset path as environment variable
export DATASET_PATH=$DATASET_PATH

# Run the script
python universal_training_script/$SCRIPT_NAME

