#!/bin/bash
# Script to execute any training script in Docker
# Usage: ./run_script.sh <script_name> [additional_args]

set -e

SCRIPT_NAME=$1
shift  # Remove first argument, rest are passed to the script

if [ -z "$SCRIPT_NAME" ]; then
    echo "Usage: ./run_script.sh <script_name> [additional_args]"
    echo "Example: ./run_script.sh QPCA_Regression.py"
    echo "Example: ./run_script.sh QPCA_CNN.py --arg1 value1"
    exit 1
fi

# Check if script exists
if [ ! -f "../universal_training_script/$SCRIPT_NAME" ]; then
    echo "Error: Script '$SCRIPT_NAME' not found in universal_training_script folder"
    exit 1
fi

# Build Docker image if not exists
if ! docker images | grep -q "universal-training"; then
    echo "Building Docker image..."
    docker build -t universal-training:latest -f Dockerfile ..
fi

# Run the script in Docker with 24-hour timeout
echo "Executing $SCRIPT_NAME in Docker container..."
echo "Timeout: 24 hours"
echo ""

docker run --rm \
    --name universal-training-run \
    --timeout 86400 \
    -v "$(pwd)/../upload:/app/data:ro" \
    -v "$(pwd)/../output:/app/output:rw" \
    -v "$(pwd)/../universal_training_script:/app/universal_training_script:ro" \
    -w /app \
    universal-training:latest \
    python universal_training_script/$SCRIPT_NAME "$@"

echo ""
echo "Execution completed!"

