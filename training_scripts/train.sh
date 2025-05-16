#!/bin/bash
# Usage: ./run.sh gemma /path/to/data /path/to/output

MODEL=$1
SAVE_PATH=$2
OUTPUT_DIR=$3

export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HOME="/fs/nexus-projects/health_ai/src/cache"

python run_training.py --model "$MODEL" --save_path "$SAVE_PATH" --output_dir "$OUTPUT_DIR"
