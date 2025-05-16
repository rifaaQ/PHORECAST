#!/bin/bash

# Usage: evaluate.sh <path_to_your_data.json> <evaluation_threshold>

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 <path_to_your_data.json> <evaluation_threshold>"
  exit 1
fi

DATA_FILE="$1"
THRESHOLD="$2"

# Evaluate accuracy and similarity 
python evaluate.py "$DATA_FILE" "$THRESHOLD"
# python eval_just_sim.py "$DATA_FILE" "$THRESHOLD"