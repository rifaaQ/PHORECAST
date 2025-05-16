#!/bin/bash

# ----------- CONFIGURABLE PARAMETERS -----------
MODEL_DIRS=("path/to/model1" "path/to/model2")  # Update with real paths
DATASET_PATH="path/to/huggingface/dataset"     # e.g., "/fs/.../dataset"
NUM_SAMPLES=10                                  # Set to -1 to evaluate the whole dataset

# ----------- ENVIRONMENT VARIABLES -----------
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HOME="${HF_HOME:-/tmp/hf_cache}"  # fallback to /tmp if not set

# ----------- PYTHON ENV CHECK -----------
if ! command -v python &> /dev/null
then
    echo "Python not found. Please install Python 3.9+."
    exit 1
fi

# ----------- OPTIONAL: Create a venv -----------
# python -m venv venv
# source venv/bin/activate
# pip install -r requirements.txt

# ----------- RUN THE SCRIPT -----------
echo "Starting evaluation..."

python <<EOF
from pathlib import Path
from model_eval import ModelEvaluator

model_dirs = ${MODEL_DIRS[@]}
dataset_path = "${DATASET_PATH}"
num_samples = ${NUM_SAMPLES if NUM_SAMPLES >= 0 else 'None'}

evaluator = ModelEvaluator(model_dirs=model_dirs, dataset_path=dataset_path)
results_df = evaluator.evaluate(num_samples=num_samples)

# Save results
output_file = Path("results.csv")
results_df.to_csv(output_file, index=False)
print(f"Saved results to {output_file}")
EOF
