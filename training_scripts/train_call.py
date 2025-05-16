# run_training.py

import os
import argparse
from trainer_config import get_config
from training import train_model

def main():
    parser = argparse.ArgumentParser(description="Run model fine-tuning.")
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model name to use (e.g., 'gemma', 'pixtral')"
    )
    parser.add_argument(
        "--save_path", type=str, required=True,
        help="Path to the training dataset"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save model checkpoints"
    )

    args = parser.parse_args()

    config = get_config(
        args.model,
        max_steps=12000,
        warmup_steps=30,
        save_path=args.save_path,
        output_dir=args.output_dir,
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        r=8,
        lora_alpha=8,
        report_to="wandb",
    )

    # Start training
    train_stats = train_model(config)
    print("Training complete.")

if __name__ == "__main__":
    main()
