import os
import json
import torch
from PIL import Image
import pandas as pd
from tqdm import tqdm
from typing import Dict, List
from datasets import load_from_disk
from unsloth import FastVisionModel
from unsloth.chat_templates import get_chat_template


class ModelEvaluator:
    def __init__(self, model_dirs: List[str], dataset_path: str):
        self.models = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        for model_dir in model_dirs:
            try:
                model_name = os.path.basename(model_dir)
                print(f"Loading model: {model_name}")
                model, tokenizer = FastVisionModel.from_pretrained(
                    model_dir, load_in_4bit=False
                )
                FastVisionModel.for_inference(model)
                self.models[model_name] = {
                    'model': model.to(self.device),
                    'tokenizer': tokenizer
                }
            except Exception as e:
                print(f"Error loading {model_dir}: {e}")
                import traceback; traceback.print_exc()

        self.dataset = load_from_disk(dataset_path)
        if 'test' in self.dataset:
            self.dataset = self.dataset['test']

        self.dataset = self.dataset.filter(lambda ex: not ex['images'].endswith('.mp4'))

        print(f"Dataset loaded with {len(self.dataset)} samples")

    def process_sample(self, sample: Dict) -> Dict:
        try:
            user_content = sample['conversations'][0]['content']
            instruction = next(
                item["text"] for item in user_content 
                if item.get("type") == "text" and item.get("text")
            )

            assistant_content = sample['conversations'][1]['content']
            expected_output = next(
                item["text"] for item in assistant_content 
                if item.get("type") == "text"
            )

            image_path = sample['images']
            image = Image.open(image_path).convert('RGB') if not isinstance(image_path, Image.Image) else image_path

            return {
                'instruction': instruction,
                'expected_output': expected_output,
                'image': image.convert("RGB")
            }
        except Exception as e:
            print(f"Error processing sample: {e}")

    def generate_response(self, model_name: str, instruction: str, image: Image) -> Dict:
        try:
            model_data = self.models[model_name]
            tokenizer = model_data['tokenizer']
            model = model_data['model']

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction}
                ]
            }]

            input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            inputs = tokenizer(images=image, text=input_text, return_tensors="pt").to(self.device)

            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']

            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                use_cache=False,
                temperature=1.5,
                min_p=0.1
            )

            generated_text = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

            return {
                "model": model_name,
                "response": generated_text.strip()
            }
        except Exception as e:
            return {
                "model": model_name,
                "response": f"Generation Error: {e}"
            }

    def evaluate(self, num_samples: int = None, save_path: str = None, output_dir: str = None) -> pd.DataFrame:
        num_samples = min(num_samples or len(self.dataset), len(self.dataset))
        results = []

        for i in tqdm(range(num_samples)):
            sample = self.dataset[i]
            processed = self.process_sample(sample)

            if processed['image'] is None:
                continue

            for model_name in self.models:
                gen = self.generate_response(model_name, processed['instruction'], processed['image'])
                results.append({
                    "sample_id": i,
                    "model": model_name,
                    "instruction": processed['instruction'],
                    "expected_output": processed['expected_output'],
                    "response": gen['response']
                })

        df = pd.DataFrame(results)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = save_path or os.path.join(output_dir, "results.json")

        if save_path:
            df.to_json(save_path, orient='records', indent=2)
            print(f"Results saved to {save_path}")

        return df
