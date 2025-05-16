from dataclasses import dataclass
from typing import Optional, Dict
import torch 

@dataclass
class BaseTrainingConfig:
    model_name: str = "unsloth/Pixtral-12B-2409"
    load_in_4bit: bool = True #True
    # load_in_8bit: bool = True #False #True
    use_gradient_checkpointing: bool = True
    
    full_finetuning: bool = False

    # LoRA parameters
    finetune_vision_layers: bool = False
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True
    r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0
    bias: str = "none"
    random_state: int = 3407
    use_rslora: bool = False
    loftq_config: Optional[Dict] = None
    
    save_path: str = "./dataset"
    
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    warmup_steps: int = 5
    max_steps: int = None #500 #250, 
    num_train_epochs = 0 # if num_train_epochs > 0, max_steps=None
    learning_rate: float = 2e-4 # 5e-5 #2e-4 #5e-5 #2e-4
    logging_steps: int = 1
    optim: str = "paged_adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 3407
    report_to: str =  "none" #"wandb"
    wandb_project: str = "health_ai"
    wandb_run_name : Optional[str] = None
    wandb_tags: Optional[list] = None
    output_dir: str = "./output_directory"
    # You MUST put the below items for vision finetuning:
    remove_unused_columns = False,
    dataset_text_field = "text", #text
    dataset_kwargs = {"skip_prepare_dataset": True},
    dataset_num_proc = 4,
    max_seq_length = 2048, #512, #1024 #2048,
    # dtype = None #torch.bfloat16,
    
    def get_save_name(self):
        """Generate descriptive save name based on config parameters"""
        model_shortname = self.model_name.split('/')[-1].lower()
                
        params = [
            f"max_seq{self.max_seq_length}",
            f"bs{self.per_device_train_batch_size}",
            f"ga{self.gradient_accumulation_steps}",
            f"lr{self.learning_rate:.1e}",
            f"steps{self.max_steps}",
            f"epochs{self.num_train_epochs}",
            f"r{self.r}",
            f"alpha{self.lora_alpha}",
            
        ]
        
        if self.finetune_vision_layers:
            params.append("vision")
        if self.finetune_language_layers:
            params.append("lang")
        if self.finetune_attention_modules:
            params.append("attn")
        if self.finetune_mlp_modules:
            params.append("mlp")
            
        param_str = "_".join(params)
        return f"newDataset_{model_shortname}_{param_str}"


@dataclass
class PixtralConfig(BaseTrainingConfig):
    model_name: str = "unsloth/Pixtral-12B-2409"
    finetune_attention_modules: bool = False
    per_device_train_batch_size: int = 1  # Lower for larger model
    gradient_accumulation_steps: int = 4

@dataclass
class LLavaConfig(BaseTrainingConfig):
    model_name: str = "unsloth/llava-1.5-7b-hf-bnb-4bit"
    per_device_train_batch_size: int = 1  # Can be higher for smaller model
    gradient_accumulation_steps: int = 4

@dataclass
class LlamaConfig(BaseTrainingConfig):
    model_name: str = "unsloth/Llama-3.2-11B-Vision-Instruct" #"unsloth/Llama-3.2-11B-Vision-bnb-4bit"
    finetune_attention_modules: bool = True
    # per_device_train_batch_size: int = 2
    # gradient_accumulation_steps: int = 4
    # optim = "adamw_8bit",
    r = 16, 
    lora_alpha = 16
    # learning_rate: float = 3e-4  # Slightly higher for Llama

@dataclass
class GemmaConfig(BaseTrainingConfig):
    """Configuration for Gemma models"""
    model_name: str = "unsloth/gemma-3-12b-it" #"unsloth/gemma-3-12b-pt-unsloth-bnb-4bit" #unsloth/gemma-3-4b-it-bnb-4bit" #unsloth/gemma-3-12b-pt-unsloth-bnb-4bit" #unsloth/gemma-3-4b-it-bnb-4bit" #unsloth/gemma-3-12b-pt-unsloth-bnb-4bit" #unsloth/gemma-3-4b-it-bnb-4bit" #unsloth/gemma-3-12b-pt-unsloth-bnb-4bit" #unsloth/gemma-3-4b-it-bnb-4bit" #"unsloth/gemma-3-4b-it" #"unsloth/gemma-3-4b-it-unsloth-bnb-4bit" #"unsloth/gemma-3-4b-it" #"unsloth/gemma-7b"
    # per_device_train_batch_size: int = 2
    # gradient_accumulation_steps: int = 4
    # per_device_train_batch_size: int = 1
    # gradient_accumulation_steps: int = 2
    finetune_attention_modules: bool = True,
    
@dataclass 
class LlavaNextConfig(BaseTrainingConfig):
    """Configuration for LlavaNext models"""
    model_name: str = "unsloth/llava-1.5-7b-hf-bnb-4bit" #"unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit"
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    finetune_attention_modules: bool = True
    # full_finetuning = False,
    
@dataclass
class QwenConfig(BaseTrainingConfig):
    """Configuration for Qwen models"""
    model_name: str = "unsloth/Qwen2-VL-7B-Instruct"
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    finetune_attention_modules: bool = True
    r = 16
    lora_alpha = 16
    lora_dropout = 0
    
    # full_finetuning = False,
    
# Dictionary mapping for easy access
MODEL_CONFIGS = {
    "pixtral": PixtralConfig,
    "llava": LLavaConfig,
    "qwen": QwenConfig,
    "llama": LlamaConfig,
    "gemma": GemmaConfig,
}

def get_config(model_type: str = "pixtral", **kwargs):
    """Factory function to get appropriate config"""
    config_class = MODEL_CONFIGS.get(model_type.lower(), PixtralConfig)
    return config_class(**kwargs)
