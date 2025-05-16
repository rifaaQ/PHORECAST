import torch
from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch
from datasets import load_dataset, load_from_disk
from PIL import Image

from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
import wandb
from transformers.integrations import WandbCallback
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import standardize_data_formats
from unsloth.chat_templates import train_on_responses_only

from unsloth import is_bf16_supported

# print(torch.cuda.is_available()) 

def get_dataset(save_path,filter_imgs = True):
    dataset = load_from_disk(save_path)
    if filter_imgs:
        filtered_dataset = dataset.filter(lambda example: not example['media_path'].endswith('.mp4')) #media_path
        return filtered_dataset
    return dataset

def train_model(args):
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or args.get_save_name(),
        tags=args.wandb_tags,
        config=args  
    )
    model, tokenizer = FastVisionModel.from_pretrained(
        args.model_name, 
        load_in_4bit=args.load_in_4bit,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
    )
    
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=args.finetune_vision_layers,
        finetune_language_layers=args.finetune_language_layers,
        finetune_attention_modules=args.finetune_attention_modules,
        finetune_mlp_modules=args.finetune_mlp_modules,
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.bias,
        random_state=args.random_state,
        use_rslora=args.use_rslora,
        loftq_config=args.loftq_config,
        # dtype = args.dtype, # loading model without + activate the bf16 mixed precision training
    )
    print(model)
    FastVisionModel.for_training(model)
    
    dataset = get_dataset(args.save_path, filter_imgs=False)
    # tokenizer = get_chat_template(tokenizer)
    
    if args.model_name == "unsloth/gemma-3-4b-it-bnb-4bit" or args.model_name == "unsloth/gemma-3-12b-pt-unsloth-bnb-4bit": #"unsloth/gemma-3-4b-it":
        tokenizer = get_chat_template(
            tokenizer,
            chat_template = "gemma-3",
        )
        print(dataset)
    
    
    callbacks = [WandbCallback] if args.report_to == "wandb" else []
    
    if args.num_train_epochs > 0:
        args.max_steps = 0 #None
    elif args.max_steps == 0:
        args.num_train_epochs = 1 #None
        
    training_args = SFTConfig( 
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        num_train_epochs = args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16= not args.load_in_4bit, 
        bf16= args.load_in_4bit, 
        logging_steps=args.logging_steps,
        optim=args.optim,
        weight_decay=args.weight_decay,
        lr_scheduler_type="linear", #args.lr_scheduler_type,
        seed=3407,
        report_to=args.report_to,
        save_strategy = "steps",
        save_steps = 0.1 * args.max_steps,
        
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 4,
        max_seq_length = 2048,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        callbacks=callbacks,
        train_dataset=dataset,
        args=training_args,
    )


    #@title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    
    trainer_stats = trainer.train()
    
    wandb.log({
        "final/train_loss": trainer_stats.metrics["train_loss"],
        "final/train_runtime": trainer_stats.metrics["train_runtime"],
        "final/train_samples_per_second": trainer_stats.metrics["train_samples_per_second"],
    })
    
    #@title Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
    
    save_name = args.get_save_name()
    
    model.save_pretrained(save_name)
    tokenizer.save_pretrained(save_name)
    print(f"Model saved to {save_name}")
    
    wandb.finish()
    return trainer_stats




