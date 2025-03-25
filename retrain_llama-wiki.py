import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import os
import numpy as np
import threading
import time


# Ensure environment variables are set
os.environ["TOKENIZERS_PARALLELISM"] = "true"  # Disable tokenizer parallelism to avoid warnings


def cleanup_checkpoints(checkpoint_dir, max_checkpoints=10, interval=3600):
    """
    Periodically clean up old checkpoints to prevent disk space overflow.
    
    :param checkpoint_dir: Directory containing model checkpoints
    :param max_checkpoints: Maximum number of recent checkpoints to keep
    :param interval: Time between cleanup runs (in seconds)
    """
    while True:
        try:
            # Get all checkpoint directories sorted by creation time
            checkpoints = [
                os.path.join(checkpoint_dir, d) 
                for d in os.listdir(checkpoint_dir) 
                if os.path.isdir(os.path.join(checkpoint_dir, d)) and d.startswith("checkpoint-")
            ]
            
            # Sort checkpoints by creation time (newest first)
            checkpoints.sort(key=lambda x: os.path.getctime(x), reverse=True)
            
            # Remove older checkpoints
            if len(checkpoints) > max_checkpoints:
                for checkpoint in checkpoints[max_checkpoints:]:
                    try:
                        print(f"Removing old checkpoint: {checkpoint}")
                        shutil.rmtree(checkpoint)
                    except Exception as e:
                        print(f"Error removing checkpoint {checkpoint}: {e}")
            
            # Wait before next cleanup
            time.sleep(interval)
        
        except Exception as e:
            print(f"Checkpoint cleanup error: {e}")
            time.sleep(interval)
            
def retrain_tiny_llama(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                      dataset_name="wikitext", 
                      dataset_config="wikitext-2-raw-v1", 
                      output_dir="./tiny_llama_retrained", 
                      epochs=1):
    """
    Downloads a small Llama model from Hugging Face, retrains it on an online dataset, and saves the retrained model.
    """
    # Check GPU setup
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs available: {gpu_count}")
    
    # 1. Model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Add pad token
    
    # Load model for single GPU first (we'll use Trainer's DDP for multi-GPU)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Move model to first GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Initial model loading device: {device}")
    model.to(device)
    model.train()
    
    # Unfreeze all layers
    print("Unfreezing all layers...")
    for param in model.parameters():
        param.requires_grad = True
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {trainable_params:,}")
    
    # 2. Dataset
    dataset = load_dataset(dataset_name, dataset_config)
    train_dataset = dataset["train"]
    
    # 3. Process the dataset
    def preprocess_function(examples):
        """Tokenize and prepare inputs for the model"""
        # Filter empty strings
        texts = [text for text in examples["text"] if text and len(text.strip()) > 0]
        if not texts:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        
        # Tokenize inputs
        model_inputs = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors=None  # Return lists, not tensors
        )
        
        # Set up the labels
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        return model_inputs
    
    # Apply preprocessing
    tokenized_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Preprocessing dataset",
    )
    
    # Filter empty examples
    def is_not_empty(example):
        return len(example["input_ids"]) > 0
    
    tokenized_dataset = tokenized_dataset.filter(is_not_empty)
    print(f"Dataset size after preprocessing: {len(tokenized_dataset)}")
    
    # 4. Training Arguments - using proper distributed training
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,  # Start conservatively
        gradient_accumulation_steps=1,
        save_steps=1000,
        save_total_limit=5,
        logging_dir="./logs",
        logging_steps=50,
        
        # Distributed training settings
        ddp_backend="nccl",  # Use NCCL for GPU communication
        fp16=True,           # Use mixed precision
        
        # Disable DataParallel
        remove_unused_columns=True,
        
        # Avoid hanging
        ddp_timeout=7200,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )
    
    # 5. Trainer - let the Trainer handle DDP
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    # Start checkpoint cleanup thread
    cleanup_thread = threading.Thread(
        target=cleanup_checkpoints, 
        args=(output_dir, 1, 30),  # Keep 1 most recent checkpoints, run every 30 seconds
        daemon=True  # Allows thread to be killed when main program exits
    )
    cleanup_thread.start()
    
    # 6. Train
    effective_batch = training_args.per_device_train_batch_size * gpu_count * training_args.gradient_accumulation_steps
    print(f"Starting training with effective batch size: {effective_batch}")
    
    # Use try-except to better catch errors
    try:
        trainer.train()
        print("Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        
        # Additional debugging info
        if "chunk expects" in str(e):
            print("\nDebugging the chunking error:")
            print("This is likely due to tensor dimensionality issues in the DataParallel scatter operation.")
            print("Try running with TORCH_DISTRIBUTED_DEBUG=INFO for more information.")
    
    # 7. Save if training was successful
    try:
        print(f"Saving model to {output_dir}...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("Model saved successfully!")
    except Exception as e:
        print(f"Error saving model: {e}")

# Direct script execution
if __name__ == "__main__":
    # Set up distributed environment variables if needed
    if "RANK" not in os.environ and "WORLD_SIZE" not in os.environ:
        print("Setting up distributed environment manually...")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"  # Use all GPUs
    
    retrain_tiny_llama()
