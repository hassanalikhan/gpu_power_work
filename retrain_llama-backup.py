import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import os
import numpy as np
import random

# Ensure environment variables are set
os.environ["TOKENIZERS_PARALLELISM"] = "true"  # Disable tokenizer parallelism to avoid warnings

def retrain_tiny_llama(
		model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
		dataset_name="c4", 
		dataset_config="realnewslike", 
		output_dir="./tiny_llama_retrained", 
		epochs=1
):
		"""
		Downloads a small Llama model from Hugging Face, retrains it on a selected dataset, and saves the retrained model.
		"""
		# Check GPU setup
		gpu_count = torch.cuda.device_count()
		print(f"Number of GPUs available: {gpu_count}")

		# 1. Model and Tokenizer
		tokenizer = AutoTokenizer.from_pretrained(model_name)
		if tokenizer.pad_token is None:
				tokenizer.pad_token = tokenizer.eos_token  # Add pad token
		
		model = AutoModelForCausalLM.from_pretrained(model_name)
		
		# Move model to GPU if available
		device = "cuda:0" if torch.cuda.is_available() else "cpu"
		print(f"Initial model loading device: {device}")
		model.to(device)
		model.train()
		
		# Unfreeze all layers for fine-tuning
		print("Unfreezing all layers...")
		for param in model.parameters():
				param.requires_grad = True
		
		trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
		print(f"Total trainable parameters: {trainable_params:,}")

		reduction_factor = 5
		# 2. Dataset selection
		if dataset_name == "c4":
			dataset = load_dataset("allenai/c4", "realnewslike")
		elif dataset_name == "openwebtext":
			dataset = load_dataset("openwebtext")
		elif dataset_name == "bookcorpus":
			dataset = load_dataset("bookcorpus")
		elif dataset_name == "pile":
			dataset = load_dataset("the_pile", split="train")
		elif dataset_name == "wikitext":
			dataset = load_dataset(dataset_name, "wikitext-2-raw-v1")
			reduction_factor = 1
		else:
			raise ValueError(f"Dataset {dataset_name} not supported!")

		train_dataset = dataset["train"].shuffle().select(range(1000))

		# 3. Process the dataset
		def preprocess_function(examples):
				"""Tokenize and prepare inputs for the model"""
				texts = [text for text in examples["text"] if text and len(text.strip()) > 0]
				if not texts:
					return {"input_ids": [], "attention_mask": [], "labels": []}

				model_inputs = tokenizer(
							texts,
							truncation=True,
							padding="max_length",
							truncation=True,
							max_length=256,
							return_tensors=None
				)

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

		# 4. Training Arguments
		training_args = TrainingArguments(
				output_dir=output_dir,
				overwrite_output_dir=True,
				per_device_train_batch_size=1,
				gradient_accumulation_steps=4,
				num_train_epochs=3,
				learning_rate=1e-5,
				weight_decay=0.01,
				warmup_steps=1000,
				logging_dir="./logs",
				logging_steps=10,
				save_steps=1000,
				eval_steps=1000,
				evaluation_strategy="steps",
				bf16=True,
				report_to="none",
				save_total_limit=2,
				# DeepSpeed specific settings
				deepspeed="ds_config.json",  # Path to DeepSpeed config file
				local_rank=int(os.environ.get("LOCAL_RANK", -1)),
				)

		# 5. Trainer
		trainer = Trainer(
				model=model,
				args=training_args,
				train_dataset=tokenized_dataset,
		)

		# 6. Train
		effective_batch = training_args.per_device_train_batch_size * gpu_count * training_args.gradient_accumulation_steps
		print(f"Starting training with effective batch size: {effective_batch}")

		try:
				trainer.train()
				print("Training completed successfully!")
		except Exception as e:
				print(f"Error during training: {e}")
				import traceback
				traceback.print_exc()

		# 7. Save the model
		try:
				print(f"Saving model to {output_dir}...")
				model.save_pretrained(output_dir)
				tokenizer.save_pretrained(output_dir)
				print("Model saved successfully!")
		except Exception as e:
				print(f"Error saving model: {e}")

# Direct script execution
if __name__ == "__main__":

		# TinyLlama/TinyLlama-1.1B-Chat-v1.0
		# meta-llama/Llama-2-7b-hf
		# meta-llama/Llama-3.1-8B
		# NousResearch/Llama-2-7b-chat-hf
		
		# wikitext
		# allenai/c4
		# pile
		# book corpus
		# openwebtext
		
		retrain_tiny_llama(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", dataset_name="wikitext")

