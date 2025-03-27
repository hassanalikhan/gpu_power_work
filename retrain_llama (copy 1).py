"""
	pip install deepspeed
	pip install transformers[torch]

	Usage: deepspeed --num_gpus=8 test-from-claude-deepspeed.py
"""

import torch
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	TrainingArguments,
	Trainer,
	DataCollatorForLanguageModeling
)
from datasets import load_dataset
from huggingface_hub import login
import deepspeed
import os

os.environ['HF_HOME'] = '/mnt/azureuser/huggingface'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup_model_and_tokenizer(model_name="meta-llama/Llama-2-70b", hf_token=None):
	if hf_token:
		login(token=hf_token)

	# Initialize tokenizer
	tokenizer = AutoTokenizer.from_pretrained(
		model_name,
		token=hf_token
	)
	if tokenizer.pad_token_id is None:
		tokenizer.pad_token = tokenizer.eos_token

	# Load model
	model = AutoModelForCausalLM.from_pretrained(
		model_name,
		token=hf_token,
		torch_dtype=torch.bfloat16,
		cache_dir="/mnt/azureuser/huggingface/hub"
	)

	return model, tokenizer

def prepare_dataset(tokenizer, max_length=512):
	
	# Select dataset (uncomment the desired dataset)
	# dataset = load_dataset("openwebtext")
	# dataset = load_dataset("bookcorpus")
	# dataset = load_dataset("the_pile", split="train")
        #dataset = load_dataset("TigerResearch/pretrain_en")
        dataset = load_dataset('wikitext', "wikitext-2-raw-v1")
        dataset["train"] = dataset["train"].shuffle().select(range(5000))

        def tokenize_function(examples):
                text_column = "content" if "content" in examples else list(examples.keys())[0]
                return tokenizer(
                        examples[text_column],
                        truncation=True,
                        max_length=max_length,
                        padding="max_length"
                )

        tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset["train"].column_names
        )

        return tokenized_dataset


def main():
	HF_TOKEN = "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"
	#MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
	MODEL_NAME = "meta-llama/Llama-2-7b-hf"
	# MODEL_NAME = "meta-llama/Llama-3.1-8B"
	# MODEL_NAME = "NousResearch/Llama-2-7b-chat-hf"
	# MODEL_NAME = "meta-llama/Meta-Llama-3-70B"

	# Define training arguments with DeepSpeed integration
	training_args = TrainingArguments(
		output_dir="/mnt/azureuser/llama-training-output",
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
		eval_strategy="steps",
		bf16=True,
		report_to="none",
		save_total_limit=2,
		# DeepSpeed specific settings
		# deepspeed="ds_config.json",  # Path to DeepSpeed config file
		local_rank=int(os.environ.get("LOCAL_RANK", -1)),
		)


	# Setup model and tokenizer
	model, tokenizer = setup_model_and_tokenizer(MODEL_NAME, HF_TOKEN)

	# Prepare dataset
	dataset = prepare_dataset(tokenizer, max_length=256)
	dataset = dataset["train"].train_test_split(test_size=0.1)

	# Initialize data collator
	data_collator = DataCollatorForLanguageModeling(
		tokenizer=tokenizer,
		mlm=False
	)

	# Initialize trainer
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=dataset["train"],
		eval_dataset=dataset["test"],
		data_collator=data_collator,
	)


if __name__ == "__main__":
	main()

