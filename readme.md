chmod +x run_llama_training.sh
./run_llama_training.sh

# Shell Script Explanation: Distributed Training with Power Monitoring

## Overview

The script performs the following steps:

1.  **Environment Setup:** Initializes the conda environment and activates the required environment.
2.  **Power Monitoring:** Starts a background process to monitor and save power consumption data.
3.  **Training Preparation:** Waits for a specified time before starting the training process.
4.  **Distributed Training:** Launches a distributed training job using `torchrun` across 8 GPUs.
5.  **Training Completion:** Waits for the training process to finish and records the total training time.
6.  **Power Analysis Preparation:** Waits for a specified time before generating the power report.
7.  **Power Analysis:** Runs a Python script to analyze the power consumption data.
8.  **Process Termination:** Gracefully terminates the power monitoring process.
9.  **Ctrl+C Handling:** Implements a signal handler to terminate all running processes if Ctrl+C is pressed.
10. **Console Logging:** Redirects the training script console output into a log file.

## Detailed Explanation

### 1. Environment Setup

```bash
CONDA_PATH="/home/azureuser/miniconda3"
eval "$($CONDA_PATH/bin/conda shell.bash hook)"
conda activate llama

## Models and Datasets
# TinyLlama/TinyLlama-1.1B-Chat-v1.0
                # meta-llama/Llama-2-7b-hf
                # meta-llama/Llama-3.1-8B
                # NousResearch/Llama-2-7b-chat-hf

                # wikitext
                # c4
                # pile
                # book corpus
                # openwebtext
