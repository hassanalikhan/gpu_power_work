#!/bin/bash

# Function to handle Ctrl+C (SIGINT)
handle_ctrl_c() {
  echo "Ctrl+C pressed. Terminating processes..."

  # Terminate get_and_save_power.py
  if [[ -n "$GET_POWER_PID" ]]; then
    kill $GET_POWER_PID 2>/dev/null
    echo "Terminated get_and_save_power.py (PID $GET_POWER_PID)"
  fi

  # Terminate retrain_llama.py
  if [[ -n "$TRAINING_PID" ]]; then
    kill $TRAINING_PID 2>/dev/null
    echo "Terminated retrain_llama.py (PID $TRAINING_PID)"
  fi

  exit 1
}

# Trap Ctrl+C (SIGINT) signal
trap handle_ctrl_c INT

# Full path to Conda
CONDA_PATH="/home/azureuser/miniconda3"
ENV_NAME="llama"

# Ensure Conda is initialized
source "$CONDA_PATH/etc/profile.d/conda.sh"

# Activate the Conda environment
conda activate "$ENV_NAME"

# Install matplotlib if not present
if ! python -c "import matplotlib" &>/dev/null; then
  echo "matplotlib not found. Installing..."
  conda install -y matplotlib
else
  echo "matplotlib already installed."
fi

# Run the first script in the background with proper Conda environment
nohup python get_and_save_power.py > get_power.log 2>&1 &
GET_POWER_PID=$!
echo "Started get_and_save_power.py with PID $GET_POWER_PID"

# Wait for 30 seconds
echo "Waiting 30 seconds before starting training..."
WAIT_TIME=30
while [ $WAIT_TIME -gt 0 ]; do
    echo -e "\tWaiting $WAIT_TIME seconds..."
    sleep 10
    WAIT_TIME=$((WAIT_TIME - 10))
done
echo "Starting training..."

# Start time for training
TRAINING_START=$(date +%s)

# Run the second script (torchrun) with proper Conda environment
echo "Starting distributed training on 8 GPUs..."
nohup torchrun --nproc_per_node=8 retrain_llama.py > training_console.log 2>&1 &
TRAINING_PID=$!
echo "Started training with PID $TRAINING_PID"

# Wait for the training to finish - monitor the process
echo "Waiting for training to complete..."
while kill -0 $TRAINING_PID 2>/dev/null; do
    echo -e "\tTraining still running... ($(date))"
    sleep 60
done
echo "Training process completed"

# End time for training
TRAINING_END=$(date +%s)

# Calculate training time in seconds
TRAINING_DURATION=$((TRAINING_END - TRAINING_START))

# Convert training time to hours, minutes, and seconds
HOURS=$((TRAINING_DURATION / 3600))
MINUTES=$(( (TRAINING_DURATION % 3600) / 60 ))
SECONDS=$((TRAINING_DURATION % 60))

echo "Total training time: ${HOURS} hours, ${MINUTES} minutes, ${SECONDS} seconds"

# Wait 5 more minutes (300 seconds)
echo "Waiting 30 seconds before generating power report..."
WAIT_TIME=70
while [ $WAIT_TIME -gt 0 ]; do
    echo -e "\tWaiting $WAIT_TIME seconds..."
    sleep 10
    WAIT_TIME=$((WAIT_TIME - 10))
done
echo "Running power analysis..."

# Activate the Conda environment again before running draw_power.py
conda activate "$ENV_NAME"

# Debug: Check Python path and installed packages
which python >> debug_log.txt
which pip >> debug_log.txt
python -m pip list | grep matplotlib >> debug_log.txt

# Run draw_power.py using the full path to Python inside the Conda environment
nohup "$CONDA_PATH/envs/$ENV_NAME/bin/python" draw_power.py > power_analysis.log 2>&1 &
echo "Power analysis completed"

sleep 10

# Gracefully terminate the power monitoring process
echo "Stopping power monitoring process..."
if kill -0 $GET_POWER_PID 2>/dev/null; then
    kill $GET_POWER_PID
    echo "Power monitoring process stopped"
else
    echo "Power monitoring process already ended"
fi

echo "All tasks completed successfully"

