import threading
import time
import csv
import os
from datetime import datetime
import pynvml

# Initialize NVIDIA Management Library
pynvml.nvmlInit()

# Get the number of GPUs
num_gpus = pynvml.nvmlDeviceGetCount()

# Global variables
gpu_data = []
sampling_interval = 0.05
data_duration = 60
lock = threading.Lock()
stop_event = threading.Event()  # Stop signal

import time
from datetime import datetime
import os
import shutil

def remove_files_and_dirs(directory):
    """
    Removes all files and directories within the specified directory, if the directory exists.

    Args:
        directory (str): The path to the directory.
    """
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove files and symbolic links
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directories recursively
            except Exception as e:
                print(f"Failed to remove {file_path}. Reason: {e}")
        print(f"Contents of '{directory}' removed successfully.")
    else:
        print(f"Directory '{directory}' does not exist.")

def print_time_every_minute():
    """Prints the current time every minute."""
    while True:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Current time: {current_time}")

        # Wait for the next minute
        time.sleep(data_duration)  # Sleep for 60 seconds

def get_gpu_power():
    """Fetches power draw of all GPUs with timestamps."""
    timestamp = time.time()
    power_draws = []


    for i in range(num_gpus):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
        power_draws.append((i, power_draw, timestamp))

    return power_draws


def collect_gpu_data():
    """Collects GPU power draw data every 30ms and stores it in a list."""
    global gpu_data

    while not stop_event.is_set():  # Stop when event is set
        start_time = time.time()

        # Fetch GPU power data
        data = get_gpu_power()

        # Append data with thread safety
        with lock:
            gpu_data.extend(data)

        # Sleep for remaining time in interval
        elapsed_time = time.time() - start_time
        sleep_time = max(0, sampling_interval - elapsed_time)
        stop_event.wait(sleep_time)  # Allows stopping during wait


def save_data():
    """Periodically saves GPU power draw data to a file every 10 minutes."""
    global gpu_data


    while not stop_event.is_set():  # Stop when event is set
        if stop_event.wait(data_duration):
            break  # Exit if stop signal is set

        # Copy and clear data safely
        with lock:
            data_to_save = gpu_data[:]
            gpu_data = []

        if data_to_save:
            # Create directory for logs
            os.makedirs("gpu_logs", exist_ok=True)

            # Generate timestamped filename
            start_time = datetime.fromtimestamp(data_to_save[0][2]).strftime("%Y-%m-%d_%H-%M-%S")
            end_time = datetime.fromtimestamp(data_to_save[-1][2]).strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"gpu_logs/gpu_power_{start_time}_to_{end_time}.csv"

            # Save to CSV
            with open(filename, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["GPU_ID", "Power_Usage_W", "Timestamp"])
                writer.writerows(data_to_save)

            print(f"\tSaved {len(data_to_save)} records to {filename}")

remove_files_and_dirs('tiny_llama_retrained')
remove_files_and_dirs('gpu_logs')
# Start data collection thread
collector_thread = threading.Thread(target=collect_gpu_data, daemon=True)
print('GPU Power Data Collection - Starting Thread')
collector_thread.start()

# Start data saving thread
saver_thread = threading.Thread(target=save_data, daemon=True)
print('GPU Power Data Saving - Starting Thread')
saver_thread.start()

time_thread = threading.Thread(target=print_time_every_minute, daemon=True)
time_thread.start()


# Keep main thread alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopping GPU monitoring...")

    # Signal threads to stop
    stop_event.set()

    # Wait for threads to exit
    collector_thread.join()
    saver_thread.join()

    # Shutdown NVML
    pynvml.nvmlShutdown()
    print("GPU monitoring stopped successfully.")

