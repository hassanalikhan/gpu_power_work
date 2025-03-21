import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates  # Import for better date formatting

# Directory where GPU log files are saved
log_dir = "gpu_logs"
# Directory where graphs will be saved
graph_dir = "gpu_graphs"

def plot_combined_gpu_power():
    """Generate and save a combined line plot for GPU power usage from all log files,
       replacing values less than 50 with a moving average."""
    all_data = []

    # Loop through all CSV files in the logs directory
    ldf = list(os.listdir(log_dir))
    ldf.sort()
    for log_file in ldf:
        if log_file.endswith(".csv"):
            log_file_path = os.path.join(log_dir, log_file)

            if os.path.getsize(log_file_path) == 0:
                print(f"Skipping empty file: {log_file}")
                continue

            try:
                df = pd.read_csv(log_file_path)

                if df.empty or not set(['GPU_ID', 'Power_Usage_W', 'Timestamp']).issubset(df.columns):
                    print(f"Skipping invalid file (missing expected columns): {log_file}")
                    continue

                df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
                df['Log_File'] = log_file

                # Apply moving average to replace values less than 50
                for gpu_id in df['GPU_ID'].unique():
                    gpu_data = df[df['GPU_ID'] == gpu_id].copy()
                    low_power_indices = gpu_data[gpu_data['Power_Usage_W'] < 50].index
                    for idx in low_power_indices:
                        window = 100
                        start = max(0, idx - window // 2)
                        end = min(len(gpu_data), idx + window // 2 + 1)
                        valid_values = gpu_data.loc[start:end, 'Power_Usage_W']
                        valid_values = valid_values[valid_values >= 50]
                        if not valid_values.empty:
                            gpu_data.loc[idx, 'Power_Usage_W'] = valid_values.mean()
                    df.update(gpu_data)

                all_data.append(df)

            except Exception as e:
                print(f"Error reading file {log_file}: {e}")
                continue

    if not all_data:
        print("No valid data found to plot.")
        return

    combined_df = pd.concat(all_data)

    plt.figure(figsize=(26, 6))

    for gpu_id in combined_df['GPU_ID'].unique():
        gpu_data = combined_df[combined_df['GPU_ID'] == gpu_id]
        plt.plot(gpu_data['Timestamp'], gpu_data['Power_Usage_W'], label=f"GPU {gpu_id}")

    plt.title("Combined GPU Power Usage Over Time (All Logs) - Moving Average Applied (Values < 50)")
    plt.xlabel("Timestamp")
    plt.ylabel("Power Usage (W)")
    plt.legend()
    plt.grid(True)

    # **Improve x-axis label density and formatting**
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # Auto spacing of x-axis labels
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))  # Better formatting
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

    os.makedirs(graph_dir, exist_ok=True)
    graph_filename = os.path.join(graph_dir, "combined_gpu_power_usage_moving_average_less_50.png")
    plt.savefig(graph_filename, bbox_inches="tight")
    plt.close()
    print(f"Saved combined graph as {graph_filename}")

if __name__ == "__main__":
    plot_combined_gpu_power()

