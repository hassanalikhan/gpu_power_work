import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates  # Import for better date formatting

# Directory where GPU log files are saved
log_dir = "gpu_logs"
# Directory where graphs will be saved
graph_dir = "gpu_graphs"

total_gpus_power = {}

def plot_combined_gpu_power():
        """Generate and save a combined line plot for GPU power usage from all log files,
           replacing values less than 50 with a moving average."""
        all_data = []

        # Loop through all CSV files in the logs directory
        ldf = sorted(os.listdir(log_dir))
        for log_file in ldf:
                if log_file.endswith(".csv"):
                        log_file_path = os.path.join(log_dir, log_file)

                        if os.path.getsize(log_file_path) == 0:
                                print(f"Skipping empty file: {log_file}")
                                continue

                        try:
                                df = pd.read_csv(log_file_path)

                                if df.empty or not {'GPU_ID', 'Power_Usage_W', 'Timestamp'}.issubset(df.columns):
                                        print(f"Skipping invalid file (missing expected columns): {log_file}")
                                        continue

                                df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
                                df['Log_File'] = log_file

                                # Apply moving average to replace values less than 50
                                for gpu_id in df['GPU_ID'].unique():
                                        gpu_data = df[df['GPU_ID'] == gpu_id].copy()

                                        if gpu_id not in total_gpus_power:
                                                total_gpus_power[gpu_id] = []

                                        total_gpus_power[gpu_id] += list(gpu_data['Power_Usage_W'].values)

                                        
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
        
        avg_values = 20

        for gpu_id in combined_df['GPU_ID'].unique():
                gpu_data = combined_df[combined_df['GPU_ID'] == gpu_id]
                #for i in range(1, len(total_gpus_power[gpu_id])- avg_values):
                #	if sum(total_gpus_power[gpu_id][i:i+avg_values])/avg_values > 200 and sum(total_gpus_power[gpu_id][i:avg_values])/avg_values < 250:
                #		for j in range(i, i+avg_values):
                #			total_gpus_power[gpu_id][j] += 200
                #		i = j
                	#	print(total_gpus_power[gpu_id][i])
                	#if total_gpus_power[gpu_id][i] <= 50:
                	#	print('Replacing vals')
                	#	total_gpus_power[gpu_id][i] = total_gpus_power[gpu_id][i-1]
                plt.plot(total_gpus_power[gpu_id], label=f"GPU {gpu_id}")

        plt.title("Combined GPU Power Usage Over Time (All Logs) - Moving Average Applied (Values < 50)")
        plt.xlabel("Timestamp")
        plt.ylabel("Power Usage (W)")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Grid on both major and minor ticks

        # **Improve x-axis label density and formatting**
        ax = plt.gca()
        #ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # Auto spacing of x-axis labels
        #ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))  # Better date formatting
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

        # **More X Ticks with Minor Ticks**
        #ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=5))  # Minor ticks every 5 minutes
        ax.tick_params(axis='x', which='minor', length=4, color='gray')  # Style minor ticks

        # **More Y Ticks with Minor Ticks**
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=20))  # Increase number of y-axis ticks
        ax.yaxis.set_minor_locator(plt.MultipleLocator(5))  # Minor ticks every 5W
        ax.tick_params(axis='y', which='minor', length=4, color='gray')  # Style minor ticks

        os.makedirs(graph_dir, exist_ok=True)
        graph_filename = os.path.join(graph_dir, "combined_gpu_power_usage_moving_average_less_50.png")
        plt.savefig(graph_filename, bbox_inches="tight")
        plt.close()
        print(f"Saved combined graph as {graph_filename}")

if __name__ == "__main__":
        plot_combined_gpu_power()

