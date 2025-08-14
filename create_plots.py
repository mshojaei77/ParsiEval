import json
import os
import re
import time
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


def parse_parameters(model_name):
    # For names like 'gemma3:4b', 'qwen2.5:1.5b', 'llama3.2:1b'
    match = re.search(r'(\d*\.?\d+)(b|B)', model_name)
    if match:
        return float(match.group(1))

    # For names like 'EXAONE-3.5-2.4B'
    match = re.search(r'(\d+\.\d+)B', model_name)
    if match:
        return float(match.group(1))

    return None

def get_api_model_params():
    return {
        'gpt-4o': 200,
        'openrouter/horizon-beta': 120,  
        'moonshotai/kimi-k2-instruct': 1000,  # 1T = 1000B
        'llama-4-maverick-17b-128e-instruct': 402,
        'deepseek-v3-0324': 685,
        'llama-3.3-70b': 70,
        'llama-4-scout-17b-16e-instruct': 108,
        'qwen-3-235b-a22b-instruct-2507': 235,
        'openai/gpt-oss-120b': 120,
        'google/gemma-3-27b-it': 27,
        'qwen/qwen3-30b-a3b-thinking': 30,
        'openai/gpt-oss-20b': 20,
        'gpt-4.1-nano': 175,
        'gemma2-9b-it': 9,
        'mistral-small-2503': 7,
        'qwen-3-32b': 32,
        'qwen3-30b-a3b-instruct-2507': 30,
        'gemma-3n-e4b-it': 4,
        'qwen-3-32b-thinking': 32,
        'qwen-3-235b-a22b-thinking-2507': 235
    }
def load_and_process_data(filepath, is_local=False):
    with open(filepath, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df['accuracy'] = df['accuracy'].str.rstrip('%').astype(float)
    df['avg_latency'] = df['avg_latency'].str.rstrip('s').astype(float)

    if is_local:
        df['parameters'] = df['model'].apply(parse_parameters)
    else:
        api_params = get_api_model_params()
        df['parameters'] = df['model'].map(api_params)

    # Clean up model names for plotting
    df['model_short'] = df['model'].apply(lambda x: x.split('/')[-1])

    return df.sort_values('accuracy', ascending=False)

def create_plots_dir():
    if not os.path.exists('plots'):
        os.makedirs('plots')

def plot_accuracy(df, title, filename):
    plt.figure(figsize=(12, 8))
    plt.bar(df['model_short'], df['accuracy'], color=plt.cm.viridis(df['accuracy'] / 100))
    plt.ylabel('Accuracy (%)')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'plots/{filename}')
    plt.close()

def plot_accuracy_vs_latency(df, title, filename):
    plt.figure(figsize=(10, 6))
    for i, row in df.iterrows():
        plt.scatter(row['avg_latency'], row['accuracy'], label=row['model_short'])
    plt.xlabel('Average Latency (s)')
    plt.ylabel('Accuracy (%)')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'plots/{filename}')
    plt.close()

def plot_accuracy_vs_parameters(df, title, filename, xticks=None):
    df = df.dropna(subset=['parameters'])
    plt.figure(figsize=(10, 6))
    for i, row in df.iterrows():
        plt.scatter(row['parameters'], row['accuracy'], label=row['model_short'])
    plt.xlabel('Parameters (Billions)')
    plt.ylabel('Accuracy (%)')
    plt.title(title)
    plt.xscale('log')
    ax = plt.gca()

    def billions_formatter(x, pos):
        if x >= 1:
            return f'{int(x)}b'
        else:
            return f'{x}b'

    if xticks:
        ax.set_xticks(xticks)

    ax.xaxis.set_major_formatter(FuncFormatter(billions_formatter))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'plots/{filename}')
    plt.close()

def generate_plots(from_watcher=False):
    # Use non-interactive backend when called from file watcher to avoid thread issues
    if from_watcher:
        current_backend = mpl.get_backend()
        mpl.use('Agg')  # Use non-interactive backend
    
    create_plots_dir()

    local_df = load_and_process_data('evaluation_results_local.json', is_local=True)
    api_df = load_and_process_data('evaluation_results_api.json')

    # Plot 1: Accuracy - Top Models
    plot_accuracy(api_df, 'Accuracy of API-Based Models', 'accuracy_top_models.png')

    # Plot 2: Accuracy - Edge-Device Models
    plot_accuracy(local_df, 'Accuracy of Edge-Device Models', 'accuracy_edge_models.png')

    # Plot 3: Accuracy vs Latency - Top Models
    plot_accuracy_vs_latency(api_df, 'Accuracy vs. Latency for API-Based Models', 'accuracy_vs_latency_top_models.png')

    # Plot 4: Accuracy vs Latency - Edge-Device Models
    plot_accuracy_vs_latency(local_df, 'Accuracy vs. Latency for Edge-Device Models', 'accuracy_vs_latency_edge_models.png')

    # Dynamically determine parameter ticks based on data
    api_params = api_df['parameters'].dropna().unique()
    api_ticks = sorted([p for p in api_params if p >= 1])
    if len(api_ticks) < 3:  # Fallback if not enough data points
        api_ticks = [10, 30, 70, 120, 400, 1000]
    
    # Plot 5: Accuracy vs Parameters - Top Models
    plot_accuracy_vs_parameters(api_df, 'Accuracy vs. Parameters for API-Based Models', 'accuracy_vs_parameters_top_models.png', xticks=api_ticks)

    # Dynamically determine parameter ticks for edge models
    edge_params = local_df['parameters'].dropna().unique()
    edge_ticks = sorted([p for p in edge_params if p >= 0.1])
    if len(edge_ticks) < 3:  # Fallback if not enough data points
        edge_ticks = [1, 2, 3, 4]
    
    # Plot 6: Accuracy vs Parameters - Edge-Device Models
    plot_accuracy_vs_parameters(local_df, 'Accuracy vs. Parameters for Edge-Device Models', 'accuracy_vs_parameters_edge_models.png', xticks=edge_ticks)

    print("All 6 plots have been generated and saved in the 'plots' directory.")


class JsonFileHandler(FileSystemEventHandler):
    def __init__(self, target_files):
        self.target_files = target_files
        self.last_modified = {}
        for file in target_files:
            try:
                self.last_modified[file] = os.path.getmtime(file)
            except FileNotFoundError:
                self.last_modified[file] = 0
        
        # Generate plots on startup
        generate_plots()
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path in self.target_files:
            # Check if the file was actually modified (to avoid duplicate events)
            current_mtime = os.path.getmtime(event.src_path)
            if current_mtime > self.last_modified.get(event.src_path, 0):
                self.last_modified[event.src_path] = current_mtime
                print(f"File {event.src_path} has been modified. Regenerating plots...")
                generate_plots(from_watcher=True)


def main():
    # Initial plot generation
    generate_plots()
    
    # Set up file monitoring
    target_files = [
        os.path.abspath('evaluation_results_local.json'),
        os.path.abspath('evaluation_results_api.json')
    ]
    
    event_handler = JsonFileHandler(target_files)
    observer = Observer()
    observer.schedule(event_handler, path=os.path.dirname(os.path.abspath(__file__)), recursive=False)
    observer.start()
    
    print(f"Monitoring files for changes: {', '.join(target_files)}")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == '__main__':
    main()