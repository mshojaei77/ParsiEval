import json
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


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

def main():
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

    # Plot 5: Accuracy vs Parameters - Top Models
    api_ticks = [10, 30, 70, 120, 400, 1000]
    plot_accuracy_vs_parameters(api_df, 'Accuracy vs. Parameters for API-Based Models', 'accuracy_vs_parameters_top_models.png', xticks=api_ticks)

    # Plot 6: Accuracy vs Parameters - Edge-Device Models
    edge_ticks = [1, 2, 3, 4]
    plot_accuracy_vs_parameters(local_df, 'Accuracy vs. Parameters for Edge-Device Models', 'accuracy_vs_parameters_edge_models.png', xticks=edge_ticks)

    print("All 6 plots have been generated and saved in the 'plots' directory.")

if __name__ == '__main__':
    main()