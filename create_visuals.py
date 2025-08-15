import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path

INPUT_FILE = "results/parsi-eval-1.json"
PLOTS_DIR = "plots"
README_FILE = "README.md"


def create_directory_if_not_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_data(file_path):
    """Load data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def prepare_dataframe(data):
    """Convert JSON data to pandas DataFrame and prepare for visualization"""
    df = pd.DataFrame(data)
    
    # Convert accuracy from string to float
    df['accuracy_float'] = df['accuracy'].apply(lambda x: float(x.strip('%')))
    
    # Convert latency from string to float
    df['avg_latency_float'] = df['avg_latency'].apply(lambda x: float(x.strip('s')))
    df['total_latency_float'] = df['total_latency'].apply(lambda x: float(x.strip('s')))
    
    # Add is_local flag - check if system exists and is a dictionary
    df['is_local'] = df.apply(lambda row: 'system' in row and isinstance(row['system'], dict), axis=1)
    
    return df


def create_top_models_plot(df, output_path):
    """Create plot for top 10 models by accuracy"""
    # Sort by accuracy and get top 10
    top_models = df.sort_values('accuracy_float', ascending=False).head(10)
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(top_models['model'], top_models['accuracy_float'], color='skyblue')
    
    # Add accuracy percentage to the bars
    for i, bar in enumerate(bars):
        accuracy = top_models.iloc[i]['accuracy_float']
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                 f"{accuracy:.2f}%", va='center', fontsize=9)
    
    plt.xlabel('Accuracy (%)')
    plt.ylabel('Model')
    plt.title('Top 10 Models by Accuracy')
    plt.xlim(0, 100)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_top_local_models_plot(df, output_path):
    """Create plot for top 10 local models by accuracy"""
    # Filter local models and sort by accuracy
    local_models = df[df['is_local']].sort_values('accuracy_float', ascending=False).head(10)
    
    # Check if we have any local models
    if local_models.empty:
        print("No local models found in the dataset. Skipping local models plot.")
        # Create a simple plot indicating no local models
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No local models found in the dataset", 
                 ha='center', va='center', fontsize=14)
        plt.axis('off')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(local_models['model'], local_models['accuracy_float'], color='lightgreen')
    
    # Add accuracy percentage to the bars
    for i, bar in enumerate(bars):
        accuracy = local_models.iloc[i]['accuracy_float']
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                 f"{accuracy:.2f}%", va='center', fontsize=9)
    
    plt.xlabel('Accuracy (%)')
    plt.ylabel('Model')
    plt.title('Top 10 Local Models by Accuracy')
    plt.xlim(0, 100)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_results_table(df):
    """Create markdown table with all results"""
    # Sort by accuracy
    sorted_df = df.sort_values('accuracy_float', ascending=False).reset_index(drop=True)
    
    # Create table header
    table = "| Rank | Model | Size | License | API Provider | Avg Latency | Total Latency | Accuracy |\n"
    table += "|------|-------|------|---------|----------|-------------|---------------|----------|\n"
    
    # Add rows
    for i, row in sorted_df.iterrows():
        rank = i + 1
        provider = row['provider']
        model = row['model']
        size = row['model_size']
        license_type = row['license']
        avg_latency = row['avg_latency']
        total_latency = row['total_latency']
        accuracy = row['accuracy']
        
        table += f"| {rank} | {model} | {size} | {license_type} | {provider} | {avg_latency} | {total_latency} | {accuracy} |\n"
    
    return table


def update_readme(readme_path, table_content):
    """Update README.md with results table and plots"""
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Define the section to replace - match from ## Results to ## Future Plan
    results_section_pattern = r'## Results.*?(?=## Future Plan)'  # Match from ## Results to ## Future Plan
    
    # Create new results section
    new_results_section = """## Results

Here are the results of the evaluation for different models:

{}

### Accuracy

#### Top Models
Analysis of the highest performing models

![Accuracy of Top Models](plots/accuracy_top_models.png)

#### Local Models
Examination of models running on local hardware

![Accuracy of Local Models](plots/accuracy_local_models.png)

""".format(table_content)
    
    # Replace the section
    new_content = re.sub(results_section_pattern, new_results_section, content, flags=re.DOTALL)
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(new_content)


def main():
    """Main function to create visualizations and update README"""
    # Create plots directory if it doesn't exist
    create_directory_if_not_exists(PLOTS_DIR)
    
    # Load and prepare data
    data = load_data(INPUT_FILE)
    df = prepare_dataframe(data)
    
    # Create plots
    create_top_models_plot(df, os.path.join(PLOTS_DIR, 'accuracy_top_models.png'))
    create_top_local_models_plot(df, os.path.join(PLOTS_DIR, 'accuracy_local_models.png'))
    
    # Create and update results table in README
    table_content = create_results_table(df)
    update_readme(README_FILE, table_content)
    
    print("Visualizations created and README updated successfully!")


if __name__ == "__main__":
    main()
