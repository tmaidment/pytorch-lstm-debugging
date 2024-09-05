import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def filter_versions(data, exclude_versions=None):
    if exclude_versions:
        return data[~data['torch_version'].isin(exclude_versions)]
    return data

def jitter(x, factor=0.1):
    return x + np.random.normal(0, factor, size=len(x))

def create_plots(data, exclude_versions=None):
    filtered_data = filter_versions(data, exclude_versions)
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey='row')

    plot_configs = [
        ('avg_warmup_inference_time', axs[0, 0], 'Avg Warmup Inference Time (s)', 'Warmup Inference Time Comparison'),
        ('avg_hot_inference_time', axs[0, 1], '', 'Hot Inference Time Comparison'),
        ('peak_memory_warmup_mb', axs[1, 0], 'Peak Memory Warmup (MB)', 'Peak Memory Warmup Comparison'),
        ('peak_memory_hot_mb', axs[1, 1], '', 'Peak Memory Hot Comparison')
    ]

    for plot_type, ax, ylabel, title in plot_configs:
        for torch_version in filtered_data['torch_version'].unique():
            df = filtered_data[filtered_data['torch_version'] == torch_version]
            ax.scatter(jitter(df['compiled']), df[plot_type], label=f'{torch_version}')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

    for ax in axs.flat:
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Not Compiled', 'Compiled'])

    axs[1, 0].set_xlabel('Compilation Status')
    axs[1, 1].set_xlabel('Compilation Status')

    for row in axs:
        y_min = min(ax.get_ylim()[0] for ax in row)
        y_max = max(ax.get_ylim()[1] for ax in row)
        for ax in row:
            ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(f'pytorch_performance_comparison{"_filtered" if exclude_versions else ""}.png')
    plt.close()

def create_heatmap(data, exclude_versions=None):
    filtered_data = filter_versions(data, exclude_versions)
    versions = filtered_data['torch_version'].unique()
    
    speedup_data = []
    for v1 in versions:
        row = []
        for v2 in versions:
            if v1 == v2:
                row.append(1)
            else:
                speedup = filtered_data[filtered_data['torch_version'] == v2]['avg_hot_inference_time'].mean() / \
                          filtered_data[filtered_data['torch_version'] == v1]['avg_hot_inference_time'].mean()
                row.append(speedup)
        speedup_data.append(row)

    plt.figure(figsize=(12, 10))
    sns.heatmap(speedup_data, annot=True, fmt='.2f', xticklabels=versions, yticklabels=versions)
    plt.title('Speedup Comparison (row vs column)')
    plt.xlabel('PyTorch Version')
    plt.ylabel('PyTorch Version')
    plt.tight_layout()

    plt.savefig(f'pytorch_speedup_heatmap{"_filtered" if exclude_versions else ""}.png', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Read the CSV file
    data = pd.read_csv('pytorch_performance_results.csv')

    # Create plots with all versions
    create_plots(data)
    create_heatmap(data)

    # Create plots excluding specific versions
    exclude_versions = []  
    create_plots(data, exclude_versions)
    create_heatmap(data, exclude_versions)

    print(f"Plots saved with and without excluding {', '.join(exclude_versions)}")