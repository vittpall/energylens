#!/usr/bin/env python3
"""Script to analyze and visualize the distribution of parameters in the profiling metrics dataset."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv('results_mistral_extensive/profiling_metrics.csv')

print("=" * 60)
print("Dataset Overview")
print("=" * 60)
print(f"Total samples: {len(df)}")
print(f"Columns: {list(df.columns)}")
print()

# Separate numerical columns (excluding 'prompt' which is text)
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print("=" * 60)
print("Statistical Summary")
print("=" * 60)
print(df[numerical_cols].describe().round(4).to_string())
print()

# Create visualizations
fig, axes = plt.subplots(5, 4, figsize=(16, 20))
axes = axes.flatten()

for idx, col in enumerate(numerical_cols):
    ax = axes[idx]
    
    # Plot histogram
    data = df[col].dropna()
    ax.hist(data, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_title(f'{col}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Value', fontsize=8)
    ax.set_ylabel('Frequency', fontsize=8)
    
    # Add mean and median lines
    mean_val = data.mean()
    median_val = data.median()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='green', linestyle=':', linewidth=1.5, label=f'Median: {median_val:.2f}')
    ax.legend(fontsize=6)
    
    ax.tick_params(axis='both', labelsize=7)

# Hide unused subplots
for idx in range(len(numerical_cols), len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig('results_llava_extensive/parameter_distributions.png', dpi=150, bbox_inches='tight')
print("Saved distribution plot to: results_llava_extensive/parameter_distributions.png")

# Additional analysis: unique values for categorical-like parameters
print()
print("=" * 60)
print("Unique Values for Key Parameters")
print("=" * 60)

params_of_interest = ['image_resolution', 'batch_size', 'arrival_rate', 'gpu_memory_utilization',
                      'pipeline_parallel_size', 'tensor_parallel_size', 'max_tokens', 'min_tokens',
                      'repetition_penalty', 'temperature', 'top_k', 'top_p']

for param in params_of_interest:
    if param in df.columns:
        unique_vals = sorted(df[param].unique())
        print(f"{param}: {unique_vals} (count: {len(unique_vals)})")

# Correlation analysis for key metrics
print()
print("=" * 60)
print("Correlation with Latency")
print("=" * 60)
target_metrics = ['latency', 'energy', 'power']
feature_cols = [c for c in numerical_cols if c not in target_metrics + ['prompt']]

correlations = df[numerical_cols].corr()['latency'].sort_values(ascending=False)
print(correlations.round(4).to_string())

# Box plots for parameters vs latency
fig2, axes2 = plt.subplots(2, 3, figsize=(14, 10))
axes2 = axes2.flatten()

box_params = ['max_tokens', 'min_tokens', 'image_resolution', 'top_k', 'temperature', 'repetition_penalty']

for idx, param in enumerate(box_params):
    if param in df.columns:
        ax = axes2[idx]
        # Group by unique values and plot
        unique_vals = sorted(df[param].unique())
        data_groups = [df[df[param] == val]['latency'] for val in unique_vals]
        ax.boxplot(data_groups, labels=[f'{v:.2f}' if isinstance(v, float) else str(v) for v in unique_vals])
        ax.set_title(f'Latency by {param}', fontsize=10, fontweight='bold')
        ax.set_xlabel(param, fontsize=9)
        ax.set_ylabel('Latency (s)', fontsize=9)
        ax.tick_params(axis='x', rotation=45, labelsize=7)

plt.tight_layout()
plt.savefig('results_llava_extensive/latency_by_parameters.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved latency analysis plot to: results_llava_extensive/latency_by_parameters.png")
