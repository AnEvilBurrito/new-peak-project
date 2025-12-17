# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: new-peak-project
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Parameter Distortion Robustness Analysis
#
# Analysis of model performance degradation under parameter distortion for Section 4, Experiment 01

# %% [markdown]
# ## Init

# %%
import os

path = os.getcwd()
# find the string 'project' in the path, return index
index_project = path.find("project")
# slice the path from the index of 'project' to the end
project_path = path[: index_project + 7]
# set the working directory
os.chdir(project_path + "/src")
print(f"Project path set to: {os.getcwd()}")

# %%
from dotenv import dotenv_values
config = dotenv_values(".env")
SAVE_RESULT_PATH = config["SAVE_RESULT_PATH"]

# %%
notebook_name = 'robustness-parameter-distortion'
exp_number = '01'
section_number = '4'
version_number = 'v1'
notebook_config = {
    'notebook_name': notebook_name,
    'exp_number': exp_number,
    'version_number': version_number,
    'section_number': section_number
}

# %% [markdown]
# ## Load Data using Sequential Re-assembly
#
# Load the pre-generated parameter distortion results using the batch framework's re-assembly pattern

# %%
from scripts.batch_framework import create_batch_executor
from models.utils.s3_config_manager import S3ConfigManager
import pandas as pd

# Initialize S3 manager
s3_manager = S3ConfigManager()
print("✅ S3 connection established")

# Create batch executor for re-assembly
batch_executor = create_batch_executor(
    notebook_name='batch-parameter-distortion',
    exp_number='01',
    version_number='v1',
    section_number='4'
)

# Load data using sequential re-assembly
try:
    distortion_results = batch_executor.sequential_reassembly('parameter_distortion_results')
    print(f"✅ Loaded re-assembled parameter distortion results: {len(distortion_results)} data points")
    print(f"Data shape: {distortion_results.shape}")
    if not distortion_results.empty:
        print(f"Distortion factors: {distortion_results['Distortion Factor'].unique()}")
except Exception as e:
    print(f"❌ Re-assembly failed: {e}")
    # Create empty dataframe for fallback analysis
    distortion_results = pd.DataFrame()

# %%
# Preview the data
if not distortion_results.empty:
    print("Data Preview:")
    print(distortion_results.head())
    print("\nColumns:", distortion_results.columns.tolist())

# %% [markdown]
# ## Data Summary

# %%
import pandas as pd
import numpy as np

if not distortion_results.empty:
    # Summary statistics by distortion factor
    stats_summary = distortion_results.groupby('Distortion Factor').agg({
        'Pearson Correlation': ['mean', 'std', 'min', 'max'],
        'R2 Score': ['mean', 'std'],
        'Mean Squared Error': ['mean', 'std']
    }).round(3)
    
    print("Summary Statistics by Distortion Factor:")
    print(stats_summary)
    
    # Model performance comparison
    model_stats = distortion_results.groupby(['Model', 'Distortion Factor'])['Pearson Correlation'].mean().unstack()
    print("\nModel Performance (Average Pearson Correlation):")
    print(model_stats.round(3))

# %% [markdown]
# ## Visualizations
#
# Create visualizations showing performance degradation across distortion levels

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from models.utils.s3_config_manager import S3ConfigManager

# Set up plotting style for publication quality
sns.set_context("talk")
sns.set_style("whitegrid")
sns.set(font="Arial", font_scale=1.5)

# %%
if not distortion_results.empty:
    # Line plot for Pearson Correlation vs Distortion Factor
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=distortion_results, x='Distortion Factor', y='Pearson Correlation', 
                 hue='Feature Data', palette='Set1', marker='o', ci=95)
    plt.title('Model Performance Under Parameter Distortion')
    plt.xlabel('Distortion Factor')
    plt.ylabel('Pearson Correlation')
    plt.legend(title='Feature Data Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save figure to S3
    s3_manager.save_figure(notebook_config, plt.gcf(), "parameter_distortion_overview")
    plt.show()

# %%
if not distortion_results.empty:
    # Zoomed view for lower distortion factors
    plt.figure(figsize=(10, 6))
    filtered_data = distortion_results[distortion_results['Distortion Factor'] <= 2.0]
    sns.lineplot(data=filtered_data, x='Distortion Factor', y='Pearson Correlation', 
                 hue='Feature Data', palette='Set1', marker='o', ci=95)
    plt.title('Model Performance Under Parameter Distortion (Zoomed)')
    plt.xlabel('Distortion Factor')
    plt.ylabel('Pearson Correlation')
    plt.legend(title='Feature Data Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save figure to S3
    s3_manager.save_figure(notebook_config, plt.gcf(), "parameter_distortion_zoomed")
    plt.show()

# %%
if not distortion_results.empty:
    # Model comparison across distortion levels
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=distortion_results, x='Distortion Factor', y='Pearson Correlation', 
                 hue='Model', style='Model', palette='Dark2', markers=True, ci=95)
    plt.title('Model Comparison Under Parameter Distortion')
    plt.xlabel('Distortion Factor')
    plt.ylabel('Pearson Correlation')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save figure to S3
    s3_manager.save_figure(notebook_config, plt.gcf(), "model_comparison_distortion")
    plt.show()

# %%
if not distortion_results.empty:
    # Boxplot showing performance distribution
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=distortion_results, x='Distortion Factor', y='Pearson Correlation', 
                hue='Feature Data', palette='Set1')
    plt.title('Performance Distribution Under Parameter Distortion')
    plt.xlabel('Distortion Factor')
    plt.ylabel('Pearson Correlation')
    plt.legend(title='Feature Data Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save figure to S3
    s3_manager.save_figure(notebook_config, plt.gcf(), "performance_distribution_distortion")
    plt.show()

# %% [markdown]
# ## Statistical Analysis

# %%
if not distortion_results.empty:
    # Calculate performance degradation
    baseline_performance = distortion_results[distortion_results['Distortion Factor'] == 0]
    distorted_performance = distortion_results[distortion_results['Distortion Factor'] > 0]
    
    if not baseline_performance.empty and not distorted_performance.empty:
        # Average baseline performance
        baseline_mean = baseline_performance['Pearson Correlation'].mean()
        print(f"Baseline performance (no distortion): {baseline_mean:.3f}")
        
        # Performance at different distortion levels
        for factor in sorted(distortion_results['Distortion Factor'].unique()):
            if factor > 0:
                factor_data = distortion_results[distortion_results['Distortion Factor'] == factor]
                factor_mean = factor_data['Pearson Correlation'].mean()
                degradation = baseline_mean - factor_mean
                degradation_pct = (degradation / baseline_mean) * 100
                print(f"Distortion {factor}: {factor_mean:.3f} (degradation: {degradation:.3f}, {degradation_pct:.1f}%)")
    
    # Statistical significance testing between feature data types
    from scipy import stats
    
    print("\nFeature Data Type Comparison (Pearson Correlation):")
    feature_types = distortion_results['Feature Data'].unique()
    for i, type1 in enumerate(feature_types):
        for type2 in feature_types[i+1:]:
            data1 = distortion_results[distortion_results['Feature Data'] == type1]['Pearson Correlation']
            data2 = distortion_results[distortion_results['Feature Data'] == type2]['Pearson Correlation']
            t_stat, p_value = stats.ttest_ind(data1, data2)
            print(f"{type1} vs {type2}: t={t_stat:.3f}, p={p_value:.4f}")

# %% [markdown]
# ## Performance Metrics Summary

# %%
if not distortion_results.empty:
    # Create a comprehensive performance summary table
    performance_summary = distortion_results.groupby(['Distortion Factor', 'Model', 'Feature Data']).agg({
        'Pearson Correlation': ['mean', 'std', 'count'],
        'R2 Score': ['mean', 'std'],
        'Mean Squared Error': ['mean', 'std']
    }).round(3)
    
    print("Performance Summary Table:")
    print(performance_summary)

# %% [markdown]
# ## Save Analysis Results

# %%
# Save the summary statistics and analysis results
if not distortion_results.empty:
    # Save the filtered summary table
    simplified_summary = distortion_results.groupby(['Distortion Factor', 'Feature Data'])['Pearson Correlation'].agg(['mean', 'std', 'count']).round(3)
    
    try:
        s3_manager.save_data(notebook_config, simplified_summary, 'parameter_distortion_summary', data_format='csv')
        print("✅ Saved analysis summary to S3")
    except Exception as e:
        print(f"❌ Error saving summary: {e}")

# %% [markdown]
# ## Conclusion
#
# This analysis demonstrates the robustness of the models under parameter distortion. Key findings:
#
# 1. **Performance degradation pattern** with increasing distortion factors
# 2. **Comparison of feature data types** for robustness
# 3. **Model-specific sensitivity** to parameter perturbations
#
# The visualizations and statistical analysis provide insights into the system's tolerance to parameter variability.
