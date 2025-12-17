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
# # Expression Noise Robustness Analysis
#
# Analysis of model performance degradation under expression noise for Section 4, Experiment 02

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
notebook_name = 'robustness-expression-noise'
exp_number = '02'
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
# Load the pre-generated expression noise results using the batch framework's re-assembly pattern

# %%
from scripts.batch_framework import create_batch_executor
from models.utils.s3_config_manager import S3ConfigManager
import pandas as pd

# Initialize S3 manager
s3_manager = S3ConfigManager()
print("✅ S3 connection established")

# Create batch executor for re-assembly
batch_executor = create_batch_executor(
    notebook_name='batch-expression-noise',
    exp_number='02',
    version_number='v1',
    section_number='4'
)

# Load data using sequential re-assembly
try:
    expression_noise_results = batch_executor.sequential_reassembly('expression_noise_results')
    print(f"✅ Loaded re-assembled expression noise results: {len(expression_noise_results)} data points")
    print(f"Data shape: {expression_noise_results.shape}")
    if not expression_noise_results.empty:
        print(f"Noise levels: {expression_noise_results['Expression Noise Level'].unique()}")
except Exception as e:
    print(f"❌ Re-assembly failed: {e}")
    # Create empty dataframe for fallback analysis
    expression_noise_results = pd.DataFrame()

# %%
# Preview the data
if not expression_noise_results.empty:
    print("Data Preview:")
    print(expression_noise_results.head())
    print("\nColumns:", expression_noise_results.columns.tolist())

# %% [markdown]
# ## Data Summary

# %%
import pandas as pd
import numpy as np

if not expression_noise_results.empty:
    # Summary statistics by noise level
    stats_summary = expression_noise_results.groupby('Expression Noise Level').agg({
        'Pearson Correlation': ['mean', 'std', 'min', 'max'],
        'R2 Score': ['mean', 'std'],
        'Mean Squared Error': ['mean', 'std']
    }).round(3)
    
    print("Summary Statistics by Noise Level:")
    print(stats_summary)
    
    # Model performance comparison
    model_stats = expression_noise_results.groupby(['Model', 'Expression Noise Level'])['Pearson Correlation'].mean().unstack()
    print("\nModel Performance (Average Pearson Correlation):")
    print(model_stats.round(3))

# %% [markdown]
# ## Visualizations
#
# Create visualizations showing performance degradation across expression noise levels

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from models.utils.s3_config_manager import S3ConfigManager

# Set up plotting style for publication quality
sns.set_context("talk")
sns.set_style("whitegrid")
sns.set(font="Arial", font_scale=1.5)

# %%
if not expression_noise_results.empty:
    # Line plot for Pearson Correlation vs Expression Noise Level
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=expression_noise_results, x='Expression Noise Level', y='Pearson Correlation', 
                 hue='Feature Data', palette='Set1', marker='o', ci=95)
    plt.title('Model Performance Under Expression Noise')
    plt.xlabel('Expression Noise Level')
    plt.ylabel('Pearson Correlation')
    plt.legend(title='Feature Data Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save figure to S3
    s3_manager.save_figure(notebook_config, plt.gcf(), "expression_noise_overview")
    plt.show()

# %%
if not expression_noise_results.empty:
    # Zoomed view for lower noise levels
    plt.figure(figsize=(10, 6))
    filtered_data = expression_noise_results[expression_noise_results['Expression Noise Level'] <= 0.5]
    sns.lineplot(data=filtered_data, x='Expression Noise Level', y='Pearson Correlation', 
                 hue='Feature Data', palette='Set1', marker='o', ci=95)
    plt.title('Model Performance Under Expression Noise (Zoomed)')
    plt.xlabel('Expression Noise Level')
    plt.ylabel('Pearson Correlation')
    plt.xlim(0, 0.5)
    plt.legend(title='Feature Data Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save figure to S3
    s3_manager.save_figure(notebook_config, plt.gcf(), "expression_noise_zoomed")
    plt.show()

# %%
if not expression_noise_results.empty:
    # Model comparison across noise levels
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=expression_noise_results, x='Expression Noise Level', y='Pearson Correlation', 
                 hue='Model', style='Model', palette='Dark2', markers=True, ci=95)
    plt.title('Model Comparison Under Expression Noise')
    plt.xlabel('Expression Noise Level')
    plt.ylabel('Pearson Correlation')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save figure to S3
    s3_manager.save_figure(notebook_config, plt.gcf(), "model_comparison_expression_noise")
    plt.show()

# %%
if not expression_noise_results.empty:
    # Boxplot showing performance distribution
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=expression_noise_results, x='Expression Noise Level', y='Pearson Correlation', 
                hue='Feature Data', palette='Set1')
    plt.title('Performance Distribution Under Expression Noise')
    plt.xlabel('Expression Noise Level')
    plt.ylabel('Pearson Correlation')
    plt.legend(title='Feature Data Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save figure to S3
    s3_manager.save_figure(notebook_config, plt.gcf(), "performance_distribution_expression_noise")
    plt.show()

# %%
if not expression_noise_results.empty:
    # Compare original vs noisy feature data performance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original feature data
    original_data = expression_noise_results[expression_noise_results['Feature Data'] == 'original_feature_data']
    sns.lineplot(data=original_data, x='Expression Noise Level', y='Pearson Correlation', 
                 hue='Model', palette='Dark2', marker='o', ax=ax1)
    ax1.set_title('Original Feature Data Performance')
    ax1.set_xlabel('Expression Noise Level')
    ax1.set_ylabel('Pearson Correlation')
    ax1.legend(title='Model')
    
    # Noisy feature data
    noisy_data = expression_noise_results[expression_noise_results['Feature Data'] == 'noisy_feature_data']
    sns.lineplot(data=noisy_data, x='Expression Noise Level', y='Pearson Correlation', 
                 hue='Model', palette='Dark2', marker='o', ax=ax2)
    ax2.set_title('Noisy Feature Data Performance')
    ax2.set_xlabel('Expression Noise Level')
    ax2.set_ylabel('Pearson Correlation')
    ax2.legend(title='Model')
    
    plt.tight_layout()
    
    # Save figure to S3
    s3_manager.save_figure(notebook_config, plt.gcf(), "feature_data_comparison")
    plt.show()

# %% [markdown]
# ## Statistical Analysis

# %%
if not expression_noise_results.empty:
    # Calculate performance degradation
    baseline_performance = expression_noise_results[expression_noise_results['Expression Noise Level'] == 0]
    noisy_performance = expression_noise_results[expression_noise_results['Expression Noise Level'] > 0]
    
    if not baseline_performance.empty and not noisy_performance.empty:
        # Average baseline performance
        baseline_mean = baseline_performance['Pearson Correlation'].mean()
        print(f"Baseline performance (no noise): {baseline_mean:.3f}")
        
        # Performance at different noise levels
        for level in sorted(expression_noise_results['Expression Noise Level'].unique()):
            if level > 0:
                level_data = expression_noise_results[expression_noise_results['Expression Noise Level'] == level]
                level_mean = level_data['Pearson Correlation'].mean()
                degradation = baseline_mean - level_mean
                degradation_pct = (degradation / baseline_mean) * 100
                print(f"Noise {level}: {level_mean:.3f} (degradation: {degradation:.3f}, {degradation_pct:.1f}%)")
    
    # Statistical significance testing between feature data types
    from scipy import stats
    
    print("\nFeature Data Type Comparison (Pearson Correlation):")
    feature_types = expression_noise_results['Feature Data'].unique()
    for i, type1 in enumerate(feature_types):
        for type2 in feature_types[i+1:]:
            data1 = expression_noise_results[expression_noise_results['Feature Data'] == type1]['Pearson Correlation']
            data2 = expression_noise_results[expression_noise_results['Feature Data'] == type2]['Pearson Correlation']
            t_stat, p_value = stats.ttest_ind(data1, data2)
            print(f"{type1} vs {type2}: t={t_stat:.3f}, p={p_value:.4f}")

# %% [markdown]
# ## Performance Metrics Summary

# %%
if not expression_noise_results.empty:
    # Create a comprehensive performance summary table
    performance_summary = expression_noise_results.groupby(['Expression Noise Level', 'Model', 'Feature Data']).agg({
        'Pearson Correlation': ['mean', 'std', 'count'],
        'R2 Score': ['mean', 'std'],
        'Mean Squared Error': ['mean', 'std']
    }).round(3)
    
    print("Performance Summary Table:")
    print(performance_summary)

# %% [markdown]
# ## Noise Sensitivity Analysis

# %%
if not expression_noise_results.empty:
    # Calculate sensitivity coefficients for each model
    noise_levels = sorted(expression_noise_results['Expression Noise Level'].unique())
    models = expression_noise_results['Model'].unique()
    feature_types = expression_noise_results['Feature Data'].unique()
    
    print("Performance Sensitivity to Expression Noise:")
    
    for feature_type in ['original_feature_data', 'noisy_feature_data']:
        print(f"\nFeature Data: {feature_type}")
        for model in models:
            model_data = expression_noise_results[
                (expression_noise_results['Model'] == model) & 
                (expression_noise_results['Feature Data'] == feature_type)
            ]
            
            if not model_data.empty:
                # Fit linear regression to estimate sensitivity
                from sklearn.linear_model import LinearRegression
                X = model_data['Expression Noise Level'].values.reshape(-1, 1)
                y = model_data['Pearson Correlation'].values
                
                if len(X) > 1:
                    reg = LinearRegression().fit(X, y)
                    sensitivity = -reg.coef_[0]  # Negative coefficient means performance decreases with noise
                    print(f"{model}: sensitivity = {sensitivity:.4f} (R² = {reg.score(X, y):.3f})")

# %% [markdown]
# ## Save Analysis Results

# %%
# Save the summary statistics and analysis results
if not expression_noise_results.empty:
    # Save the filtered summary table
    simplified_summary = expression_noise_results.groupby(['Expression Noise Level', 'Feature Data'])['Pearson Correlation'].agg(['mean', 'std', 'count']).round(3)
    
    try:
        s3_manager.save_data(notebook_config, simplified_summary, 'expression_noise_summary', data_format='csv')
        print("✅ Saved analysis summary to S3")
    except Exception as e:
        print(f"❌ Error saving summary: {e}")

# %% [markdown]
# ## Conclusion
#
# This analysis demonstrates the robustness of the models under expression noise. Key findings:
#
# 1. **Performance degradation pattern** with increasing expression noise levels
# 2. **Comparison of original vs noisy feature data** performance
# 3. **Model-specific sensitivity** to expression perturbations
# 4. **Noise sensitivity coefficients** quantifying degradation rates
#
# The analysis provides insights into the system's tolerance to expression variability and measurement noise.
