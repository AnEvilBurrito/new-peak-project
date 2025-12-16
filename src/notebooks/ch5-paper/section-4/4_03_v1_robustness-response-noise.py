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
# # Response Noise Robustness Analysis
#
# Analysis of model performance degradation under response noise for Section 4, Experiment 03

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
notebook_name = 'robustness-response-noise'
exp_number = '03'
section_number = '4'
version_number = 'v1'
notebook_path = f'{SAVE_RESULT_PATH}/{exp_number}_{version_number}_{notebook_name}'
notebook_config = {
    'notebook_name': notebook_name,
    'exp_number': exp_number,
    'version_number': version_number,
    'section_number': section_number
}

# %% [markdown]
# ## Load Data using Sequential Re-assembly
#
# Load the pre-generated response noise results using the batch framework's re-assembly pattern

# %%
from scripts.batch_framework import create_batch_executor
from models.utils.s3_config_manager import S3ConfigManager
import pandas as pd

# Initialize S3 manager
s3_manager = S3ConfigManager()
print("✅ S3 connection established")

# Create batch executor for re-assembly
batch_executor = create_batch_executor(
    notebook_name='batch-response-noise',
    exp_number='03',
    version_number='v1',
    section_number='4'
)

# Load data using sequential re-assembly
try:
    response_noise_results = batch_executor.sequential_reassembly('response_noise_results')
    print(f"✅ Loaded re-assembled response noise results: {len(response_noise_results)} data points")
    print(f"Data shape: {response_noise_results.shape}")
    if not response_noise_results.empty:
        print(f"Noise levels: {response_noise_results['Response Noise Level'].unique()}")
except Exception as e:
    print(f"❌ Re-assembly failed: {e}")
    # Create empty dataframe for fallback analysis
    response_noise_results = pd.DataFrame()

# %%
# Preview the data
if not response_noise_results.empty:
    print("Data Preview:")
    print(response_noise_results.head())
    print("\nColumns:", response_noise_results.columns.tolist())

# %% [markdown]
# ## Data Summary

# %%
import pandas as pd
import numpy as np

if not response_noise_results.empty:
    # Summary statistics by noise level
    stats_summary = response_noise_results.groupby('Response Noise Level').agg({
        'Pearson Correlation': ['mean', 'std', 'min', 'max'],
        'R2 Score': ['mean', 'std'],
        'Mean Squared Error': ['mean', 'std']
    }).round(3)
    
    print("Summary Statistics by Noise Level:")
    print(stats_summary)
    
    # Model performance comparison
    model_stats = response_noise_results.groupby(['Model', 'Response Noise Level'])['Pearson Correlation'].mean().unstack()
    print("\nModel Performance (Average Pearson Correlation):")
    print(model_stats.round(3))

# %% [markdown]
# ## Visualizations
#
# Create visualizations showing performance degradation across response noise levels

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from models.utils.s3_config_manager import S3ConfigManager

# Set up plotting style for publication quality
sns.set_context("talk")
sns.set_style("whitegrid")
sns.set(font="Arial", font_scale=1.5)

# %%
if not response_noise_results.empty:
    # Line plot for Pearson Correlation vs Response Noise Level
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=response_noise_results, x='Response Noise Level', y='Pearson Correlation', 
                 hue='Feature Data', palette='Set1', marker='o', ci=95)
    plt.title('Model Performance Under Response Noise')
    plt.xlabel('Response Noise Level')
    plt.ylabel('Pearson Correlation')
    plt.legend(title='Feature Data Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save figure to S3
    s3_manager.save_figure(notebook_config, plt.gcf(), "response_noise_overview")
    plt.show()

# %%
if not response_noise_results.empty:
    # Zoomed view for lower noise levels
    plt.figure(figsize=(10, 6))
    filtered_data = response_noise_results[response_noise_results['Response Noise Level'] <= 0.3]
    sns.lineplot(data=filtered_data, x='Response Noise Level', y='Pearson Correlation', 
                 hue='Feature Data', palette='Set1', marker='o', ci=95)
    plt.title('Model Performance Under Response Noise (Zoomed)')
    plt.xlabel('Response Noise Level')
    plt.ylabel('Pearson Correlation')
    plt.xlim(0, 0.3)
    plt.legend(title='Feature Data Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save figure to S3
    s3_manager.save_figure(notebook_config, plt.gcf(), "response_noise_zoomed")
    plt.show()

# %%
if not response_noise_results.empty:
    # Model comparison across noise levels
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=response_noise_results, x='Response Noise Level', y='Pearson Correlation', 
                 hue='Model', style='Model', palette='Dark2', markers=True, ci=95)
    plt.title('Model Comparison Under Response Noise')
    plt.xlabel('Response Noise Level')
    plt.ylabel('Pearson Correlation')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save figure to S3
    s3_manager.save_figure(notebook_config, plt.gcf(), "model_comparison_response_noise")
    plt.show()

# %%
if not response_noise_results.empty:
    # Boxplot showing performance distribution
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=response_noise_results, x='Response Noise Level', y='Pearson Correlation', 
                hue='Feature Data', palette='Set1')
    plt.title('Performance Distribution Under Response Noise')
    plt.xlabel('Response Noise Level')
    plt.ylabel('Pearson Correlation')
    plt.legend(title='Feature Data Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save figure to S3
    s3_manager.save_figure(notebook_config, plt.gcf(), "performance_distribution_response_noise")
    plt.show()

# %%
if not response_noise_results.empty:
    # Performance degradation comparison across different feature data types
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    feature_data_types = ['feature_data', 'last_time_data', 'dynamic_data', 'combined_dyn_data']
    titles = ['Original Features', 'Last Time Features', 'Dynamic Features', 'Dynamic Features + Original']
    
    for i, (feature_type, title) in enumerate(zip(feature_data_types, titles)):
        ax = [ax1, ax2, ax3, ax4][i]
        feature_data = response_noise_results[response_noise_results['Feature Data'] == feature_type]
        
        sns.lineplot(data=feature_data, x='Response Noise Level', y='Pearson Correlation', 
                     hue='Model', palette='Dark2', marker='o', ax=ax)
        ax.set_title(f'{title} Performance')
        ax.set_xlabel('Response Noise Level')
        ax.set_ylabel('Pearson Correlation')
        ax.legend(title='Model')
    
    plt.tight_layout()
    
    # Save figure to S3
    s3_manager.save_figure(notebook_config, plt.gcf(), "feature_data_performance_grid")
    plt.show()

# %% [markdown]
# ## Statistical Analysis

# %%
if not response_noise_results.empty:
    # Calculate performance degradation
    baseline_performance = response_noise_results[response_noise_results['Response Noise Level'] == 0]
    noisy_performance = response_noise_results[response_noise_results['Response Noise Level'] > 0]
    
    if not baseline_performance.empty and not noisy_performance.empty:
        # Average baseline performance
        baseline_mean = baseline_performance['Pearson Correlation'].mean()
        print(f"Baseline performance (no noise): {baseline_mean:.3f}")
        
        # Performance at different noise levels
        for level in sorted(response_noise_results['Response Noise Level'].unique()):
            if level > 0:
                level_data = response_noise_results[response_noise_results['Response Noise Level'] == level]
                level_mean = level_data['Pearson Correlation'].mean()
                degradation = baseline_mean - level_mean
                degradation_pct = (degradation / baseline_mean) * 100
                print(f"Noise {level}: {level_mean:.3f} (degradation: {degradation:.3f}, {degradation_pct:.1f}%)")
    
    # Statistical significance testing between feature data types
    from scipy import stats
    
    print("\nFeature Data Type Comparison (Pearson Correlation):")
    feature_types = response_noise_results['Feature Data'].unique()
    for i, type1 in enumerate(feature_types):
        for type2 in feature_types[i+1:]:
            data1 = response_noise_results[response_noise_results['Feature Data'] == type1]['Pearson Correlation']
            data2 = response_noise_results[response_noise_results['Feature Data'] == type2]['Pearson Correlation']
            t_stat, p_value = stats.ttest_ind(data1, data2)
            print(f"{type1} vs {type2}: t={t_stat:.3f}, p={p_value:.4f}")

# %% [markdown]
# ## Performance Metrics Summary

# %%
if not response_noise_results.empty:
    # Create a comprehensive performance summary table
    performance_summary = response_noise_results.groupby(['Response Noise Level', 'Model', 'Feature Data']).agg({
        'Pearson Correlation': ['mean', 'std', 'count'],
        'R2 Score': ['mean', 'std'],
        'Mean Squared Error': ['mean', 'std']
    }).round(3)
    
    print("Performance Summary Table:")
    print(performance_summary)

# %% [markdown]
# ## Noise Sensitivity Analysis

# %%
if not response_noise_results.empty:
    # Calculate sensitivity coefficients for different feature data types
    noise_levels = sorted(response_noise_results['Response Noise Level'].unique())
    models = response_noise_results['Model'].unique()
    feature_types = response_noise_results['Feature Data'].unique()
    
    print("Performance Sensitivity to Response Noise:")
    
    for feature_type in feature_types:
        print(f"\nFeature Data: {feature_type}")
        for model in models:
            model_data = response_noise_results[
                (response_noise_results['Model'] == model) & 
                (response_noise_results['Feature Data'] == feature_type)
            ]
            
            if not model_data.empty:
                # Fit linear regression to estimate sensitivity
                from sklearn.linear_model import LinearRegression
                X = model_data['Response Noise Level'].values.reshape(-1, 1)
                y = model_data['Pearson Correlation'].values
                
                if len(X) > 1:
                    reg = LinearRegression().fit(X, y)
                    sensitivity = -reg.coef_[0]  # Negative coefficient means performance decreases with noise
                    baseline_pred = reg.predict(np.array([[0]]))[0]
                    print(f"{model}: sensitivity = {sensitivity:.4f}, baseline = {baseline_pred:.3f} (R² = {reg.score(X, y):.3f})")

# %% [markdown]
# ## Robustness Ranking

# %%
if not response_noise_results.empty:
    # Calculate robustness scores for each model-feature combination
    robustness_scores = []
    
    for model in models:
        for feature_type in feature_types:
            model_data = response_noise_results[
                (response_noise_results['Model'] == model) & 
                (response_noise_results['Feature Data'] == feature_type)
            ]
            
            if not model_data.empty:
                baseline_data = model_data[model_data['Response Noise Level'] == 0]
                noisy_data = model_data[model_data['Response Noise Level'] > 0]
                
                if not baseline_data.empty and not noisy_data.empty:
                    baseline_perf = baseline_data['Pearson Correlation'].mean()
                    avg_noisy_perf = noisy_data['Pearson Correlation'].mean()
                    robustness = avg_noisy_perf / baseline_perf if baseline_perf > 0 else 0
                    
                    robustness_scores.append({
                        'Model': model,
                        'Feature Data': feature_type,
                        'Baseline Performance': baseline_perf,
                        'Average Noisy Performance': avg_noisy_perf,
                        'Robustness Score': robustness
                    })
    
    # Create robustness ranking table
    robustness_df = pd.DataFrame(robustness_scores)
    if not robustness_df.empty:
        robustness_df = robustness_df.sort_values('Robustness Score', ascending=False).round(3)
        print("Robustness Ranking (Higher is better):")
        print(robustness_df)

# %% [markdown]
# ## Save Analysis Results

# %%
# Save the summary statistics and analysis results
if not response_noise_results.empty:
    # Save the filtered summary table
    simplified_summary = response_noise_results.groupby(['Response Noise Level', 'Feature Data'])['Pearson Correlation'].agg(['mean', 'std', 'count']).round(3)
    
    try:
        s3_manager.save_data(notebook_config, simplified_summary, 'response_noise_summary', data_format='csv')
        print("✅ Saved analysis summary to S3")
    except Exception as e:
        print(f"❌ Error saving summary: {e}")
    
    # Save robustness ranking
    if 'robustness_df' in locals() and not robustness_df.empty:
        try:
            s3_manager.save_data(notebook_config, robustness_df, 'response_noise_robustness_ranking', data_format='csv')
            print("✅ Saved robustness ranking to S3")
        except Exception as e:
            print(f"❌ Error saving robustness ranking: {e}")

# %% [markdown]
# ## Conclusion
#
# This analysis demonstrates the robustness of the models under response noise. Key findings:
#
# 1. **Performance degradation pattern** with increasing response noise levels
# 2. **Comparison of feature data types** for handling noisy responses
# 3. **Model-specific sensitivity** to response perturbations
# 4. **Robustness ranking** identifying the most resilient model-feature combinations
# 5. **Noise sensitivity coefficients** quantifying degradation rates
#
# The analysis provides comprehensive insights into the system's tolerance to response variability and prediction accuracy under noise conditions.
