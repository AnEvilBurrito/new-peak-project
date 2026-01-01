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
# # Baseline Data Loading (Direct S3 Loader)
#
# Script to load baseline virtual models directly from S3 without importing baseline_generator.
#
# **S3 Structure**: {save_result_path}/data/{model_name}_baseline_virtual_models/
# ├── baseline_features.pkl      # DataFrame: n_samples × n_features
# ├── baseline_targets.pkl       # DataFrame: n_samples × 1 (outcome)
# ├── baseline_parameters.pkl    # DataFrame: n_samples × n_parameters
# ├── baseline_timecourses.pkl   # Timecourse data
# └── baseline_metadata.pkl      # Generation metadata

# %% [markdown]
# ## Configuration

# %%
# Configuration variables
MODEL_NAME = "sy_simple"  # Can be string: "sy_simple" or list: ["sy_simple", "model_v2"]
COMPONENTS_TO_LOAD = ["features", "targets", "parameters", "metadata"]  # Which components to load
SAMPLE_SIZE = None  # None to load all, or integer to sample

# %% [markdown]
# ## Initialization

# %%
import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set up project path - same pattern as other notebooks
path = os.getcwd()
# find the string 'project' in the path, return index
index_project = path.find("project")
# slice the path from the index of 'project' to the end
project_path = path[: index_project + 7]
print(f"✅ Project path: {project_path}")

# Add src to Python path for imports
sys.path.insert(0, os.path.join(project_path, "src"))

# Import S3ConfigManager
from models.utils.s3_config_manager import S3ConfigManager

# Initialize S3 manager
s3_manager = S3ConfigManager()
print(f"✅ S3 connection established. Save result path: {s3_manager.save_result_path}")

# %% [markdown]
# ## Direct S3 Loading Functions

# %%
def process_model_config(model_config):
    """
    Convert MODEL_NAME config to list of model names for processing.
    
    Args:
        model_config: Can be string (single model) or list (multiple models)
    
    Returns:
        List of model names
    """
    if isinstance(model_config, str):
        return [model_config]
    elif isinstance(model_config, list):
        return model_config
    else:
        raise ValueError(f"MODEL_NAME must be str or list, got {type(model_config)}")


def load_baseline_direct_single(model_name, components, s3_manager):
    """
    Load baseline components for a single model directly from S3.
    
    Args:
        model_name: Name of the model (string)
        components: List of components to load
        s3_manager: S3ConfigManager instance
        
    Returns:
        Dictionary of loaded components
    """
    gen_path = s3_manager.save_result_path
    folder_name = f"{model_name}_baseline_virtual_models"
    base_path = f"{gen_path}/data/{folder_name}"
    
    print(f"📊 Loading baseline data for model: {model_name}")
    print(f"   S3 path: {base_path}")
    
    # Map component names to file names
    file_mapping = {
        'features': 'baseline_features.pkl',
        'targets': 'baseline_targets.pkl',
        'parameters': 'baseline_parameters.pkl',
        'timecourses': 'baseline_timecourses.pkl',
        'metadata': 'baseline_metadata.pkl'
    }
    
    result = {}
    
    for component in components:
        if component in file_mapping:
            filename = file_mapping[component]
            s3_path = f"{base_path}/{filename}"
            
            try:
                data = s3_manager.load_data_from_path(s3_path, data_format="pkl")
                result[component] = data
                print(f"  ✅ Loaded {component}: {type(data).__name__}")
                if isinstance(data, pd.DataFrame):
                    print(f"    Shape: {data.shape}")
            except Exception as e:
                print(f"  ❌ Error loading {component} from {s3_path}: {e}")
                result[component] = None
        else:
            print(f"  ⚠️ Unknown component: {component}")
            result[component] = None
    
    return result


def load_baseline_direct(model_config, components, s3_manager):
    """
    Load baseline components directly from S3, handling both single and multiple models.
    
    Args:
        model_config: Can be string (single model) or list (multiple models)
        components: List of components to load (e.g., ["features", "targets", "parameters"])
        s3_manager: S3ConfigManager instance
        
    Returns:
        If single model: Dictionary of loaded components
        If multiple models: Dictionary {model_name: component_dict}
    """
    model_names = process_model_config(model_config)
    
    if len(model_names) == 1:
        return load_baseline_direct_single(model_names[0], components, s3_manager)
    else:
        return {model_name: load_baseline_direct_single(model_name, components, s3_manager) for model_name in model_names}


# %%
# Load the baseline data
baseline_data = load_baseline_direct(MODEL_NAME, COMPONENTS_TO_LOAD, s3_manager)

# %%
if baseline_data and SAMPLE_SIZE:
    print(f"\n🔍 Creating sampled version ({SAMPLE_SIZE} samples)")
    sampled_data = {}
    
    for component, data in baseline_data.items():
        if data is not None and isinstance(data, pd.DataFrame):
            if len(data) > SAMPLE_SIZE:
                sampled_data[component] = data.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
                print(f"  ✅ Sampled {component}: {sampled_data[component].shape}")
            else:
                sampled_data[component] = data
                print(f"  ⚠️ {component} has only {len(data)} samples, using all")
        else:
            sampled_data[component] = data
    
    # Update baseline_data with sampled version
    baseline_data = sampled_data

# %% [markdown]
# ## Data Statistics (Optional)
#
# Basic statistics for loaded data components

# %%
SHOW_STATISTICS = True  # Set to False to skip statistics

if baseline_data and SHOW_STATISTICS:
    print("\n📈 Basic Statistics:")
    print("=" * 50)
    
    for component, data in baseline_data.items():
        if data is not None and isinstance(data, pd.DataFrame):
            print(f"\n{component.upper()} Statistics:")
            print(f"  Number of samples: {len(data)}")
            print(f"  Number of features: {len(data.columns)}")
            
            # Basic numeric statistics
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                print(f"  Numeric columns: {len(numeric_cols)}")
                # Show statistics for first few columns
                for col in numeric_cols[:3]:
                    print(f"    {col}: mean={data[col].mean():.4f}, std={data[col].std():.4f}, range=[{data[col].min():.4f}, {data[col].max():.4f}]")
