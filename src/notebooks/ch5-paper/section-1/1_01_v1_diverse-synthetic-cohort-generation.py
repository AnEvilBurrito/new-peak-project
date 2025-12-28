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
# # Init

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

# %% [markdown]
# # Config

# %%
from models.utils.s3_config_manager import S3ConfigManager

# Initialize S3 config manager
s3_manager = S3ConfigManager()

# Define notebook configuration following ch5-paper conventions
notebook_name = 'diverse-synthetic-cohort-generation'
exp_number = '01'  # 1st experiment in section
section_number = '1'  # located in section-1
version_number = 'v1'

notebook_config = {
    'exp_number': exp_number,
    'version_number': version_number, 
    'notebook_name': notebook_name,
    'section_number': section_number
}

# Calculate notebook path for S3 storage
notebook_path = f'{exp_number}_{version_number}_{notebook_name}'

# Define experiment configuration with full processing pipeline
exp_config = {
    'spec': {
        'n_layers': 2,
        'n_cascades': 3,
        'n_regs': 0,
        'gen_seed': 42,
        'basal_activation': True,
        'custom_regulations': [
            ['R1', 'R2', 'up'],
            ['R3', 'I1_2', 'up'],
            ['I1_1', 'I2_2', 'up'],
            ['I1_2', 'I2_1', 'down'],
            ['I1_2', 'I2_3', 'down'],
            ['I1_3', 'I2_2', 'up'],
            ['I2_1', 'R1', 'down'],
            ['I2_3', 'R3', 'up']
        ],
        'drug': {
            'name': "D",
            'start': 500,
            'dose': 500,
            'regulations': [
                ['R1', 'down']
            ],
            'target_all': False
        }
    },
    'parameter_generation': {
        'ic_range': [200, 1000],
        'param_range': [0.5, 2],
        'param_mul_range': [0.99, 1.01]
    },
    'parameter_sampling': {
        'sampling_seed': 42,
        'num_models': 1000,
        'num_datapoints': 1000
    },
    'feature_generation': {
        'include_parameters': False,  # kinetic parameters also used as feature data
        'excluded_layers': ['O'],
        'perturbation_type': 'lhs',
        'exclude_active_form': True
    },
    'simulation': {
        'start': 0,
        'stop': 1000,
        'step': 100
    },
    'dynamic_data': {
        'exclude_activated_form': False,
        'excluded_layers': [],
        'distortion': True,
        'distortion_factor': 2
    },
    'machine_learning': {
        'ml_seed': 42,
        'outcome_var': 'Oa',
        'n_samples': 1000,  # number of samples used
        'n_reps': 10
    }
}

# Combine configurations
full_config = {
    'notebook': notebook_config,
    'exp': exp_config
}

# Save configuration to S3 using version number as config suffix
s3_manager.save_config(notebook_config, full_config, config_suffix=version_number)
print("✅ Configuration saved to S3")


# %%
# or load existing config using version number as config suffix
loaded_config = s3_manager.load_config(notebook_config, config_suffix=version_number)

# Print configuration for verification
def print_config(d, indent=0):
    for key, value in d.items():
        print(" " * indent + str(key) + ":", end=" ")
        if isinstance(value, dict):
            print()
            print_config(value, indent + 2)
        else:
            print(str(value))

print_config(loaded_config)

# %% [markdown]
# # Run

# %% [markdown]
# ### Setup

# %%
from models.Specs.ModelSpec4 import ModelSpec4

notebook_config = loaded_config["notebook"]
config_name = notebook_config['version_number']
exp_config = loaded_config["exp"]
spec_config = exp_config['spec']
n_layers = spec_config['n_layers']
new_spec = ModelSpec4(num_intermediate_layers=n_layers)


# %%
from models.Specs.Drug import Drug

drug_config = spec_config['drug']
drug_name = drug_config['name']
drug_start = drug_config['start']
drug_dose = drug_config['dose']
drug_regulations = drug_config['regulations']


n_cascades = spec_config["n_cascades"]
n_regs = spec_config["n_regs"]
seed = spec_config["gen_seed"]

for regulation in spec_config["custom_regulations"]:
    new_spec.add_regulation(*regulation)

new_drug = Drug(name=drug_name, start_time=drug_start, default_value=drug_dose)

# check if target_all exists in drug_config, if not set to False
drug_target_all = drug_config.get('target_all', False)

if drug_target_all:
    # If the drug targets all receptors, we don't need to add specific regulations
    for n in range(n_cascades):
        target = f'R{n+1}' # assuming receptors are named R1, R2, ..., Rn
        new_drug.add_regulation(target, 'down') # assuming the type is 'down' for all
else: 
    for regs in drug_regulations:
        target, type = regs[0], regs[1]
        new_drug.add_regulation(target, type)

new_spec.generate_specifications(n_cascades, n_regs, seed)
new_spec.add_drug(new_drug)

# %%
import numpy as np

param_sampling_config = exp_config["parameter_sampling"]
base_sampling_seed = param_sampling_config["sampling_seed"]
num_models = param_sampling_config["num_models"]
# based on base seed create 1000 different seeds
seeds = np.random.default_rng(base_sampling_seed).integers(0, 1000000, size=num_models)

param_gen_config = exp_config['parameter_generation']
basal_activation = spec_config["basal_activation"]
specie_range = param_gen_config['ic_range']
param_range = param_gen_config['param_range']
param_mul_range = param_gen_config['param_mul_range']

builder_models = []
for seed in seeds:
    builder = new_spec.generate_network(
        config_name,
        specie_range,
        param_range,
        param_mul_range,
        seed,
        receptor_basal_activation=basal_activation,
    )
    builder_models.append(builder)
print(f"Generated {len(builder_models)} models.")

# %% [markdown]
# ### Simulations with Caching

# %%
from models.Solver.RoadrunnerSolver import RoadrunnerSolver
from tqdm import tqdm
import pandas as pd

def validate_simulation_cache(sim_results_df, sim_results, expected_model_count):
    """
    Validate that cached simulation results match expected structure
    """
    if len(sim_results) != expected_model_count:
        return False
    if sim_results_df is None or len(sim_results_df) == 0:
        return False
    if 'seed' not in sim_results_df.columns:
        return False
    return True

def load_or_generate_simulation_results(notebook_config, builder_models, seeds, sim_config):
    """
    Load existing simulation results if cached, otherwise generate new ones
    Returns: sim_results_df, sim_results, solver
    """
    try:
        # Try to load existing cached results
        sim_results_df = s3_manager.load_data(notebook_config, "sim_results_df", "pkl")
        sim_results = s3_manager.load_data(notebook_config, "sim_results", "pkl")
        
        # Validate that loaded data matches expected dimensions
        if validate_simulation_cache(sim_results_df, sim_results, len(builder_models)):
            print("✅ Loaded existing simulation results from cache")
            
            # For cached results, create a minimal solver from the first model
            solver = RoadrunnerSolver()
            solver.compile(builder_models[0].get_sbml_model())
            return sim_results_df, sim_results, solver
        else:
            print("⚠️ Cached results invalid, regenerating...")
            raise ValueError("Cache validation failed")
            
    except (FileNotFoundError, ValueError, Exception) as e:
        print(f"⚠️ No valid cached results found ({e}), generating new simulations...")
        
        # Create solvers
        solver_models = []
        for i, builder in enumerate(tqdm(builder_models, desc="Creating solvers")):
            solver = RoadrunnerSolver()
            solver.compile(builder.get_sbml_model())
            solver_models.append(solver)
        
        # Run simulations
        sim_start = sim_config["start"]
        sim_stop = sim_config["stop"]
        sim_step = sim_config["step"]
        
        sim_results = []
        for i, solver in enumerate(tqdm(solver_models, desc="Simulating models")):
            res = solver.simulate(sim_start, sim_stop, sim_step)
            sim_results.append(res)
        
        # Create dataframe
        sim_results_df = pd.concat(
            [pd.DataFrame(res).assign(seed=seed) for res, seed in zip(sim_results, seeds)],
            ignore_index=True,
        )
        
        # Cache for future use in both pickle and parquet formats
        s3_manager.save_data(notebook_config, sim_results_df, "sim_results_df", "pkl")
        s3_manager.save_data(notebook_config, sim_results, "sim_results", "pkl")
        
        # Also save sim_results_df in parquet format
        try:
            s3_manager.save_data(notebook_config, sim_results_df, "sim_results_df", "parquet")
            print("✅ sim_results_df saved as parquet format")
        except Exception as e:
            print(f"⚠️ Could not save sim_results_df as parquet: {e}")
        
        print("✅ New simulation results generated and cached")
        
        # Return the first solver for further processing
        return sim_results_df, sim_results, solver_models[0]

# Execute the caching simulation function
sim_config = exp_config["simulation"]
sim_results_df, sim_results, simulation_solver = load_or_generate_simulation_results(
    notebook_config, builder_models, seeds, sim_config
)


# %%
# save the results using S3 manager
s3_manager.save_data(
    notebook_config=notebook_config,
    data=sim_results_df,
    data_name="sim_results_df",
    data_format="pkl",
)

s3_manager.save_data(
    notebook_config=notebook_config,
    data=sim_results,
    data_name="sim_results",
    data_format="pkl",
)

print("✅ Simulation results saved to S3")

# %%
# Verify files are in S3
files = s3_manager.list_experiment_files(notebook_config)
print("Files in S3:")
for category, file_list in files.items():
    if file_list:
        print(f"{category.capitalize()}:")
        for file in file_list:
            print(f"  - {file}")

# %% [markdown]
# # Bulk Processing Pipeline

# %%
from models.SyntheticGen import generate_feature_data_v3, generate_target_data_diff_build, generate_model_timecourse_data_diff_build_v3
from models.utils.dynamic_calculations import dynamic_features_method, last_time_point_method
from numpy.random import default_rng
import pandas as pd

# %% [markdown]
# ## Enhanced Feature Data Generation

# %%
def generate_enhanced_feature_data(model_spec, builder_models, exp_config):
    """
    Generate feature data with optional parameter inclusion and layer exclusions
    """
    feature_config = exp_config['feature_generation']
    param_gen_config = exp_config['parameter_generation']
    ml_config = exp_config['machine_learning']
    
    # Get initial values from first model with exclusions
    builder = builder_models[0]  # Use first model as reference
    initial_values = builder.get_state_variables()
    
    if feature_config.get('exclude_active_form', True):
        initial_values = {k: v for k, v in initial_values.items() if not k.endswith('a')}
    if feature_config.get('excluded_layers'):
        for layer in feature_config['excluded_layers']:
            initial_values = {k: v for k, v in initial_values.items() if not k.startswith(layer)}
    
    # Generate base feature data
    feature_data = generate_feature_data_v3(
        model_spec, initial_values, 
        feature_config['perturbation_type'],
        {'min': param_gen_config['ic_range'][0], 'max': param_gen_config['ic_range'][1]},
        ml_config['n_samples'],
        seed=ml_config['ml_seed']
    )
    
    return feature_data

# %% [markdown]
# ## Parameter Extraction from Builder Models

# %%
def extract_parameter_sets_from_builders(builder_models):
    """
    Extract parameter sets from all builder models for processing
    """
    parameter_sets = []
    for builder in builder_models:
        params = builder.get_parameters()
        parameter_sets.append(params)
    return parameter_sets

# %% [markdown]
# ## Parameter Distortion for Robustness

# %%
def apply_parameter_distortion(parameter_sets, exp_config):
    """
    Apply parameter distortion for robustness testing
    """
    dynamic_config = exp_config['dynamic_data']
    
    if not dynamic_config.get('distortion', False):
        return parameter_sets
    
    distortion_factor = dynamic_config['distortion_factor']
    distort_range = (1 / distortion_factor, distortion_factor)
    
    ml_seed = exp_config['machine_learning']['ml_seed']
    rng = default_rng(ml_seed)
    
    modified_parameter_sets = []
    for params in parameter_sets:
        new_params = {}
        for key, value in params.items():
            new_params[key] = value * rng.uniform(distort_range[0], distort_range[1])
        modified_parameter_sets.append(new_params)
    
    return modified_parameter_sets

# %% [markdown]
# ## Complete Bulk Processing Pipeline

# %%
def complete_bulk_processing_pipeline(model_spec, solver, builder_models, exp_config):
    """
    Complete pipeline including feature generation, distortion, and all data processing
    """
    print("🚀 Starting complete bulk processing pipeline...")
    
    # Extract parameters from builder models
    parameter_sets = extract_parameter_sets_from_builders(builder_models)
    
    # Generate enhanced feature data
    feature_data = generate_enhanced_feature_data(model_spec, builder_models, exp_config)
    
    # Apply parameter distortion if enabled
    processed_parameter_sets = apply_parameter_distortion(parameter_sets, exp_config)
    
    # Simulation parameters
    sim_config = exp_config['simulation']
    sim_params = {'start': sim_config['start'], 'end': sim_config['stop'], 'points': sim_config['step']}
    
    # Generate target and timecourse data
    outcome_var = exp_config['machine_learning']['outcome_var']
    print("📊 Generating target and timecourse data...")
    target_data, timecourse_data = generate_target_data_diff_build(
        model_spec, solver, feature_data, processed_parameter_sets, sim_params,
        outcome_var=outcome_var, verbose=True
    )
    
    # Generate enhanced timecourse data for dynamic features
    print("📈 Generating enhanced timecourse data for all species...")
    enhanced_timecourse = generate_model_timecourse_data_diff_build_v3(
        builder_models[0].get_state_variables(), solver, feature_data, processed_parameter_sets, 
        sim_params, capture_species="all", verbose=True
    )
    
    # Calculate dynamic features with dynamic config exclusions
    print("⚡ Calculating dynamic features...")
    dynamic_config = exp_config['dynamic_data']
    initial_values = builder_models[0].get_state_variables()
    if dynamic_config.get('exclude_activated_form', False):
        initial_values = {k: v for k, v in initial_values.items() if not k.endswith('a')}
    if dynamic_config.get('excluded_layers'):
        for layer in dynamic_config['excluded_layers']:
            initial_values = {k: v for k, v in initial_values.items() if not k.startswith(layer)}
    
    dynamic_data = dynamic_features_method(enhanced_timecourse, initial_values.keys(), n_cores=4)
    
    # Calculate last time point data
    print("⏱️ Calculating last time point data...")
    last_time_data = last_time_point_method(enhanced_timecourse, initial_values.keys())
    
    print("✅ Complete bulk processing finished!")
    return feature_data, target_data, enhanced_timecourse, dynamic_data, last_time_data

# %% [markdown]
# ## Run Bulk Processing

# %%
# Execute the complete bulk processing pipeline
feature_data, target_data, timecourse_data, dynamic_data, last_time_data = complete_bulk_processing_pipeline(
    new_spec, simulation_solver, builder_models, exp_config
)

# %% [markdown]
# ## Save Processed Data

# %%
# Save all processed data types in both pickle and parquet formats
data_types = {
    'feature_data': feature_data,
    'target_data': target_data, 
    'timecourse_data': timecourse_data,
    'dynamic_data': dynamic_data,
    'last_time_data': last_time_data
}

for data_name, data in data_types.items():
    # Save as pickle format
    s3_manager.save_data(
        notebook_config=notebook_config,
        data=data,
        data_name=data_name,
        data_format="pkl",
    )
    print(f"✅ {data_name} saved as pickle format")
    
    # Save as parquet format (if the data is a pandas DataFrame)
    try:
        if hasattr(data, 'to_parquet') or isinstance(data, (pd.DataFrame, pd.Series)):
            s3_manager.save_data(
                notebook_config=notebook_config,
                data=data,
                data_name=data_name,
                data_format="parquet",
            )
            print(f"✅ {data_name} saved as parquet format")
    except Exception as e:
        print(f"⚠️ Could not save {data_name} as parquet: {e}")

# Verify all files are saved
print("📁 Final file list in S3:")
files = s3_manager.list_experiment_files(notebook_config)
for category, file_list in files.items():
    if file_list:
        print(f"{category.capitalize()}:")
        for file in file_list:
            print(f"  - {file}")
