"""
Batch Expression Noise Script

Executable script for batch generation of datasets with Gaussian expression noise
following the sequential re-assembly pattern.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..')
sys.path.insert(0, src_dir)

# Import project modules
from scripts.batch_framework import create_batch_executor
from models.Specs.ModelSpec3 import ModelSpec3
from models.Specs.Drug import Drug
from models.Solver.RoadrunnerSolver import RoadrunnerSolver
from models.SyntheticGen import (
    generate_feature_data_v3, 
    generate_target_data_diff_build,
    generate_model_timecourse_data_diff_build_v3
)
from models.utils.dynamic_calculations import dynamic_features_method, last_time_point_method
from numpy.random import SeedSequence, default_rng
from ml.Workflow import batch_eval_standard


def load_experiment_config(s3_manager):
    """
    Load experiment configuration for expression noise using single source of truth
    Base config loaded from S3, with batch-specific parameters appended
    """
    # Load base configuration from synthetic cohort generation (single source of truth)
    synthetic_cohort_config = {
        'notebook_name': 'diverse-synthetic-cohort-generation',
        'exp_number': '01',
        'version_number': 'v1',
        'section_number': '1'
    }
    
    # Fail-fast: if S3 config cannot be loaded, raise error immediately
    full_config = s3_manager.load_config(synthetic_cohort_config, config_suffix='v1')
    print("✅ Loaded base configuration from S3")
    
    # Extract the exp configuration which contains all the experiment parameters
    if 'exp' in full_config:
        base_config = full_config['exp']
    else:
        raise ValueError("Config does not have expected 'exp' structure")
    
    # Include clean baseline (0 noise) + specified noise levels
    batch_config = {
        'expression_noise_levels': [0, 0.1, 0.2, 0.3, 0.5, 1.0]  # Include baseline (0) + noise levels
    }
    
    # Merge base config with batch-specific parameters
    full_config = {**base_config, **batch_config}
    
    return full_config


def setup_model_and_drug(config):
    """Set up model specification and drug configuration"""
    spec_config = config['spec']
    n_layers = spec_config['n_layers']
    
    new_spec = ModelSpec3(num_intermediate_layers=n_layers)
    
    drug_config = spec_config['drug']
    drug_name = drug_config['name']
    drug_start = drug_config['start']
    drug_dose = drug_config['dose']
    drug_regulations = drug_config['regulations']
    
    n_cascades = spec_config["n_cascades"]
    n_regs = spec_config["n_regs"]
    seed = spec_config["gen_seed"]
    
    new_drug = Drug(name=drug_name, start_time=drug_start, default_value=drug_dose)
    
    drug_target_all = drug_config.get('target_all', False)
    if drug_target_all:
        for n in range(n_cascades):
            target = f'R{n+1}'
            new_drug.add_regulation(target, 'down')
    else: 
        for regs in drug_regulations:
            target, type = regs[0], regs[1]
            new_drug.add_regulation(target, type)
    
    new_spec.generate_specifications(n_cascades, n_regs, seed)
    new_spec.add_drug(new_drug)
    
    return new_spec, new_drug


def generate_parameter_sets(new_spec, config):
    """Generate multiple parameter sets for the models"""
    param_gen_config = config['parameter_generation']
    specie_range = param_gen_config['ic_range']
    param_range = param_gen_config['param_range']
    param_mul_range = param_gen_config['param_mul_range']
    n_samples = config['machine_learning']['n_samples']
    seed = config['spec']['gen_seed']
    
    builder_base = new_spec.generate_network('base', specie_range, param_range, param_mul_range, seed)
    
    ss = SeedSequence(seed)
    build_seeds = []
    for i in range(n_samples):
        child_ss = ss.spawn(1)[0]
        build_seeds.append(child_ss.generate_state(1)[0])
    
    parameter_sets = []
    all_builds = []
    for seed_val in build_seeds:
        builder = new_spec.generate_network(str(seed_val), specie_range, param_range, param_mul_range, seed_val)
        parameter_sets.append(builder.get_parameters())
        all_builds.append(builder)
    
    return builder_base, parameter_sets, all_builds


def apply_expression_noise(feature_data, noise_level, seed):
    """
    Apply Gaussian noise to expression feature data
    
    Args:
        feature_data: Original feature data DataFrame
        noise_level: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility
    
    Returns:
        Feature data with applied noise
    """
    # Handle baseline case (noise_level == 0) - return original data unchanged
    if noise_level == 0:
        return feature_data  # No noise for baseline
        
    rng = default_rng(seed)
    
    # Apply noise to each column independently
    noisy_feature_data = feature_data.copy()
    for column in feature_data.columns:
        original_values = feature_data[column].values
        noise = rng.normal(0, noise_level * np.std(original_values), len(original_values))
        noisy_feature_data[column] = original_values + noise
    
    return noisy_feature_data


def run_batch_expression_noise():
    """Main batch execution function for expression noise"""
    print("🚀 Starting Batch Expression Noise Execution")
    
    # Initialize batch framework
    batch_executor = create_batch_executor(
        notebook_name='batch-expression-noise',
        exp_number='02',
        version_number='v1',
        section_number='4'
    )
    
    # Get S3 manager from batch executor
    s3_manager = batch_executor.storage_manager
    
    # Load configuration using S3 manager
    config = load_experiment_config(s3_manager)
    
    try:
        # Check which noise levels are still pending
        noise_levels = config['expression_noise_levels']
        pending_levels = batch_executor.get_pending_assemblies(noise_levels)
        
        if not pending_levels:
            print("✅ All noise levels have already been processed")
            return None
        
        print(f"Pending noise levels: {pending_levels}")
        
        # Setup model and generate parameter sets (only once per batch)
        new_spec, new_drug = setup_model_and_drug(config)
        builder_base, parameter_sets, all_builds = generate_parameter_sets(new_spec, config)
        
        # Setup solver
        solver = RoadrunnerSolver()
        solver.compile(builder_base.get_sbml_model())
        
        # Generate base feature data (without noise)
        sim_config = config["simulation"]
        feature_config = config["feature_generation"]
        n_samples = config["machine_learning"]["n_samples"]
        ic_range = config["parameter_generation"]["ic_range"]
        
        initial_values = builder_base.get_state_variables()
        initial_values = {k: v for k, v in initial_values.items() if not k.endswith('a')}
        
        if feature_config["excluded_layers"]:
            for layer in feature_config["excluded_layers"]:
                initial_values = {k: v for k, v in initial_values.items() if not k.startswith(f'{layer}')}
        
        base_feature_data = generate_feature_data_v3(
            new_spec, initial_values, 'lhs', 
            {'min': ic_range[0], 'max': ic_range[1]}, 
            n_samples, config['spec']['gen_seed']
        )
        
        # Generate target data with original parameters (clean data)
        sim_params = {'start': sim_config["start"], 'end': sim_config["stop"], 'points': sim_config["step"]}
        target_data, timecourse_data = generate_target_data_diff_build(
            new_spec, solver, base_feature_data, parameter_sets, sim_params, 
            outcome_var='Oa', verbose=False
        )
        
        # Process each pending noise level
        all_results = []
        
        for noise_level in pending_levels:
            print(f"Processing expression noise level: {noise_level}")
            
            # Generate assembly ID for this specific config value
            assembly_id = batch_executor.generate_assembly_id(noise_level)
            print(f"Assembly ID: {assembly_id}")
            
            # Apply expression noise to feature data
            noisy_feature_data = apply_expression_noise(
                base_feature_data, noise_level, config['spec']['gen_seed']
            )
            
            # Generate timecourse data with noisy features
            noisy_timecourse = generate_model_timecourse_data_diff_build_v3(
                builder_base.get_state_variables(),
                solver,
                noisy_feature_data,
                parameter_sets,  # Using original parameters, only features are noisy
                sim_params,
                capture_species="all",
                n_cores=1,
                verbose=False,
            )
            
            # Calculate dynamic features from noisy timecourse
            dynamic_config = config['dynamic_data']
            initial_values = builder_base.get_state_variables()
            if dynamic_config["exclude_activated_form"]:
                initial_values = {k: v for k, v in initial_values.items() if not k.endswith('a')}
            if dynamic_config["excluded_layers"]:
                for layer in dynamic_config["excluded_layers"]:
                    initial_values = {k: v for k, v in initial_values.items() if not k.startswith(f'{layer}')}
            
            last_time_data = last_time_point_method(noisy_timecourse, initial_values.keys())
            dynamic_data = dynamic_features_method(noisy_timecourse, initial_values.keys(), n_cores=4)
            
            # Combine all feature data types
            combined_lp_data = pd.concat([noisy_feature_data, last_time_data], axis=1)
            combined_dyn_data = pd.concat([noisy_feature_data, dynamic_data], axis=1)
            
            # Include both original and noisy versions for comparison
            feature_data_list = [
                base_feature_data,          # Original (clean) features
                noisy_feature_data,         # Noisy features only
                last_time_data,             # Last time point from noisy simulation
                dynamic_data,               # Dynamic features from noisy simulation
                combined_lp_data,           # Noisy features + last time
                combined_dyn_data           # Noisy features + dynamic features
            ]
            
            feature_data_names = [
                'original_feature_data',
                'noisy_feature_data', 
                'last_time_data',
                'dynamic_data',
                'combined_lp_data',
                'combined_dyn_data'
            ]
            
            # Use ml.Workflow as single source of truth for ML evaluation
            ml_seed = config['machine_learning']['ml_seed']
            outcome_var = config['machine_learning']['outcome_var']
            n_reps = config['machine_learning']['n_reps']
            
            # Handle baseline (0 noise) specially
            if noise_level == 0:
                # Baseline: use clean features for baseline
                print("📊 Processing baseline (clean data)")
                baseline_feature_data_list = [base_feature_data]
                baseline_feature_data_names = ['original_feature_data']
            else:
                # Noisy case: use full feature data list
                baseline_feature_data_list = feature_data_list
                baseline_feature_data_names = feature_data_names
            
            metric_df = batch_eval_standard(
                feature_data_list=baseline_feature_data_list,
                feature_data_names=baseline_feature_data_names,
                target_data=target_data,
                target_name=outcome_var,
                num_repeats=n_reps,
                o_random_seed=ml_seed,
                n_jobs=-1
            )
            
            # Create results DataFrame for this specific config value
            metric_df['Expression Noise Level'] = noise_level
            metric_df['Assembly ID'] = assembly_id
            metric_df['Timestamp'] = datetime.now().isoformat()
            
            # Save results for this specific config value
            batch_executor.save_batch_data(metric_df, 'expression_noise_results')
            batch_executor.mark_assembly_completed()
            
            all_results.append(metric_df)
            print(f"✅ Completed processing for noise level {noise_level}")
        
        # Combine all results if needed for return
        if all_results:
            final_results = pd.concat(all_results, ignore_index=True)
            print(f"✅ Batch execution completed successfully")
            print(f"Processed {len(pending_levels)} noise levels")
            print(f"Final results shape: {final_results.shape}")
            return final_results
        else:
            print("❌ No results were generated")
            return None
        
    except Exception as e:
        print(f"❌ Batch execution failed: {e}")
        raise


def sequential_reassembly_example():
    """Example of how to use sequential re-assembly"""
    batch_executor = create_batch_executor(
        notebook_name='batch-expression-noise',
        exp_number='02',
        version_number='v1',
        section_number='4'
    )
    
    try:
        combined_data = batch_executor.sequential_reassembly('expression_noise_results')
        print(f"Re-assembled data shape: {combined_data.shape}")
        return combined_data
    except Exception as e:
        print(f"Re-assembly failed: {e}")
        return None


if __name__ == "__main__":
    # Execute batch processing
    results = run_batch_expression_noise()
    
    # Optionally demonstrate re-assembly
    # reassembled_data = sequential_reassembly_example()
