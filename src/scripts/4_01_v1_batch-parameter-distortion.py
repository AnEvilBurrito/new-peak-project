"""
Batch Parameter Distortion Script

Executable script for batch generation of datasets with parameter distortion
following the sequential re-assembly pattern.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import time

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
from ml.Workflow import batch_eval_standard
from numpy.random import SeedSequence, default_rng
from joblib import Parallel, delayed
from scripts.ntfy_notifier import notify_start, notify_success, notify_failure


def load_experiment_config(s3_manager):
    """
    Load experiment configuration for parameter distortion using single source of truth
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
    
    # Include clean baseline (0 distortion) + specified distortion levels
    batch_config = {
        'distortion_range': [0, 1.1, 1.3, 1.5, 2.0, 3.0]  # Include baseline (0) + distortion factors
    }
    
    # Merge base config with batch-specific parameters
    full_config = {**base_config, **batch_config}
    
    return full_config


def setup_model_and_drug(config):
    """Set up model specification and drug configuration"""
    spec_config = config['spec']
    n_layers = spec_config['n_layers']
    
    # Create model specification
    new_spec = ModelSpec3(num_intermediate_layers=n_layers)
    
    # Set up drug
    drug_config = spec_config['drug']
    drug_name = drug_config['name']
    drug_start = drug_config['start']
    drug_dose = drug_config['dose']
    drug_regulations = drug_config['regulations']
    
    n_cascades = spec_config["n_cascades"]
    n_regs = spec_config["n_regs"]
    seed = spec_config["gen_seed"]
    
    new_drug = Drug(name=drug_name, start_time=drug_start, default_value=drug_dose)
    
    # Add drug regulations
    drug_target_all = drug_config.get('target_all', False)
    if drug_target_all:
        for n in range(n_cascades):
            target = f'R{n+1}'
            new_drug.add_regulation(target, 'down')
    else: 
        for regs in drug_regulations:
            target, type = regs[0], regs[1]
            new_drug.add_regulation(target, type)
    
    # Generate specifications and add drug
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
    
    # Base build for reference
    builder_base = new_spec.generate_network('base', specie_range, param_range, param_mul_range, seed)
    
    # Generate multiple parameter sets with different seeds
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


def apply_parameter_distortion(parameter_sets, distortion_factor, seed):
    """Apply parameter distortion with given factor"""
    # Handle baseline case (distortion_factor == 0) - return original parameters
    if distortion_factor == 0:
        return parameter_sets  # No distortion for baseline
        
    # Apply distortion only when factor > 0
    distort_range = (1 / distortion_factor, distortion_factor)
    rng = default_rng(seed)
    
    modified_parameter_sets = []
    for params in parameter_sets:
        new_params = {}
        for key, value in params.items():
            new_params[key] = value * rng.uniform(distort_range[0], distort_range[1])
        modified_parameter_sets.append(new_params)
    
    return modified_parameter_sets


def generate_markdown_report(final_results, execution_times, config, notebook_config, total_duration):
    """Generate markdown report with execution summary and data preview"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get data preview statistics
    if final_results is not None:
        data_shape = final_results.shape
        distortion_factors = final_results['Distortion Factor'].unique()
        models = final_results['Model'].unique() if 'Model' in final_results.columns else []
        feature_types = final_results['Feature Data'].unique() if 'Feature Data' in final_results.columns else []
    else:
        data_shape = (0, 0)
        distortion_factors = []
        models = []
        feature_types = []
    
    # Format execution times
    time_summary = "\n".join([f"- Distortion factor {factor}: {duration:.2f} seconds" 
                             for factor, duration in execution_times.items()])
    
    report = f"""# Batch Parameter Distortion Execution Report

**Experiment**: Section 4 / Experiment 01 / Version 1  
**Notebook**: batch-parameter-distortion  
**Execution Timestamp**: {timestamp}  
**Total Duration**: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)

## Execution Summary

**Data Generated**: {data_shape[0]} rows × {data_shape[1]} columns  
**Distortion Factors Processed**: {', '.join(map(str, distortion_factors))}  
**Total Distortion Factors**: {len(distortion_factors)}

### Execution Time Breakdown
{time_summary}

## Configuration Summary

- **Number of Samples**: {config['machine_learning']['n_samples']}
- **Number of Repeats**: {config['machine_learning']['n_reps']}
- **Distortion Range**: {config['distortion_range']}
- **Models Used**: {len(models)} models

## Data Preview

### First 5 Rows
```
{final_results.head().to_string() if final_results is not None else 'No data generated'}
```

### Summary Statistics by Distortion Factor
```
{final_results.groupby('Distortion Factor').describe().to_string() if final_results is not None else 'No data generated'}
```

## Storage Information

- **S3 Base Path**: {notebook_config.get('s3_base_path', 'Not specified')}
- **Report Location**: {notebook_config.get('version', 'v1')}_parameter_distortion_report.md
- **Data Files**: Uploaded to S3 data folder

## Success Summary

✅ Batch execution completed successfully  
✅ {len(distortion_factors)} distortion factors processed  
✅ {data_shape[0]} total data rows generated  
✅ Report and data uploaded to S3 storage

*Generated automatically by batch execution system*
"""
    
    return report


def run_batch_parameter_distortion():
    """Main batch execution function for parameter distortion"""
    print("🚀 Starting Batch Parameter Distortion Execution")
    start_time = time.time()
    
    # Send start notification
    script_name = 'batch-parameter-distortion'
    notify_start(script_name)
    
    # Initialize batch framework
    batch_executor = create_batch_executor(
        notebook_name='batch-parameter-distortion',
        exp_number='01',
        version_number='v1',
        section_number='4'
    )
    
    # Get S3 manager from batch executor
    s3_manager = batch_executor.storage_manager
    
    # Load configuration using S3 manager
    config = load_experiment_config(s3_manager)
    
    # Track execution times per distortion factor
    execution_times = {}
    
    try:
        # Check which distortion factors are still pending
        distortion_range = config['distortion_range']
        pending_factors = batch_executor.get_pending_assemblies(distortion_range)
        
        if not pending_factors:
            print("✅ All distortion factors have already been processed")
            return None
        
        print(f"Pending distortion factors: {pending_factors}")
        
        # Setup model and generate parameter sets (only once per batch)
        new_spec, new_drug = setup_model_and_drug(config)
        builder_base, parameter_sets, all_builds = generate_parameter_sets(new_spec, config)
        
        # Setup solver
        solver = RoadrunnerSolver()
        solver.compile(builder_base.get_sbml_model())
        
        # Generate feature data
        sim_config = config["simulation"]
        feature_config = config["feature_generation"]
        n_samples = config["machine_learning"]["n_samples"]
        ic_range = config["parameter_generation"]["ic_range"]
        
        initial_values = builder_base.get_state_variables()
        initial_values = {k: v for k, v in initial_values.items() if not k.endswith('a')}
        
        if feature_config["excluded_layers"]:
            for layer in feature_config["excluded_layers"]:
                initial_values = {k: v for k, v in initial_values.items() if not k.startswith(f'{layer}')}
        
        feature_data = generate_feature_data_v3(
            new_spec, initial_values, 'lhs', 
            {'min': ic_range[0], 'max': ic_range[1]}, 
            n_samples, config['spec']['gen_seed']
        )
        
        # Generate target data with original parameters
        sim_params = {'start': sim_config["start"], 'end': sim_config["stop"], 'points': sim_config["step"]}
        target_data, timecourse_data = generate_target_data_diff_build(
            new_spec, solver, feature_data, parameter_sets, sim_params, 
            outcome_var='Oa', verbose=False
        )
        
        # Process each pending distortion factor
        all_results = []
        
        for dis_factor in pending_factors:
            print(f"Processing distortion factor: {dis_factor}")
            factor_start_time = time.time()
            
            # Generate assembly ID for this specific config value
            assembly_id = batch_executor.generate_assembly_id(dis_factor)
            print(f"Assembly ID: {assembly_id}")
            
            # Apply parameter distortion
            modified_parameter_sets = apply_parameter_distortion(
                parameter_sets, dis_factor, config['spec']['gen_seed']
            )
            
            # Generate distorted timecourse data
            distorted_timecourse = generate_model_timecourse_data_diff_build_v3(
                builder_base.get_state_variables(),
                solver,
                feature_data,
                modified_parameter_sets,
                sim_params,
                capture_species="all",
                n_cores=1,
                verbose=False,
            )
            
            # Calculate dynamic features
            dynamic_config = config['dynamic_data']
            initial_values = builder_base.get_state_variables()
            if dynamic_config["exclude_activated_form"]:
                initial_values = {k: v for k, v in initial_values.items() if not k.endswith('a')}
            if dynamic_config["excluded_layers"]:
                for layer in dynamic_config["excluded_layers"]:
                    initial_values = {k: v for k, v in initial_values.items() if not k.startswith(f'{layer}')}
            
            last_time_data = last_time_point_method(distorted_timecourse, initial_values.keys())
            dynamic_data = dynamic_features_method(distorted_timecourse, initial_values.keys(), n_cores=4)
            
            # Combine feature data types
            combined_lp_data = pd.concat([feature_data, last_time_data], axis=1)
            combined_dyn_data = pd.concat([feature_data, dynamic_data], axis=1)
            feature_data_list = [feature_data, last_time_data, dynamic_data, combined_lp_data, combined_dyn_data]
            feature_data_names = ['feature_data', 'last_time_data', 'dynamic_data', 'combined_lp_data', 'combined_dyn_data']
            
            # Handle baseline (0 distortion) specially - generate baseline timecourse data properly
            if dis_factor == 0:
                # Baseline: use original parameters (no distortion) - generate proper DataFrame
                print("📊 Processing baseline (clean data)")
                # Generate baseline timecourse using original parameters
                baseline_timecourse = generate_model_timecourse_data_diff_build_v3(
                    builder_base.get_state_variables(),
                    solver,
                    feature_data,
                    parameter_sets,  # Original parameters for baseline
                    sim_params,
                    capture_species="all",
                    n_cores=1,
                    verbose=False,
                )
                baseline_last_time_data = last_time_point_method(baseline_timecourse, initial_values.keys())
                baseline_dynamic_data = dynamic_features_method(baseline_timecourse, initial_values.keys(), n_cores=4)
                
                baseline_combined_lp_data = pd.concat([feature_data, baseline_last_time_data], axis=1)
                baseline_combined_dyn_data = pd.concat([feature_data, baseline_dynamic_data], axis=1)
                baseline_feature_data_list = [feature_data, baseline_last_time_data, baseline_dynamic_data, baseline_combined_lp_data, baseline_combined_dyn_data]
                baseline_feature_data_names = ['feature_data', 'last_time_data', 'dynamic_data', 'combined_lp_data', 'combined_dyn_data']
            else:
                # Distorted case: use distorted timecourse data (already proper DataFrame)
                baseline_last_time_data = last_time_data
                baseline_dynamic_data = dynamic_data
                baseline_combined_lp_data = combined_lp_data
                baseline_combined_dyn_data = combined_dyn_data
                baseline_feature_data_list = feature_data_list
                baseline_feature_data_names = feature_data_names
            
            # Use ml.Workflow as single source of truth for ML evaluation
            ml_seed = config['machine_learning']['ml_seed']
            outcome_var = config['machine_learning']['outcome_var']
            n_reps = config['machine_learning']['n_reps']
            
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
            metric_df['Distortion Factor'] = dis_factor
            metric_df['Assembly ID'] = assembly_id
            metric_df['Timestamp'] = datetime.now().isoformat()
            
            # Save results for this specific config value
            batch_executor.save_batch_data(metric_df, 'parameter_distortion_results')
            batch_executor.mark_assembly_completed()
            
            # Record execution time for this distortion factor
            factor_duration = time.time() - factor_start_time
            execution_times[dis_factor] = factor_duration
            all_results.append(metric_df)
            print(f"✅ Completed processing for distortion factor {dis_factor} ({factor_duration:.2f}s)")
        
        # Calculate total execution time
        total_duration = time.time() - start_time
        
        # Combine all results if needed for return
        if all_results:
            final_results = pd.concat(all_results, ignore_index=True)
            print(f"✅ Batch execution completed successfully")
            print(f"Processed {len(pending_factors)} distortion factors")
            print(f"Final results shape: {final_results.shape}")
            print(f"Total execution time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
            
            # Send success notification
            notify_success(script_name, total_duration, processed_count=len(pending_factors))
            
            # Generate and upload markdown report
            try:
                # Create notebook config for S3 upload
                notebook_config = {
                    'notebook_name': 'batch-parameter-distortion',
                    'exp_number': '01',
                    'version_number': 'v1',
                    'section_number': '4',
                    's3_base_path': s3_manager._get_s3_key({
                        'notebook_name': 'batch-parameter-distortion',
                        'exp_number': '01',
                        'version_number': 'v1',
                        'section_number': '4'
                    })
                }
                
                # Generate report content
                report_content = generate_markdown_report(final_results, execution_times, config, notebook_config, total_duration)
                
                # Upload report to S3 report folder
                report_key = s3_manager._get_s3_key(notebook_config, subfolder='report', filename='4_01_v1_parameter_distortion_report.md')
                s3_manager._upload_with_progress(report_content, report_key, content_type='text/markdown')
                print(f"✅ Report uploaded to S3: {report_key}")
                
            except Exception as report_error:
                print(f"⚠️ Failed to generate/upload report: {report_error}")
            
            return final_results
        else:
            print("❌ No results were generated")
            # Still send success notification? No results but execution completed.
            notify_success(script_name, total_duration, processed_count=0)
            return None
        
    except Exception as e:
        print(f"❌ Batch execution failed: {e}")
        # Send failure notification
        notify_failure(script_name, e, duration_seconds=time.time() - start_time)
        raise


def sequential_reassembly_example():
    """Example of how to use sequential re-assembly"""
    batch_executor = create_batch_executor(
        notebook_name='batch-parameter-distortion',
        exp_number='01',
        version_number='v1',
        section_number='4'
    )
    
    try:
        combined_data = batch_executor.sequential_reassembly('parameter_distortion_results')
        print(f"Re-assembled data shape: {combined_data.shape}")
        return combined_data
    except Exception as e:
        print(f"Re-assembly failed: {e}")
        return None


if __name__ == "__main__":
    # Execute batch processing
    results = run_batch_parameter_distortion()
    
    # Optionally demonstrate re-assembly
    # reassembled_data = sequential_reassembly_example()
