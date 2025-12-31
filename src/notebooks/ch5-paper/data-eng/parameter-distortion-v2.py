"""
Complete Parameter Distortion Data Generation Script

Generates comprehensive datasets with distorted parameter sets including:
- Distorted parameters
- Feature data
- Target data with make_target_data_with_params
- Timecourse data
- Dynamic features
- Last time point data

Follows the complete S3 storage pattern of sy_simple-make-data-v1.py.
"""

import sys
import os
import argparse
from dotenv import dotenv_values
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "../../..")
sys.path.insert(0, src_dir)
config = dotenv_values(dotenv_path=src_dir)

from models.utils.s3_config_manager import S3ConfigManager
from models.utils.data_generation_helpers import make_target_data_with_params
from models.utils.dynamic_calculations import dynamic_features_method, last_time_point_method
from models.Solver.RoadrunnerSolver import RoadrunnerSolver
from scripts.ntfy_notifier import notify_start, notify_success, notify_failure
from numpy.random import default_rng
from tqdm import tqdm
import warnings

# Import shared utilities for CSV generation
from ml_task_utils import BaseTaskGenerator, save_task_csv, print_task_summary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParameterDistortionTaskGenerator(BaseTaskGenerator):
    """
    Task generator for parameter-distortion-v2 experiment pattern.
    
    This class encapsulates the pattern-specific logic for generating
    CSV task lists for parameter distortion experiments.
    """
    
    def __init__(self, model_name: str = "sy_simple"):
        super().__init__(model_name)
        self.experiment_type = "parameter-distortion-v2"
        self.distortion_factors = [0, 1.1, 1.3, 1.5, 2.0, 3.0]
        
    def get_levels(self):
        return self.distortion_factors
        
    def get_base_folder(self):
        return f"{self.model_name}_parameter_distortion_v2"
        
    def get_feature_files(self, distortion_factor):
        """Get feature files for a given distortion factor"""
        base_path = f"{self.get_base_folder()}/distortion_{distortion_factor}"
        
        return [
            {
                "path": f"{base_path}/features.pkl",
                "label": f"features_{distortion_factor}"
            },
            {
                "path": f"{base_path}/dynamic_features.pkl",
                "label": f"dynamic_features_{distortion_factor}"
            },
            {
                "path": f"{base_path}/dynamic_features_no_outcome.pkl",
                "label": f"dynamic_features_no_outcome_{distortion_factor}"
            },
            {
                "path": f"{base_path}/last_time_points.pkl",
                "label": f"last_time_points_{distortion_factor}"
            },
            {
                "path": f"{base_path}/last_time_points_no_outcome.pkl",
                "label": f"last_time_points_no_outcome_{distortion_factor}"
            }
        ]
        
    def get_target_files(self, distortion_factor):
        """Get target files for a given distortion factor"""
        base_path = f"{self.get_base_folder()}/distortion_{distortion_factor}"
        
        return [
            {
                "path": f"{base_path}/targets.pkl",
                "label": "original_targets"
            }
        ]


def apply_gaussian_distortion(original_params, distortion_factor, seed=42):
    """
    Apply Gaussian noise distortion to parameters
    
    Args:
        original_params: Dictionary of original parameters from model_builder
        distortion_factor: Controls strength of distortion (0 = no distortion)
        seed: Random seed for reproducible distortion
    
    Returns:
        Dictionary of distorted parameters
    """
    if distortion_factor == 0:
        return original_params  # No distortion for baseline
    
    rng = default_rng(seed)
    distorted_params = {}
    
    for key, value in original_params.items():
        # Apply Gaussian noise scaled by distortion factor
        relative_noise = rng.normal(loc=0, scale=distortion_factor)
        noise_amount = value * relative_noise
        distorted_params[key] = max(value + noise_amount, 1e-8)  # Ensure non-negative
    
    return distorted_params


def generate_distorted_parameter_sets(model_builder, distortion_factors, n_samples=2000, seed=42):
    """
    Generate multiple parameter sets with different distortion levels
    
    Args:
        model_builder: ModelBuilder instance
        distortion_factors: List of distortion factors to apply
        n_samples: Number of parameter sets per distortion factor
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary mapping distortion_factor -> list of parameter dictionaries
    """
    # Get original parameters from model
    original_params = model_builder.get_parameters()
    logger.info(f"Extracted {len(original_params)} parameters from model")
    
    # Generate different random seeds for each sample
    rng = default_rng(seed)
    sample_seeds = rng.integers(0, 2**32, size=n_samples)
    
    all_distorted_sets = {}
    
    for distortion_factor in distortion_factors:
        logger.info(f"Generating {n_samples} parameter sets with distortion factor {distortion_factor}")
        
        parameter_sets = []
        for i in range(n_samples):
            # Use different seed for each sample for variation
            sample_seed = int(sample_seeds[i])
            distorted_params = apply_gaussian_distortion(
                original_params, distortion_factor, sample_seed
            )
            parameter_sets.append(distorted_params)
        
        all_distorted_sets[distortion_factor] = parameter_sets
    
    return all_distorted_sets


def generate_feature_data(model_spec, initial_values, n_samples=2000, seed=42):
    """
    Generate feature data using lhs perturbation
    
    Args:
        model_spec: ModelSpecification instance
        initial_values: Dictionary of initial values (inactive state variables)
        n_samples: Number of samples
        seed: Random seed
    
    Returns:
        DataFrame of feature data
    """
    from models.utils.make_feature_data import make_feature_data
    
    feature_data = make_feature_data(
        initial_values=initial_values,
        perturbation_type='lhs',
        perturbation_params={'min': 0.1, 'max': 10.0},  # Using default range
        n_samples=n_samples,
        seed=seed
    )
    
    logger.info(f"Generated feature data with shape: {feature_data.shape}")
    return feature_data


def make_target_data_with_params_robust(
    model_spec,
    solver,
    feature_df,
    parameter_df,
    simulation_params,
    n_cores=1,
    outcome_var='Oa',
    capture_all_species=True,
    verbose=False
):
    """
    Robust version of make_target_data_with_params that handles CVODE errors
    by removing failed samples and maintaining data alignment
    
    Returns:
        Tuple of (target_df, timecourse_data, success_mask)
        where success_mask is boolean array indicating which samples succeeded
    """
    logger.info("Running robust simulation with CVODE error handling...")
    
    # First, try to run the standard function
    try:
        target_data, timecourse_data = make_target_data_with_params(
            model_spec=model_spec,
            solver=solver,
            feature_df=feature_df,
            parameter_df=parameter_df,
            simulation_params=simulation_params,
            n_cores=n_cores,
            outcome_var=outcome_var,
            capture_all_species=capture_all_species,
            verbose=verbose
        )
        
        # If successful, return all data with success_mask indicating all succeeded
        success_mask = pd.Series([True] * len(target_data), index=target_data.index)
        return target_data, timecourse_data, success_mask
        
    except RuntimeError as e:
        # Check if this is a CVODE error
        if "CV_TOO_MUCH_WORK" in str(e) or "CVODE" in str(e):
            logger.warning(f"CVODE error detected: {str(e)[:100]}...")
            logger.info("Falling back to robust sequential simulation with error handling")
            
            # Initialize result containers
            all_targets = []
            all_timecourses = []
            success_indices = []
            
            # Get simulation parameters
            start = simulation_params['start']
            end = simulation_params['end']
            points = simulation_params['points']
            
            # Sequential simulation with error handling
            for i in tqdm(range(feature_df.shape[0]), desc='Robust simulation', disable=not verbose):
                try:
                    # Get values for this sample
                    feature_values = feature_df.iloc[i].to_dict()
                    param_values = parameter_df.iloc[i].to_dict() if parameter_df is not None else None
                    
                    # Set values in solver
                    solver.set_state_values(feature_values)
                    if param_values is not None:
                        solver.set_parameter_values(param_values)
                    
                    # Run simulation
                    res = solver.simulate(start, end, points)
                    
                    # Extract results
                    target_value = res[outcome_var].iloc[-1]
                    
                    if capture_all_species:
                        # Capture all species
                        timecourse = {}
                        for col in res.columns:
                            if col != 'time':
                                timecourse[col] = res[col].values
                        all_timecourses.append(timecourse)
                    else:
                        # Capture only outcome variable
                        all_timecourses.append(res[outcome_var].values)
                    
                    all_targets.append(target_value)
                    success_indices.append(i)
                    
                except RuntimeError as sim_error:
                    # Check if this is a CVODE error
                    if "CV_TOO_MUCH_WORK" in str(sim_error) or "CVODE" in str(sim_error):
                        logger.debug(f"Sample {i} failed with CVODE error: {str(sim_error)[:50]}")
                        continue  # Skip this sample
                    else:
                        # Re-raise unexpected errors
                        raise
                except Exception as sim_error:
                    logger.debug(f"Sample {i} failed with error: {str(sim_error)[:50]}")
                    continue  # Skip this sample
            
            # Create success mask
            success_mask = pd.Series([False] * feature_df.shape[0])
            success_mask.iloc[success_indices] = True
            
            # Create target DataFrame
            target_data = pd.DataFrame(all_targets, columns=[outcome_var])
            target_data.index = success_indices
            
            # Create timecourse DataFrame if capture_all_species
            if capture_all_species and all_timecourses:
                timecourse_data = pd.DataFrame(all_timecourses)
                timecourse_data.index = success_indices
            else:
                timecourse_data = all_timecourses  # List of arrays
            
            logger.info(f"Robust simulation completed: {len(success_indices)}/{feature_df.shape[0]} samples succeeded")
            
            return target_data, timecourse_data, success_mask
            
        else:
            # Re-raise non-CVODE errors
            raise
            
    except Exception as e:
        # Re-raise other errors
        raise


def generate_complete_dataset_for_distortion_level(
    feature_data,
    parameter_df,
    model_spec,
    model_builder,
    solver,
    distortion_factor,
    simulation_params
):
    """
    Generate complete dataset for a specific distortion level
    
    Args:
        feature_data: DataFrame of feature data
        parameter_df: DataFrame of distorted parameters
        model_spec: ModelSpecification instance
        solver: Solver instance
        distortion_factor: Current distortion factor
        simulation_params: Simulation parameters
    
    Returns:
        Dictionary containing all dataset components
    """
    logger.info(f"Generating complete dataset for distortion factor {distortion_factor}")
    
    # Generate target and timecourse data using robust version
    logger.info("Generating target and timecourse data (robust)...")
    target_data, timecourse_data, success_mask = make_target_data_with_params_robust(
        model_spec=model_spec,
        solver=solver,
        feature_df=feature_data,
        parameter_df=parameter_df,
        simulation_params=simulation_params,
        n_cores=1,
        outcome_var='Oa',
        capture_all_species=True,
        verbose=False
    )
    
    # Filter all datasets to keep only successful samples
    feature_data = feature_data[success_mask].reset_index(drop=True)
    parameter_df = parameter_df[success_mask].reset_index(drop=True)
    target_data = target_data.reset_index(drop=True)
    
    # Reset timecourse data indices if it's a DataFrame
    if isinstance(timecourse_data, pd.DataFrame):
        timecourse_data = timecourse_data.reset_index(drop=True)
    
    # Calculate dynamic features
    logger.info("Calculating dynamic features...")
    initial_values = {k: v for k, v in model_builder.get_state_variables().items() if not k.endswith('a')}
    if 'O' in initial_values:
        del initial_values['O']
    
    dynamic_features = dynamic_features_method(
        timecourse_data, 
        selected_features=initial_values.keys(), 
        n_cores=1, 
        verbose=False
    )
    
    last_time_points = last_time_point_method(
        timecourse_data, 
        selected_species=initial_values.keys()
    )
    
    # Calculate dynamic features without outcome
    states_no_outcome = {k: v for k, v in model_builder.get_state_variables().items() 
                        if k not in ['Oa', 'O']}
    dynamic_features_no_outcome = dynamic_features_method(
        timecourse_data, 
        selected_features=states_no_outcome.keys(), 
        n_cores=1, 
        verbose=False
    )
    
    last_time_points_no_outcome = last_time_point_method(
        timecourse_data, 
        selected_species=states_no_outcome.keys()
    )
    
    return {
        'features': feature_data,
        'targets': target_data,
        'parameters': parameter_df,
        'timecourses': timecourse_data,
        'dynamic_features': dynamic_features,
        'last_time_points': last_time_points,
        'dynamic_features_no_outcome': dynamic_features_no_outcome,
        'last_time_points_no_outcome': last_time_points_no_outcome
    }


def save_complete_dataset(dataset_dict, distortion_factor, model_name, s3_manager):
    """
    Save complete dataset to S3 following sy_simple-make-data-v1.py pattern
    
    Args:
        dataset_dict: Dictionary containing all dataset components
        distortion_factor: Distortion factor for this dataset
        model_name: Name of the model
        s3_manager: S3ConfigManager instance
    """
    gen_path = s3_manager.save_result_path
    folder_name = f"{model_name}_parameter_distortion_v2"
    subfolder_name = f"distortion_{distortion_factor}"
    full_path = f"{gen_path}/data/{folder_name}/{subfolder_name}"
    
    # Save each component
    file_mappings = {
        'features': 'features.pkl',
        'targets': 'targets.pkl',
        'parameters': 'parameter_sets.pkl',
        'timecourses': 'timecourses.pkl',
        'dynamic_features': 'dynamic_features.pkl',
        'last_time_points': 'last_time_points.pkl',
        'dynamic_features_no_outcome': 'dynamic_features_no_outcome.pkl',
        'last_time_points_no_outcome': 'last_time_points_no_outcome.pkl'
    }
    
    for key, filename in file_mappings.items():
        if key in dataset_dict:
            s3_path = f"{full_path}/{filename}"
            s3_manager.save_data_from_path(s3_path, dataset_dict[key], data_format="pkl")
            logger.info(f"✅ Saved {key} to S3: {s3_path}")
    
    # Save metadata for this distortion level
    metadata = {
        'distortion_factor': distortion_factor,
        'model_name': model_name,
        'n_samples': dataset_dict['features'].shape[0],
        'generation_timestamp': datetime.now().isoformat(),
        'files_generated': list(file_mappings.keys())
    }
    
    metadata_path = f"{full_path}/metadata.pkl"
    s3_manager.save_data_from_path(metadata_path, metadata, data_format="pkl")
    logger.info(f"✅ Saved metadata to S3: {metadata_path}")


def load_model_objects(model_name, s3_manager):
    """
    Load all model objects from S3
    
    Args:
        model_name: Name of the model (e.g., 'sy_simple')
        s3_manager: S3ConfigManager instance
    
    Returns:
        Tuple of (model_spec, model_builder, model_tuner)
    """
    gen_path = s3_manager.save_result_path
    
    logger.info(f"Loading model objects for: {model_name}")
    
    model_spec = s3_manager.load_data_from_path(
        f"{gen_path}/models/{model_name}/model_spec.pkl", 
        data_format='pkl'
    )
    
    model_builder = s3_manager.load_data_from_path(
        f"{gen_path}/models/{model_name}/model_builder.pkl", 
        data_format='pkl'
    )
    
    model_tuner = s3_manager.load_data_from_path(
        f"{gen_path}/models/{model_name}/model_tuner.pkl", 
        data_format='pkl'
    )
    
    logger.info(f"✅ Loaded model objects for {model_name}")
    return model_spec, model_builder, model_tuner


def generate_csv_task_list():
    """
    Generate CSV task list for parameter distortion experiments.
    
    This function provides a command-line interface for generating
    CSV task lists without running the full data generation pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Generate CSV task list for parameter-distortion-v2 experiments"
    )
    parser.add_argument(
        "--output", "-o", 
        required=True,
        help="Output CSV file path"
    )
    parser.add_argument(
        "--model", 
        default="sy_simple",
        help="Model name (default: sy_simple)"
    )
    parser.add_argument(
        "--verify", 
        action="store_true",
        help="Verify files exist in S3 before adding to list"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary of generated task list"
    )
    
    args = parser.parse_args()
    
    logger.info("🚀 Generating CSV task list for parameter-distortion-v2")
    
    # Create task generator
    generator = ParameterDistortionTaskGenerator(model_name=args.model)
    
    # Initialize S3 manager if verification is requested
    s3_manager = None
    if args.verify:
        s3_manager = S3ConfigManager()
        logger.info("File verification enabled - checking S3 file existence")
    
    # Generate task list
    task_df = generator.generate_task_list(
        output_csv=args.output,
        verify_exists=args.verify,
        s3_manager=s3_manager
    )
    
    if args.summary:
        print_task_summary(task_df)
    
    logger.info(f"✅ CSV generation complete: {args.output}")


def main():
    """Main execution function"""
    # Check if CSV generation mode is requested
    if len(sys.argv) > 1 and sys.argv[1] in ['--output', '-o', '--generate-csv']:
        generate_csv_task_list()
        return
    
    logger.info("🚀 Starting Complete Parameter Distortion Data Generation")
    
    # Send start notification
    script_name = 'parameter-distortion-v2'
    notify_start(script_name)
    
    start_time = datetime.now()
    
    try:
        # Initialize S3 manager
        s3_manager = S3ConfigManager()
        
        # Configuration
        model_name = "sy_simple"
        distortion_factors = [0, 1.1, 1.3, 1.5, 2.0, 3.0]
        n_samples = 2000
        seed = 42
        
        # Load model objects
        model_spec, model_builder, model_tuner = load_model_objects(model_name, s3_manager)
        
        # Setup solver
        solver = RoadrunnerSolver()
        solver.compile(model_builder.get_sbml_model())
        
        # Get initial values (inactive state variables)
        state_variables = model_builder.get_state_variables()
        initial_values = {k: v for k, v in state_variables.items() if not k.endswith('a')}
        if 'O' in initial_values:
            del initial_values['O']
        
        # Simulation parameters (matching sy_simple-make-data-v1.py)
        simulation_params = {'start': 0, 'end': 10000, 'points': 101}
        
        # Generate base feature data (same for all distortion levels)
        logger.info("Generating base feature data...")
        feature_data = generate_feature_data(model_spec, initial_values, n_samples, seed)
        
        # Generate distorted parameter sets
        logger.info("Generating distorted parameter sets...")
        all_distorted_sets = generate_distorted_parameter_sets(
            model_builder, distortion_factors, n_samples, seed
        )
        
        # Process each distortion factor
        total_datasets = 0
        for distortion_factor in distortion_factors:
            logger.info(f"Processing distortion factor: {distortion_factor}")
            
            # Convert parameter list to DataFrame
            parameter_sets = all_distorted_sets[distortion_factor]
            parameter_df = pd.DataFrame(parameter_sets)
            parameter_df['distortion_factor'] = distortion_factor
            parameter_df['sample_id'] = range(n_samples)
            
            # Create clean parameter DataFrame for simulation (without metadata columns)
            clean_parameter_df = pd.DataFrame(parameter_sets)  # Only kinetic parameters
            
            # Generate complete dataset for this distortion level
            dataset_dict = generate_complete_dataset_for_distortion_level(
                feature_data=feature_data,
                parameter_df=clean_parameter_df,
                model_spec=model_spec,
                model_builder=model_builder,
                solver=solver,
                distortion_factor=distortion_factor,
                simulation_params=simulation_params
            )
            
            # Save complete dataset to S3
            save_complete_dataset(dataset_dict, distortion_factor, model_name, s3_manager)
            
            total_datasets += 1
            logger.info(f"✅ Completed distortion factor {distortion_factor}")
        
        # Calculate execution time
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Send success notification
        notify_success(script_name, duration, processed_count=len(distortion_factors))
        
        logger.info("✅ Complete parameter distortion data generation finished successfully")
        logger.info(f"Generated {len(distortion_factors)} distortion levels")
        logger.info(f"Total datasets created: {total_datasets}")
        logger.info(f"Total execution time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        
        # List files for verification
        folder_name = f"{model_name}_parameter_distortion_v2"
        item_list = s3_manager.list_files_from_path(f"{s3_manager.save_result_path}/data/{folder_name}/")
        
        logger.info("Files in S3:")
        for item in item_list:
            logger.info(f"  - {item}")
        
    except Exception as e:
        logger.error(f"❌ Parameter distortion data generation failed: {e}")
        # Send failure notification
        duration = (datetime.now() - start_time).total_seconds()
        notify_failure(script_name, e, duration_seconds=duration)
        raise


if __name__ == "__main__":
    # Check command line arguments to determine mode
    if len(sys.argv) > 1:
        main()
    else:
        # Default: run data generation with default parameters
        main()
