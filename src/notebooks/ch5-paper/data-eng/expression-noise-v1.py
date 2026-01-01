"""
Complete Expression Noise Data Generation Script - Configuration Version

Generates comprehensive datasets with Gaussian noise applied to feature data including:
- Noisy feature data at different noise levels
- Target data generated with make_target_data_with_params using noisy features
- Timecourse data
- Dynamic features
- Last time point data

Follows the complete S3 storage pattern of sy_simple-make-data-v1.py.

CONFIGURATION-BASED VERSION:
For remote batch job execution where modifying script variables is more practical than CLI arguments.
Supports single model (string) or multiple models (list) for multiplexing.
"""

import sys
import os
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


# ===== CONFIGURATION SECTION =====
# MODIFY THESE VARIABLES FOR YOUR BATCH JOB
MODEL_NAME = "sy_simple"  # Can be string: "sy_simple" or list: ["sy_simple", "model_v2"]
NOISE_LEVELS = [0, 0.1, 0.2, 0.3, 0.5, 1.0]
N_SAMPLES = 2000
SEED = 42
SIMULATION_PARAMS = {'start': 0, 'end': 10000, 'points': 101}
OUTCOME_VAR = "Oa"
UPLOAD_S3 = True
SEND_NOTIFICATIONS = True
GENERATE_ML_TASK_LIST = True
# ===== END CONFIGURATION =====


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


class ExpressionNoiseTaskGenerator(BaseTaskGenerator):
    """
    Task generator for expression-noise-v1 experiment pattern.
    
    This class encapsulates the pattern-specific logic for generating
    CSV task lists for expression noise experiments.
    """
    
    def __init__(self, model_name: str = "sy_simple"):
        super().__init__(model_name)
        self.experiment_type = "expression-noise-v1"
        self.noise_levels = NOISE_LEVELS  # Use configuration
        
    def get_levels(self):
        return self.noise_levels
        
    def get_base_folder(self):
        return f"{self.model_name}_expression_noise_v1"
        
    def get_feature_files(self, noise_level):
        """Get feature files for a given noise level"""
        base_path = f"{self.get_base_folder()}/noise_{noise_level}"
        
        feature_files = [
            {
                "path": f"{base_path}/noisy_features.pkl",
                "label": f"noisy_features_{noise_level}"
            },
            {
                "path": f"{base_path}/dynamic_features.pkl", 
                "label": f"dynamic_features_{noise_level}"
            },
            {
                "path": f"{base_path}/dynamic_features_no_outcome.pkl",
                "label": f"dynamic_features_no_outcome_{noise_level}"
            },
            {
                "path": f"{base_path}/last_time_points.pkl",
                "label": f"last_time_points_{noise_level}"
            },
            {
                "path": f"{base_path}/last_time_points_no_outcome.pkl",
                "label": f"last_time_points_no_outcome_{noise_level}"
            }
        ]
        
        # Also include original features for comparison at noise_level=0
        if noise_level == 0:
            feature_files.append({
                "path": f"{base_path}/original_features.pkl",
                "label": "original_features_0"
            })
            
        return feature_files
        
    def get_target_files(self, noise_level):
        """Get target files for a given noise level"""
        # All noise levels should use the BASELINE targets (ground truth)
        # from baseline virtual models for ML prediction tasks
        base_path = f"{self.get_base_folder()}"
        
        return [
            {
                "path": f"{base_path}/baseline_targets.pkl",
                "label": "baseline_targets"
            }
        ]


def apply_expression_noise(feature_data, noise_level, seed):
    """
    Apply Gaussian noise to expression feature data

    Args:
        feature_data: Original feature data DataFrame
        noise_level: Standard deviation of Gaussian noise as fraction of each value
        seed: Random seed for reproducibility

    Returns:
        Feature data with applied noise
    """
    # Handle baseline case (noise_level == 0) - return original data unchanged
    if noise_level == 0:
        return feature_data.copy()  # No noise for baseline

    rng = default_rng(seed)

    # Apply noise to each column independently
    noisy_feature_data = feature_data.copy()
    for column in feature_data.columns:
        original_values = feature_data[column].values

        # Relative noise: noise std = noise_level × absolute value
        # Add small epsilon to handle zeros
        epsilon = 1e-6
        noise_std = np.abs(original_values) * noise_level + epsilon

        # Generate noise for each point with its own std
        noise = rng.normal(0, noise_std)

        # Apply noise and ensure no negative values
        noisy_values = original_values + noise
        noisy_feature_data[column] = np.maximum(noisy_values, 0)

    return noisy_feature_data


def generate_base_feature_data(model_spec, initial_values, n_samples=N_SAMPLES, seed=SEED):
    """
    Generate base feature data using lhs perturbation
    
    Args:
        model_spec: ModelSpecification instance
        initial_values: Dictionary of initial values (inactive state variables)
        n_samples: Number of samples (from configuration)
        seed: Random seed (from configuration)
    
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
    
    logger.info(f"Generated base feature data with shape: {feature_data.shape}")
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


def generate_complete_dataset_for_noise_level(
    base_feature_data,
    noisy_feature_data,
    parameter_df,
    model_spec,
    model_builder,
    solver,
    noise_level,
    simulation_params
):
    """
    Generate complete dataset for a specific noise level
    
    Args:
        base_feature_data: Original (clean) feature data DataFrame
        noisy_feature_data: Feature data with applied noise
        parameter_df: DataFrame of parameters (original, not distorted)
        model_spec: ModelSpecification instance
        solver: Solver instance
        noise_level: Current noise level
        simulation_params: Simulation parameters
    
    Returns:
        Dictionary containing all dataset components
    """
    logger.info(f"Generating complete dataset for noise level {noise_level}")
    
    # Generate target and timecourse data with noisy features using robust version
    logger.info("Generating target and timecourse data with noisy features (robust)...")
    target_data, timecourse_data, success_mask = make_target_data_with_params_robust(
        model_spec=model_spec,
        solver=solver,
        feature_df=noisy_feature_data,
        parameter_df=parameter_df,
        simulation_params=simulation_params,
        n_cores=1,
        outcome_var=OUTCOME_VAR,
        capture_all_species=True,
        verbose=False
    )
    
    # Filter all datasets to keep only successful samples
    noisy_feature_data = noisy_feature_data[success_mask].reset_index(drop=True)
    base_feature_data = base_feature_data[success_mask].reset_index(drop=True)
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
    
    # Also include original features for comparison (using robust version)
    original_target_data, original_timecourse_data, original_success_mask = make_target_data_with_params_robust(
        model_spec=model_spec,
        solver=solver,
        feature_df=base_feature_data,
        parameter_df=parameter_df,
        simulation_params=simulation_params,
        n_cores=1,
        outcome_var=OUTCOME_VAR,
        capture_all_species=True,
        verbose=False
    )
    
    # Note: base_feature_data and parameter_df are already filtered by success_mask from noisy simulation
    # Both should have the same success_mask since they use the same parameter_df
    
    return {
        'original_features': base_feature_data,
        'noisy_features': noisy_feature_data,
        'targets': target_data,
        'original_targets': original_target_data,
        'parameters': parameter_df,
        'timecourses': timecourse_data,
        'original_timecourses': original_timecourse_data,
        'dynamic_features': dynamic_features,
        'last_time_points': last_time_points,
        'dynamic_features_no_outcome': dynamic_features_no_outcome,
        'last_time_points_no_outcome': last_time_points_no_outcome
    }


def save_complete_dataset(dataset_dict, noise_level, model_name, s3_manager):
    """
    Save complete dataset to S3 following sy_simple-make-data-v1.py pattern
    
    Args:
        dataset_dict: Dictionary containing all dataset components
        noise_level: Noise level for this dataset
        model_name: Name of the model
        s3_manager: S3ConfigManager instance
    """
    if not UPLOAD_S3:
        logger.info(f"Skipping S3 upload for model {model_name}, noise level {noise_level}")
        return
    
    gen_path = s3_manager.save_result_path
    folder_name = f"{model_name}_expression_noise_v1"
    subfolder_name = f"noise_{noise_level}"
    full_path = f"{gen_path}/data/{folder_name}/{subfolder_name}"
    
    # Save each component
    file_mappings = {
        'original_features': 'original_features.pkl',
        'noisy_features': 'noisy_features.pkl',
        'targets': 'targets.pkl',
        'original_targets': 'original_targets.pkl',
        'parameters': 'parameter_sets.pkl',
        'timecourses': 'timecourses.pkl',
        'original_timecourses': 'original_timecourses.pkl',
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
    
    # Save metadata for this noise level
    metadata = {
        'noise_level': noise_level,
        'model_name': model_name,
        'n_samples': dataset_dict['noisy_features'].shape[0],
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
    Generate CSV task list for expression noise experiments.
    
    This function provides a configuration-based interface for generating
    CSV task lists without running the full data generation pipeline.
    """
    if not GENERATE_ML_TASK_LIST:
        logger.info("Skipping ML task list generation (GENERATE_ML_TASK_LIST=False)")
        return pd.DataFrame()
    
    # Check if S3 upload is enabled
    if not UPLOAD_S3:
        logger.info("Skipping ML task list generation (UPLOAD_S3=False)")
        return pd.DataFrame()
    
    logger.info("🚀 Generating CSV task list for expression-noise-v1")
    
    # Process model configuration
    model_names = process_model_config(MODEL_NAME)
    
    all_task_rows = []
    
    # Initialize S3 manager if UPLOAD_S3 is True
    s3_manager = S3ConfigManager() if UPLOAD_S3 else None
    
    for model_name in model_names:
        logger.info(f"Generating tasks for model: {model_name}")
        
        # Create task generator
        generator = ExpressionNoiseTaskGenerator(model_name=model_name)
        
        # Generate task list for this model
        task_rows = []
        for level in generator.get_levels():
            feature_files = generator.get_feature_files(level)
            target_files = generator.get_target_files(level)
            
            # Create all combinations of feature and target files
            for feature in feature_files:
                for target in target_files:
                    task_rows.append({
                        "feature_data": feature["path"],
                        "feature_data_label": feature["label"],
                        "target_data": target["path"],
                        "target_data_label": target["label"],
                        "experiment_type": generator.experiment_type,
                        "level": level,
                        "model_name": model_name
                    })
        
        all_task_rows.extend(task_rows)
        logger.info(f"Generated {len(task_rows)} tasks for model {model_name}")
    
    # Save combined task list
    if all_task_rows:
        task_df = pd.DataFrame(all_task_rows)
        
        # Upload to S3 if enabled
        if UPLOAD_S3 and s3_manager:
            output_csv = "task_list.csv"
            folder_name = generator.get_base_folder() if 'generator' in locals() else f"{model_names[0]}_expression_noise_v1"
            gen_path = s3_manager.save_result_path
            s3_csv_path = f"{gen_path}/data/{folder_name}/{output_csv}"
            
            s3_manager.save_data_from_path(s3_csv_path, task_df, data_format="csv")
            logger.info(f"✅ Uploaded CSV to S3: {s3_csv_path}")
            
            if len(model_names) > 1:
                print_task_summary(task_df)
            
            return task_df
        else:
            logger.info("Skipping CSV save (UPLOAD_S3=False or no S3 manager)")
            return pd.DataFrame()
    else:
        logger.warning("No tasks generated")
        return pd.DataFrame()


def process_single_model(model_name, s3_manager):
    """
    Process a single model through the expression noise pipeline.
    
    Uses baseline virtual models generated with lognormal perturbation (like sy_simple-make-data-v1.py)
    and applies additional Gaussian noise to features only.
    
    Args:
        model_name: Name of the model to process
        s3_manager: S3ConfigManager instance
    
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"🚀 Starting expression noise generation for model: {model_name}")
        
        # Load model objects
        model_spec, model_builder, model_tuner = load_model_objects(model_name, s3_manager)
        
        # Setup solver
        solver = RoadrunnerSolver()
        solver.compile(model_builder.get_sbml_model())
        
        # Step 1: Generate or load baseline virtual models
        logger.info("Generating baseline virtual models with lognormal perturbation...")
        from baseline_generator import generate_baseline_virtual_models
        
        baseline_data = generate_baseline_virtual_models(
            model_spec=model_spec,
            model_builder=model_builder,
            solver=solver,
            n_samples=N_SAMPLES,
            seed=SEED,
            simulation_params=SIMULATION_PARAMS
        )
        
        baseline_features = baseline_data['features']
        baseline_targets = baseline_data['targets']
        baseline_parameters = baseline_data['parameters']
        
        logger.info(f"Baseline features shape: {baseline_features.shape}")
        logger.info(f"Baseline parameters shape: {baseline_parameters.shape}")
        logger.info(f"Baseline targets shape: {baseline_targets.shape}")
        
        # Save baseline for reference
        if UPLOAD_S3 and s3_manager:
            from baseline_generator import save_baseline_to_s3
            baseline_paths = save_baseline_to_s3(
                baseline_data, model_name, s3_manager, upload=UPLOAD_S3
            )
            logger.info(f"✅ Saved baseline virtual models to S3")
        
        # Step 2: Identify samples that succeed at ALL noise levels
        logger.info("Identifying samples that succeed at ALL noise levels...")
        all_success_masks = {}
        
        for noise_level in NOISE_LEVELS:
            logger.info(f"Testing samples at noise level {noise_level}...")
            
            # Apply noise to baseline features
            noisy_features = apply_expression_noise(
                baseline_features, noise_level, SEED
            )
            
            # Run simulation to get success mask
            target_data, timecourse_data, success_mask = make_target_data_with_params_robust(
                model_spec=model_spec,
                solver=solver,
                feature_df=noisy_features,
                parameter_df=baseline_parameters,
                simulation_params=SIMULATION_PARAMS,
                n_cores=1,
                outcome_var=OUTCOME_VAR,
                capture_all_species=True,
                verbose=False
            )
            
            all_success_masks[noise_level] = success_mask
        
        # Find intersection of successful samples across ALL noise levels
        success_intersection = pd.Series([True] * len(baseline_features))
        for noise_level, success_mask in all_success_masks.items():
            success_intersection = success_intersection & success_mask
        
        success_indices = success_intersection[success_intersection].index.tolist()
        logger.info(f"Found {len(success_indices)} samples that succeed at ALL noise levels out of {len(baseline_features)} total")
        
        if len(success_indices) == 0:
            logger.error("❌ No samples succeeded at all noise levels. Cannot proceed.")
            return False
        
        # Step 3: Save baseline targets (ground truth) for ML tasks
        if UPLOAD_S3 and s3_manager:
            # Filter baseline targets to only successful samples
            baseline_targets_filtered = baseline_targets.loc[success_indices].reset_index(drop=True)
            
            gen_path = s3_manager.save_result_path
            folder_name = f"{model_name}_expression_noise_v1"
            baseline_targets_path = f"{gen_path}/data/{folder_name}/baseline_targets.pkl"
            s3_manager.save_data_from_path(baseline_targets_path, baseline_targets_filtered, data_format="pkl")
            logger.info(f"✅ Saved baseline targets (ground truth) to S3: {baseline_targets_path}")
        
        # Step 4: Process each noise level
        total_datasets = 0
        for noise_level in NOISE_LEVELS:
            logger.info(f"Processing noise level: {noise_level}")
            
            # Apply expression noise to feature data (filtered to successful samples)
            logger.info(f"Applying Gaussian noise (level={noise_level}) to feature data...")
            baseline_features_filtered = baseline_features.iloc[success_indices].reset_index(drop=True)
            baseline_parameters_filtered = baseline_parameters.iloc[success_indices].reset_index(drop=True)
            
            noisy_feature_data = apply_expression_noise(
                baseline_features_filtered, noise_level, SEED
            )
            
            # Generate complete dataset for this noise level
            dataset_dict = generate_complete_dataset_for_noise_level(
                base_feature_data=baseline_features_filtered,
                noisy_feature_data=noisy_feature_data,
                parameter_df=baseline_parameters_filtered,
                model_spec=model_spec,
                model_builder=model_builder,
                solver=solver,
                noise_level=noise_level,
                simulation_params=SIMULATION_PARAMS
            )
            
            # Save complete dataset to S3
            save_complete_dataset(dataset_dict, noise_level, model_name, s3_manager)
            
            total_datasets += 1
            logger.info(f"✅ Completed noise level {noise_level}")
        
        # Save success indices and success rates for reference
        if UPLOAD_S3 and s3_manager:
            gen_path = s3_manager.save_result_path
            folder_name = f"{model_name}_expression_noise_v1"
            
            # Save success indices
            success_path = f"{gen_path}/data/{folder_name}/success_indices.pkl"
            s3_manager.save_data_from_path(success_path, success_indices, data_format="pkl")
            
            # Save success rates per noise level
            success_rates = {}
            for noise_level, success_mask in all_success_masks.items():
                success_rates[noise_level] = success_mask.sum() / len(success_mask)
            
            rates_path = f"{gen_path}/data/{folder_name}/success_rates.pkl"
            s3_manager.save_data_from_path(rates_path, success_rates, data_format="pkl")
            
            logger.info(f"✅ Saved success data to S3: {success_path}, {rates_path}")
            logger.info(f"Success rates: {success_rates}")
        
        logger.info(f"✅ Successfully processed model {model_name}: {total_datasets} datasets created")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to process model {model_name}: {e}")
        return False


def main():
    """Main execution function - configuration-based version"""
    
    # Send start notification if enabled
    if SEND_NOTIFICATIONS:
        script_name = 'expression-noise-config'
        notify_start(script_name)
    
    start_time = datetime.now()
    
    # Process model configuration
    model_names = process_model_config(MODEL_NAME)
    logger.info(f"Processing {len(model_names)} model(s): {model_names}")
    
    # Initialize S3 manager
    s3_manager = S3ConfigManager() if UPLOAD_S3 else None
    
    # Process each model
    successful_models = []
    failed_models = []
    
    for i, model_name in enumerate(model_names, 1):
        logger.info(f"Processing model {i}/{len(model_names)}: {model_name}")
        
        success = process_single_model(model_name, s3_manager)
        
        if success:
            successful_models.append(model_name)
        else:
            failed_models.append(model_name)
    
    # Generate CSV task list if enabled
    if GENERATE_ML_TASK_LIST:
        logger.info("Generating ML task list CSV...")
        task_df = generate_csv_task_list()
        if not task_df.empty:
            logger.info(f"✅ Generated task list with {len(task_df)} rows")
    
    # Calculate execution time
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Send notifications if enabled
    if SEND_NOTIFICATIONS:
        if failed_models:
            error_msg = f"Failed models: {failed_models}"
            notify_failure('expression-noise-config', error_msg, duration_seconds=duration)
        else:
            notify_success('expression-noise-config', duration, processed_count=len(successful_models))
    
    # Summary
    logger.info("=" * 60)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total models: {len(model_names)}")
    logger.info(f"Successful: {len(successful_models)} - {successful_models}")
    logger.info(f"Failed: {len(failed_models)} - {failed_models}")
    logger.info(f"Total execution time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    logger.info("=" * 60)
    
    if failed_models:
        raise RuntimeError(f"Processing failed for models: {failed_models}")


if __name__ == "__main__":
    main()
