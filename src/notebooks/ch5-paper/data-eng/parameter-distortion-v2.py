"""
Complete Parameter Distortion Data Generation Script - Configuration Version

Generates comprehensive datasets with distorted parameter sets including:
- Distorted parameters
- Feature data
- Target data with make_target_data_with_params
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
DISTORTION_FACTORS = [0, 0.1, 0.2, 0.3, 0.5, 1.0]  # Multiplicative noise levels (fraction of parameter value for noise std)
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


class ParameterDistortionTaskGenerator(BaseTaskGenerator):
    """
    Task generator for parameter-distortion-v2 experiment pattern.
    
    This class encapsulates the pattern-specific logic for generating
    CSV task lists for parameter distortion experiments.
    """
    
    def __init__(self, model_name: str = "sy_simple"):
        super().__init__(model_name)
        self.experiment_type = "parameter-distortion-v2"
        self.distortion_factors = DISTORTION_FACTORS  # Use configuration
        
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
        # All distortion levels should use the BASELINE targets (ground truth)
        # from baseline virtual models for ML prediction tasks
        base_path = f"{self.get_base_folder()}"
        
        return [
            {
                "path": f"{base_path}/baseline_targets.pkl",
                "label": "baseline_targets"
            }
        ]


def apply_gaussian_distortion(original_params, distortion_factor, seed=42):
    """
    Apply multiplicative Gaussian noise to parameters with scale-proportional distortion.
    
    Args:
        original_params: Dictionary of original parameters from model_builder
        distortion_factor: Controls strength of distortion (0 = no distortion)
                         Interpreted as fraction of parameter value for noise std
        seed: Random seed for reproducible distortion
    
    Returns:
        Dictionary of distorted parameters
    """
    if distortion_factor == 0:
        return original_params  # No distortion for baseline
    
    rng = default_rng(seed)
    distorted_params = {}
    
    for key, value in original_params.items():
        # Apply multiplicative Gaussian noise with std = distortion_factor * value
        # This ensures noise scales proportionally with parameter magnitude
        noise_std = distortion_factor * abs(value)
        noise = rng.normal(loc=0, scale=noise_std)
        distorted_params[key] = max(value + noise, 1e-8)  # Ensure non-negative
    
    return distorted_params


def generate_distorted_parameter_sets(model_builder, distortion_factors, n_samples=N_SAMPLES, seed=SEED):
    """
    Generate multiple parameter sets with different distortion levels
    
    Args:
        model_builder: ModelBuilder instance
        distortion_factors: List of distortion factors to apply (from configuration)
        n_samples: Number of parameter sets per distortion factor (from configuration)
        seed: Random seed for reproducibility (from configuration)
    
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


def generate_feature_data(model_spec, initial_values, n_samples=N_SAMPLES, seed=SEED):
    """
    Generate feature data using lhs perturbation
    
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
    simulation_params,
    success_indices=None
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
        success_indices: Optional list of indices to use (if None, will run simulation)
    
    Returns:
        Dictionary containing all dataset components
    """
    logger.info(f"Generating complete dataset for distortion factor {distortion_factor}")
    
    if success_indices is not None:
        # Use pre-determined successful indices
        logger.info(f"Using {len(success_indices)} pre-selected successful samples")
        
        # Filter data to only successful indices
        feature_data_filtered = feature_data.iloc[success_indices].reset_index(drop=True)
        parameter_df_filtered = parameter_df.iloc[success_indices].reset_index(drop=True)
        
        # Generate target data for filtered samples
        target_data, timecourse_data, _ = make_target_data_with_params_robust(
            model_spec=model_spec,
            solver=solver,
            feature_df=feature_data_filtered,
            parameter_df=parameter_df_filtered,
            simulation_params=simulation_params,
            n_cores=1,
            outcome_var=OUTCOME_VAR,
            capture_all_species=True,
            verbose=False
        )
        
        # Reset indices
        target_data = target_data.reset_index(drop=True)
        if isinstance(timecourse_data, pd.DataFrame):
            timecourse_data = timecourse_data.reset_index(drop=True)
        
        # Create success mask for internal tracking
        success_mask = pd.Series([True] * len(success_indices))
        
    else:
        # Generate target and timecourse data using robust version (original behavior)
        logger.info("Generating target and timecourse data (robust)...")
        target_data, timecourse_data, success_mask = make_target_data_with_params_robust(
            model_spec=model_spec,
            solver=solver,
            feature_df=feature_data,
            parameter_df=parameter_df,
            simulation_params=simulation_params,
            n_cores=1,
            outcome_var=OUTCOME_VAR,
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
    
    # Clip extreme values in dynamic features to prevent float32 overflow in ML
    # Values exceeding ±1e9 can cause sklearn's float32 conversion to fail
    clipping_threshold = 1e9
    logger.info(f"Clipping dynamic features to ±{clipping_threshold:.1e} to prevent float32 overflow")
    
    def clip_dataframe(df, threshold):
        """Clip DataFrame values to prevent extreme values"""
        df_clipped = df.copy()
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Count extreme values before clipping
                extreme_pos = (df[col] > threshold).sum()
                extreme_neg = (df[col] < -threshold).sum()
                if extreme_pos > 0 or extreme_neg > 0:
                    logger.debug(f"  {col}: clipping {extreme_pos} > {threshold:.1e}, {extreme_neg} < -{threshold:.1e}")
                # Clip values
                df_clipped[col] = df_clipped[col].clip(lower=-threshold, upper=threshold)
        return df_clipped
    
    dynamic_features = clip_dataframe(dynamic_features, clipping_threshold)
    dynamic_features_no_outcome = clip_dataframe(dynamic_features_no_outcome, clipping_threshold)
    
    # Also consider clipping last_time_points (though they're usually smaller)
    last_time_points = clip_dataframe(last_time_points, clipping_threshold)
    last_time_points_no_outcome = clip_dataframe(last_time_points_no_outcome, clipping_threshold)
    
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
    if not UPLOAD_S3:
        logger.info(f"Skipping S3 upload for model {model_name}, distortion factor {distortion_factor}")
        return
    
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
    
    logger.info("🚀 Generating CSV task list for parameter-distortion-v2")
    
    # Process model configuration
    model_names = process_model_config(MODEL_NAME)
    
    all_task_rows = []
    
    # Initialize S3 manager if UPLOAD_S3 is True
    s3_manager = S3ConfigManager() if UPLOAD_S3 else None
    
    for model_name in model_names:
        logger.info(f"Generating tasks for model: {model_name}")
        
        # Create task generator
        generator = ParameterDistortionTaskGenerator(model_name=model_name)
        
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
            folder_name = generator.get_base_folder() if 'generator' in locals() else f"{model_names[0]}_parameter_distortion_v2"
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


def check_baseline_exists(model_name, s3_manager):
    """
    Check if shared baseline exists and validate it has enough samples.
    
    Args:
        model_name: Name of the model
        s3_manager: S3ConfigManager instance
    
    Returns:
        Tuple of (baseline_data, baseline_samples_count) if baseline exists and is valid
        Raises RuntimeError if baseline doesn't exist or doesn't have enough samples
    """
    try:
        from baseline_generator import load_baseline_from_s3
        baseline_data = load_baseline_from_s3(model_name, s3_manager)
        
        # Check if we have the essential components
        if (baseline_data.get('features') is None or 
            baseline_data.get('targets') is None):
            raise RuntimeError(f"Incomplete baseline found for {model_name}. Missing essential components.")
        
        baseline_samples = len(baseline_data['features'])
        logger.info(f"✅ Shared baseline found for {model_name} with {baseline_samples} samples")
        
        # Validate that baseline has enough samples for our N_SAMPLES configuration
        if baseline_samples < N_SAMPLES:
            raise RuntimeError(
                f"Baseline has only {baseline_samples} samples, but N_SAMPLES={N_SAMPLES}. "
                f"Either regenerate baseline with more samples or reduce N_SAMPLES."
            )
        
        return baseline_data, baseline_samples
        
    except Exception as e:
        if "Could not load" in str(e) or "No such file" in str(e):
            raise RuntimeError(
                f"❌ Shared baseline not found for model {model_name}. "
                f"Please run generate-shared-baseline.py first to create the baseline."
            )
        else:
            raise RuntimeError(f"❌ Error loading baseline for {model_name}: {e}")


def process_single_model(model_name, s3_manager):
    """
    Process a single model through the parameter distortion pipeline.
    
    Uses SHARED baseline virtual models loaded from S3 (generated by generate-shared-baseline.py)
    and applies additional Gaussian distortion to parameters only.
    
    Args:
        model_name: Name of the model to process
        s3_manager: S3ConfigManager instance
    
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"🚀 Starting parameter distortion generation for model: {model_name}")
        
        # Load model objects
        model_spec, model_builder, model_tuner = load_model_objects(model_name, s3_manager)
        
        # Setup solver
        solver = RoadrunnerSolver()
        solver.compile(model_builder.get_sbml_model())
        
        # Step 1: Load shared baseline virtual models (FAIL FAST if missing)
        logger.info("Loading shared baseline virtual models from S3...")
        baseline_data, baseline_samples = check_baseline_exists(model_name, s3_manager)
        
        # Use first N_SAMPLES from baseline (or all if N_SAMPLES > baseline_samples)
        # Note: check_baseline_exists already validated N_SAMPLES <= baseline_samples
        baseline_features = baseline_data['features'].iloc[:N_SAMPLES].reset_index(drop=True)
        baseline_targets = baseline_data['targets'].iloc[:N_SAMPLES].reset_index(drop=True)
        baseline_parameters = baseline_data['parameters']
        if baseline_parameters is not None:
            baseline_parameters = baseline_parameters.iloc[:N_SAMPLES].reset_index(drop=True)
        
        logger.info(f"✅ Using {N_SAMPLES} samples from shared baseline ({baseline_samples} total available)")
        logger.info(f"Baseline features shape: {baseline_features.shape}")
        logger.info(f"Baseline parameters shape: {baseline_parameters.shape if baseline_parameters is not None else 'N/A'}")
        logger.info(f"Baseline targets shape: {baseline_targets.shape}")
        
        # Step 2: Generate distorted parameter sets from baseline
        logger.info("Generating distorted parameter sets from baseline...")
        all_distorted_sets = {}
        
        for distortion_factor in DISTORTION_FACTORS:
            logger.info(f"Generating parameter sets with distortion factor {distortion_factor}")
            
            distorted_parameter_sets = []
            for i in range(len(baseline_parameters)):
                # Get baseline parameters for this virtual model
                baseline_params = baseline_parameters.iloc[i].to_dict()
                
                # Apply additional Gaussian distortion
                distorted_params = apply_gaussian_distortion(
                    baseline_params, distortion_factor, SEED + i
                )
                distorted_parameter_sets.append(distorted_params)
            
            all_distorted_sets[distortion_factor] = distorted_parameter_sets
        
        # Step 3: Identify samples that succeed at ALL distortion levels
        logger.info("Identifying samples that succeed at ALL distortion levels...")
        all_success_masks = {}
        
        for distortion_factor in DISTORTION_FACTORS:
            logger.info(f"Testing samples at distortion factor {distortion_factor}...")
            parameter_sets = all_distorted_sets[distortion_factor]
            clean_parameter_df = pd.DataFrame(parameter_sets)
            
            # Run simulation to get success mask
            target_data, timecourse_data, success_mask = make_target_data_with_params_robust(
                model_spec=model_spec,
                solver=solver,
                feature_df=baseline_features,
                parameter_df=clean_parameter_df,
                simulation_params=SIMULATION_PARAMS,
                n_cores=1,
                outcome_var=OUTCOME_VAR,
                capture_all_species=True,
                verbose=False
            )
            
            all_success_masks[distortion_factor] = success_mask
        
        # Find intersection of successful samples across ALL distortion levels
        success_intersection = pd.Series([True] * len(baseline_features))
        for distortion_factor, success_mask in all_success_masks.items():
            success_intersection = success_intersection & success_mask
        
        success_indices = success_intersection[success_intersection].index.tolist()
        logger.info(f"Found {len(success_indices)} samples that succeed at ALL distortion levels out of {len(baseline_features)} total")
        
        if len(success_indices) == 0:
            logger.error("❌ No samples succeeded at all distortion levels. Cannot proceed.")
            return False
        
        # Step 4: Save baseline targets (ground truth) for ML tasks
        if UPLOAD_S3 and s3_manager:
            # Filter baseline targets to only successful samples
            baseline_targets_filtered = baseline_targets.loc[success_indices].reset_index(drop=True)
            
            gen_path = s3_manager.save_result_path
            folder_name = f"{model_name}_parameter_distortion_v2"
            baseline_targets_path = f"{gen_path}/data/{folder_name}/baseline_targets.pkl"
            s3_manager.save_data_from_path(baseline_targets_path, baseline_targets_filtered, data_format="pkl")
            logger.info(f"✅ Saved baseline targets (ground truth) to S3: {baseline_targets_path}")
        
        # Step 5: Process each distortion factor
        total_datasets = 0
        for distortion_factor in DISTORTION_FACTORS:
            logger.info(f"Processing distortion factor: {distortion_factor}")
            
            # Convert parameter list to DataFrame
            parameter_sets = all_distorted_sets[distortion_factor]
            parameter_df = pd.DataFrame(parameter_sets)
            parameter_df['distortion_factor'] = distortion_factor
            parameter_df['sample_id'] = range(len(baseline_features))
            
            # Create clean parameter DataFrame for simulation (without metadata columns)
            clean_parameter_df = pd.DataFrame(parameter_sets)
            
            # Generate dataset for this distortion level using successful indices
            dataset_dict = generate_complete_dataset_for_distortion_level(
                feature_data=baseline_features,
                parameter_df=clean_parameter_df,
                model_spec=model_spec,
                model_builder=model_builder,
                solver=solver,
                distortion_factor=distortion_factor,
                simulation_params=SIMULATION_PARAMS,
                success_indices=success_indices
            )
            
            # Save complete dataset to S3
            save_complete_dataset(dataset_dict, distortion_factor, model_name, s3_manager)
            
            total_datasets += 1
            logger.info(f"✅ Completed distortion factor {distortion_factor}")
        
        # Save success indices and success rates for reference
        if UPLOAD_S3 and s3_manager:
            gen_path = s3_manager.save_result_path
            folder_name = f"{model_name}_parameter_distortion_v2"
            
            # Save success indices
            success_path = f"{gen_path}/data/{folder_name}/success_indices.pkl"
            s3_manager.save_data_from_path(success_path, success_indices, data_format="pkl")
            
            # Save success rates per distortion level
            success_rates = {}
            for distortion_factor, success_mask in all_success_masks.items():
                success_rates[distortion_factor] = success_mask.sum() / len(success_mask)
            
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
        script_name = 'parameter-distortion-config'
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
            notify_failure('parameter-distortion-config', error_msg, duration_seconds=duration)
        else:
            notify_success('parameter-distortion-config', duration, processed_count=len(successful_models))
    
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
