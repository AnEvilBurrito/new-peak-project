"""
Complete Response Noise Data Generation Script - Configuration Version

Generates comprehensive datasets with Gaussian noise applied to target data including:
- Clean feature data generated with make_data_extended
- Target data with applied Gaussian noise at different levels
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
from models.utils.data_generation_helpers import make_data_extended, make_target_data_with_params
from models.utils.dynamic_calculations import dynamic_features_method, last_time_point_method
from models.Solver.RoadrunnerSolver import RoadrunnerSolver
from scripts.ntfy_notifier import notify_start, notify_success, notify_failure
from numpy.random import default_rng
from tqdm import tqdm

# Import shared utilities for CSV generation
from ml_task_utils import BaseTaskGenerator, save_task_csv, print_task_summary

# Import shared utilities for combined feature generation
from combined_feature_utils import create_combined_feature_sets

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


class ResponseNoiseTaskGenerator(BaseTaskGenerator):
    """
    Task generator for response-noise-v1 experiment pattern.
    
    This class encapsulates the pattern-specific logic for generating
    CSV task lists for response noise experiments.
    """
    
    def __init__(self, model_name: str = "sy_simple"):
        super().__init__(model_name)
        self.experiment_type = "response-noise-v1"
        self.noise_levels = NOISE_LEVELS  # Use configuration
        
    def get_levels(self):
        return self.noise_levels
        
    def get_base_folder(self):
        return f"{self.model_name}_response_noise_v1"
        
    def get_feature_files(self, noise_level):
        """Get feature files for a given noise level"""
        base_path = f"{self.get_base_folder()}/noise_{noise_level}"
        
        return [
            {
                "path": f"{base_path}/features.pkl",
                "label": f"features_{noise_level}"
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
            },
            {
                "path": f"{base_path}/original_plus_dynamic_no_outcome.pkl",
                "label": f"original_plus_dynamic_no_outcome_{noise_level}"
            },
            {
                "path": f"{base_path}/original_plus_last_no_outcome.pkl",
                "label": f"original_plus_last_no_outcome_{noise_level}"
            },
            {
                "path": f"{base_path}/original_plus_dynamic_with_outcome.pkl",
                "label": f"original_plus_dynamic_with_outcome_{noise_level}"
            },
            {
                "path": f"{base_path}/original_plus_last_with_outcome.pkl",
                "label": f"original_plus_last_with_outcome_{noise_level}"
            }
        ]
        
    def get_target_files(self, noise_level):
        """Get target files for a given noise level"""
        base_path = f"{self.get_base_folder()}/noise_{noise_level}"
        
        # For response noise experiments, ML tasks should predict NOISY targets
        # (response measurement noise), not clean targets
        return [
            {
                "path": f"{base_path}/noisy_targets.pkl",
                "label": f"noisy_targets_{noise_level}"
            }
        ]


def apply_response_noise(target_data, noise_level, seed):
    """
    Apply multiplicative Gaussian noise: x' = x × (1 + ε), ε ~ N(0, noise_level)

    Args:
        target_data: Original target data DataFrame
        noise_level: Standard deviation of relative error (e.g., 0.1 = 10%)
        seed: Random seed for reproducibility

    Returns:
        Noisy target data (clipped at zero to ensure non‑negative values)
    """
    if noise_level == 0:
        return target_data.copy()
    rng = default_rng(seed)

    noisy_target_data = target_data.copy()
    for column in target_data.columns:
        original_values = target_data[column].values
        
        # Generate relative noise: ε ~ N(0, noise_level)
        relative_noise = rng.normal(0, noise_level, len(original_values))
        
        # Apply multiplicative noise: x' = x × (1 + ε)
        noisy_values = original_values * (1 + relative_noise)
        
        # Ensure non‑negative values (concentrations cannot be negative)
        noisy_target_data[column] = np.maximum(noisy_values, 0)
    
    return noisy_target_data


def generate_complete_dataset_for_noise_level(
    clean_dataset,
    noisy_target_data,
    noise_level,
    baseline_dynamics
):
    """
    Generate complete dataset for a specific noise level using pre-computed baseline dynamics.
    
    Args:
        clean_dataset: Dictionary containing clean dataset components from make_data_extended
        noisy_target_data: Target data with applied noise
        noise_level: Current noise level
        baseline_dynamics: Dictionary containing pre-computed baseline dynamics:
            - dynamic_features_with_outcome
            - last_time_points_with_outcome
            - dynamic_features_no_outcome
            - last_time_points_no_outcome
    
    Returns:
        Dictionary containing all dataset components
    """
    logger.info(f"Generating complete dataset for noise level {noise_level} using baseline dynamics")
    
    
    # Use pre-computed dynamic features from baseline dynamics
    logger.info("Using pre-computed dynamic features from baseline dynamics...")
    
    dynamic_features = baseline_dynamics['dynamic_features_with_outcome']
    last_time_points = baseline_dynamics['last_time_points_with_outcome']
    dynamic_features_no_outcome = baseline_dynamics['dynamic_features_no_outcome']
    last_time_points_no_outcome = baseline_dynamics['last_time_points_no_outcome']
    
    logger.info(f"Dynamic features shape: {dynamic_features.shape}")
    logger.info(f"Last time points shape: {last_time_points.shape}")
    logger.info(f"Dynamic features (no outcome) shape: {dynamic_features_no_outcome.shape}")
    logger.info(f"Last time points (no outcome) shape: {last_time_points_no_outcome.shape}")
    
    # Create combined feature sets
    logger.info("Creating combined feature sets...")
    combined_features = create_combined_feature_sets(
        original_features=clean_dataset['features'],
        dynamic_features_with_outcome=dynamic_features,
        last_time_points_with_outcome=last_time_points,
        dynamic_features_no_outcome=dynamic_features_no_outcome,
        last_time_points_no_outcome=last_time_points_no_outcome
    )
    
    return {
        'features': clean_dataset['features'],
        'clean_targets': clean_dataset['targets'],
        'noisy_targets': noisy_target_data,
        'parameters': clean_dataset['parameters'],
        'dynamic_features': dynamic_features,
        'last_time_points': last_time_points,
        'dynamic_features_no_outcome': dynamic_features_no_outcome,
        'last_time_points_no_outcome': last_time_points_no_outcome,
        'combined_features': combined_features
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
    folder_name = f"{model_name}_response_noise_v1"
    subfolder_name = f"noise_{noise_level}"
    full_path = f"{gen_path}/data/{folder_name}/{subfolder_name}"
    
    # Save each component
    file_mappings = {
        'features': 'features.pkl',
        'clean_targets': 'clean_targets.pkl',
        'noisy_targets': 'noisy_targets.pkl',
        'parameters': 'parameter_sets.pkl',
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
    
    # Save combined feature sets
    if 'combined_features' in dataset_dict and dataset_dict['combined_features']:
        for combined_name, combined_df in dataset_dict['combined_features'].items():
            s3_path = f"{full_path}/{combined_name}.pkl"
            s3_manager.save_data_from_path(s3_path, combined_df, data_format="pkl")
            logger.info(f"✅ Saved combined feature set to S3: {s3_path}")
    
    # Save metadata for this noise level
    metadata = {
        'noise_level': noise_level,
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


def load_baseline_dynamics(model_name, s3_manager):
    """
    Load baseline dynamics data from S3 (with and without outcome versions).
    
    Baseline dynamics data is already aligned with successful samples from baseline generation,
    so no additional filtering is needed.
    
    Args:
        model_name: Name of the model
        s3_manager: S3ConfigManager instance
    
    Returns:
        Dictionary containing baseline dynamics components:
        - dynamic_features_with_outcome
        - last_time_points_with_outcome
        - dynamic_features_no_outcome
        - last_time_points_no_outcome
    """
    gen_path = s3_manager.save_result_path
    folder_name = f"{model_name}_baseline_dynamics_v1"
    full_path = f"{gen_path}/data/{folder_name}"
    
    logger.info(f"Loading baseline dynamics for model: {model_name}")
    
    try:
        # Load all four baseline dynamics files
        dynamic_features_with_outcome = s3_manager.load_data_from_path(
            f"{full_path}/dynamic_features_with_outcome.pkl", 
            data_format='pkl'
        )
        
        last_time_points_with_outcome = s3_manager.load_data_from_path(
            f"{full_path}/last_time_points_with_outcome.pkl", 
            data_format='pkl'
        )
        
        dynamic_features_no_outcome = s3_manager.load_data_from_path(
            f"{full_path}/dynamic_features_no_outcome.pkl", 
            data_format='pkl'
        )
        
        last_time_points_no_outcome = s3_manager.load_data_from_path(
            f"{full_path}/last_time_points_no_outcome.pkl", 
            data_format='pkl'
        )
        
        logger.info(f"✅ Loaded baseline dynamics for {model_name}")
        logger.info(f"  dynamic_features_with_outcome shape: {dynamic_features_with_outcome.shape}")
        logger.info(f"  last_time_points_with_outcome shape: {last_time_points_with_outcome.shape}")
        logger.info(f"  dynamic_features_no_outcome shape: {dynamic_features_no_outcome.shape}")
        logger.info(f"  last_time_points_no_outcome shape: {last_time_points_no_outcome.shape}")
        
        return {
            'dynamic_features_with_outcome': dynamic_features_with_outcome,
            'last_time_points_with_outcome': last_time_points_with_outcome,
            'dynamic_features_no_outcome': dynamic_features_no_outcome,
            'last_time_points_no_outcome': last_time_points_no_outcome
        }
        
    except Exception as e:
        if "Could not load" in str(e) or "No such file" in str(e):
            raise RuntimeError(
                f"❌ Baseline dynamics not found for model {model_name}. "
                f"Please run generate-baseline-dynamics-v1.py first to create baseline dynamics."
            )
        else:
            raise RuntimeError(f"❌ Error loading baseline dynamics for {model_name}: {e}")


def generate_csv_task_list():
    """
    Generate CSV task list for response noise experiments.
    
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
    
    logger.info("🚀 Generating CSV task list for response-noise-v1")
    
    # Process model configuration
    model_names = process_model_config(MODEL_NAME)
    
    all_task_rows = []
    
    # Initialize S3 manager if UPLOAD_S3 is True
    s3_manager_instance = S3ConfigManager() if UPLOAD_S3 else None
    
    for model_name in model_names:
        logger.info(f"Generating tasks for model: {model_name}")
        
        # Create task generator
        generator = ResponseNoiseTaskGenerator(model_name=model_name)
        
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
        if UPLOAD_S3 and s3_manager_instance:
            output_csv = "task_list.csv"
            folder_name = generator.get_base_folder() if 'generator' in locals() else f"{model_names[0]}_response_noise_v1"
            gen_path = s3_manager_instance.save_result_path
            s3_csv_path = f"{gen_path}/data/{folder_name}/{output_csv}"
            
            s3_manager_instance.save_data_from_path(s3_csv_path, task_df, data_format="csv")
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
    Process a single model through the response noise pipeline.
    
    Uses SHARED baseline virtual models loaded from S3 (generated by generate-shared-baseline.py)
    and applies Gaussian noise to target data only (response measurement noise).
    
    Args:
        model_name: Name of the model to process
        s3_manager: S3ConfigManager instance
    
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"🚀 Starting response noise generation for model: {model_name}")
        
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
        baseline_features = baseline_data['features'].iloc[:N_SAMPLES]
        baseline_targets = baseline_data['targets'].iloc[:N_SAMPLES]
        baseline_parameters = baseline_data['parameters']
        if baseline_parameters is not None:
            baseline_parameters = baseline_parameters.iloc[:N_SAMPLES]

        
        logger.info(f"✅ Using {N_SAMPLES} samples from shared baseline ({baseline_samples} total available)")
        logger.info(f"Baseline features shape: {baseline_features.shape}")
        logger.info(f"Baseline parameters shape: {baseline_parameters.shape if baseline_parameters is not None else 'N/A'}")
        logger.info(f"Baseline targets shape: {baseline_targets.shape}")
        
        # Step 2: Baseline data is already guaranteed to contain only successful samples
        # (generated by generate-shared-baseline.py with failed samples removed)
        # So we can use all samples directly without re-running simulations
        logger.info(f"Using all {len(baseline_features)} baseline samples (already filtered during baseline generation)...")
        
        # Step 3: Save baseline targets (ground truth) for reference
        if UPLOAD_S3 and s3_manager:
            # All baseline samples are successful (failed samples were removed during baseline generation)
            gen_path = s3_manager.save_result_path
            folder_name = f"{model_name}_response_noise_v1"
            baseline_targets_path = f"{gen_path}/data/{folder_name}/baseline_targets.pkl"
            s3_manager.save_data_from_path(baseline_targets_path, baseline_targets, data_format="pkl")
            logger.info(f"✅ Saved baseline targets (ground truth) to S3: {baseline_targets_path}")
        
        # Step 3.5: Load baseline dynamics data (all samples are successful)
        logger.info("Loading baseline dynamics data...")
        baseline_dynamics = load_baseline_dynamics(model_name, s3_manager)
        
        # Step 4: Process each noise level (all samples are successful)
        total_datasets = 0
        for noise_level in NOISE_LEVELS:
            logger.info(f"Processing noise level: {noise_level}")
            
            # Apply response noise to target data (all samples are successful)
            logger.info(f"Applying Gaussian noise (level={noise_level}) to target data...")
            
            noisy_target_data = apply_response_noise(
                baseline_targets, noise_level, SEED
            )
            
            # Generate complete dataset for this noise level using pre-computed baseline dynamics
            dataset_dict = generate_complete_dataset_for_noise_level(
                clean_dataset={
                    'features': baseline_features,
                    'targets': baseline_targets,
                    'parameters': baseline_parameters,
                    'metadata': None
                },
                noisy_target_data=noisy_target_data,
                noise_level=noise_level,
                baseline_dynamics=baseline_dynamics
            )
            
            # Save complete dataset to S3
            save_complete_dataset(dataset_dict, noise_level, model_name, s3_manager)
            
            total_datasets += 1
            logger.info(f"✅ Completed noise level {noise_level}")
        
        # Note: All baseline samples are successful (failed samples were removed during baseline generation)
        # Success rate is 100% by design
        if UPLOAD_S3 and s3_manager:
            logger.info("✅ All baseline samples successful (100% success rate)")
        
        logger.info(f"✅ Successfully processed model {model_name}: {total_datasets} datasets created")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to process model {model_name}: {e}")
        return False


def main():
    """Main execution function - configuration-based version"""
    
    # Send start notification if enabled
    if SEND_NOTIFICATIONS:
        script_name = 'response-noise-config'
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
            notify_failure('response-noise-config', error_msg, duration_seconds=duration)
        else:
            notify_success('response-noise-config', duration, processed_count=len(successful_models))
    
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