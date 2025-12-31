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
from models.utils.data_generation_helpers import make_data_extended
from models.utils.dynamic_calculations import dynamic_features_method, last_time_point_method
from models.Solver.RoadrunnerSolver import RoadrunnerSolver
from scripts.ntfy_notifier import notify_start, notify_success, notify_failure
from numpy.random import default_rng
from tqdm import tqdm

# Import shared utilities for CSV generation
from ml_task_utils import BaseTaskGenerator, save_task_csv, print_task_summary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===== CONFIGURATION SECTION =====
# MODIFY THESE VARIABLES FOR YOUR BATCH JOB
MODEL_NAME = "sy_simple"  # Can be string: "sy_simple" or list: ["sy_simple", "model_v2"]
NOISE_LEVELS = [0, 0.05, 0.1, 0.2, 0.3, 0.5]
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
            }
        ]
        
    def get_target_files(self, noise_level):
        """Get target files for a given noise level"""
        base_path = f"{self.get_base_folder()}/noise_{noise_level}"
        
        # For response noise, we have both clean_targets and noisy_targets
        # Based on user requirements, use clean_targets (original)
        return [
            {
                "path": f"{base_path}/clean_targets.pkl",
                "label": "original_targets"
            }
        ]


def apply_response_noise(target_data, noise_level, seed):
    """
    Apply Gaussian noise to response target data
    
    Args:
        target_data: Original target data DataFrame
        noise_level: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility
    
    Returns:
        Target data with applied noise
    """
    rng = default_rng(seed)
    
    # Apply noise to each column independently
    noisy_target_data = target_data.copy()
    for column in target_data.columns:
        original_values = target_data[column].values
        noise = rng.normal(0, noise_level * np.std(original_values), len(original_values))
        noisy_target_data[column] = original_values + noise
    
    return noisy_target_data


def generate_complete_dataset_for_noise_level(
    clean_dataset,
    noisy_target_data,
    noise_level,
    model_builder,
    solver
):
    """
    Generate complete dataset for a specific noise level
    
    Args:
        clean_dataset: Dictionary containing clean dataset components from make_data_extended
        noisy_target_data: Target data with applied noise
        noise_level: Current noise level
        solver: Solver instance
    
    Returns:
        Dictionary containing all dataset components
    """
    logger.info(f"Generating complete dataset for noise level {noise_level}")
    
    # Use timecourse data from clean dataset
    timecourse_data = clean_dataset['timecourse']
    
    # Calculate dynamic features from clean timecourse
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
        'features': clean_dataset['features'],
        'clean_targets': clean_dataset['targets'],
        'noisy_targets': noisy_target_data,
        'parameters': clean_dataset['parameters'],
        'timecourses': timecourse_data,
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
    folder_name = f"{model_name}_response_noise_v1"
    subfolder_name = f"noise_{noise_level}"
    full_path = f"{gen_path}/data/{folder_name}/{subfolder_name}"
    
    # Save each component
    file_mappings = {
        'features': 'features.pkl',
        'clean_targets': 'clean_targets.pkl',
        'noisy_targets': 'noisy_targets.pkl',
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


def generate_csv_task_list():
    """
    Generate CSV task list for response noise experiments.
    
    This function provides a configuration-based interface for generating
    CSV task lists without running the full data generation pipeline.
    """
    if not GENERATE_ML_TASK_LIST:
        logger.info("Skipping ML task list generation (GENERATE_ML_TASK_LIST=False)")
        return pd.DataFrame()
    
    logger.info("🚀 Generating CSV task list for response-noise-v1")
    
    # Process model configuration
    model_names = process_model_config(MODEL_NAME)
    
    all_task_rows = []
    
    # Initialize S3 manager if UPLOAD_S3 is True
    s3_manager = S3ConfigManager() if UPLOAD_S3 else None
    
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
        output_csv = "task_list.csv"
        task_df = pd.DataFrame(all_task_rows)
        
        # Save locally
        task_df.to_csv(output_csv, index=False)
        logger.info(f"✅ Generated task list with {len(task_df)} rows to: {output_csv}")
        
        # Upload to S3 if enabled
        if UPLOAD_S3 and s3_manager:
            folder_name = generator.get_base_folder() if 'generator' in locals() else f"{model_names[0]}_response_noise_v1"
            gen_path = s3_manager.save_result_path
            s3_csv_path = f"{gen_path}/data/{folder_name}/{output_csv}"
            
            s3_manager.save_data_from_path(s3_csv_path, task_df, data_format="csv")
            logger.info(f"✅ Uploaded CSV to S3: {s3_csv_path}")
        
        if len(model_names) > 1:
            print_task_summary(task_df)
        
        return task_df
    else:
        logger.warning("No tasks generated")
        return pd.DataFrame()


def process_single_model(model_name, s3_manager):
    """
    Process a single model through the response noise pipeline.
    
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
        
        # Get initial values (inactive state variables)
        state_variables = model_builder.get_state_variables()
        initial_values = {k: v for k, v in state_variables.items() if not k.endswith('a')}
        if 'O' in initial_values:
            del initial_values['O']
        
        # Get kinetic parameters
        kinetic_parameters = model_builder.get_parameters()
        
        # Generate clean dataset using make_data_extended (once for all noise levels)
        logger.info("Generating clean dataset using make_data_extended...")
        clean_dataset = make_data_extended(
            initial_values=initial_values,
            perturbation_type="lognormal",
            perturbation_params={"shape": 0.5},
            parameter_values=kinetic_parameters,
            param_perturbation_type="lognormal",
            param_perturbation_params={"shape": 0.5},
            n_samples=N_SAMPLES,
            model_spec=model_spec,
            solver=solver,
            simulation_params=SIMULATION_PARAMS,
            seed=SEED,
            outcome_var=OUTCOME_VAR,
            capture_all_species=True,
        )
        
        X, y, parameters, timecourses, metadata = (
            clean_dataset["features"],
            clean_dataset["targets"],
            clean_dataset["parameters"],
            clean_dataset["timecourse"],
            clean_dataset["metadata"],
        )
        
        logger.info(f"Generated clean dataset with {X.shape[0]} samples and {X.shape[1]} features.")
        logger.info(f"Target variable shape: {y.shape}")
        
        # Ensure parameters are clean (no metadata columns) by checking and filtering if needed
        if isinstance(parameters, pd.DataFrame):
            # Remove any potential metadata columns that aren't actual SBML parameters
            # Get actual SBML parameter names from the model
            sbml_params = set(model_builder.get_parameters().keys())
            actual_param_cols = [col for col in parameters.columns if col in sbml_params]
            if len(actual_param_cols) < len(parameters.columns):
                logger.info(f"Filtering parameters: keeping {len(actual_param_cols)} SBML parameters, removing {len(parameters.columns) - len(actual_param_cols)} metadata columns")
                parameters = parameters[actual_param_cols]
        
        # Process each noise level
        total_datasets = 0
        for noise_level in NOISE_LEVELS:
            logger.info(f"Processing noise level: {noise_level}")
            
            # Apply response noise to target data
            logger.info(f"Applying Gaussian noise (level={noise_level}) to target data...")
            noisy_target_data = apply_response_noise(y, noise_level, SEED)
            
            # Generate complete dataset for this noise level
            dataset_dict = generate_complete_dataset_for_noise_level(
                clean_dataset={
                    'features': X,
                    'targets': y,
                    'parameters': parameters,
                    'timecourse': timecourses,
                    'metadata': metadata
                },
                noisy_target_data=noisy_target_data,
                noise_level=noise_level,
                model_builder=model_builder,
                solver=solver
            )
            
            # Save complete dataset to S3
            save_complete_dataset(dataset_dict, noise_level, model_name, s3_manager)
            
            total_datasets += 1
            logger.info(f"✅ Completed noise level {noise_level}")
        
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
