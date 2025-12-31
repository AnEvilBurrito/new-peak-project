"""
Complete Response Noise Data Generation Script

Generates comprehensive datasets with Gaussian noise applied to target data including:
- Clean feature data generated with make_data_extended
- Target data with applied Gaussian noise at different levels
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


class ResponseNoiseTaskGenerator(BaseTaskGenerator):
    """
    Task generator for response-noise-v1 experiment pattern.
    
    This class encapsulates the pattern-specific logic for generating
    CSV task lists for response noise experiments.
    """
    
    def __init__(self, model_name: str = "sy_simple"):
        super().__init__(model_name)
        self.experiment_type = "response-noise-v1"
        self.noise_levels = [0, 0.05, 0.1, 0.2, 0.3, 0.5]
        
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
    
    This function provides a command-line interface for generating
    CSV task lists without running the full data generation pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Generate CSV task list for response-noise-v1 experiments"
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
    
    logger.info("🚀 Generating CSV task list for response-noise-v1")
    
    # Create task generator
    generator = ResponseNoiseTaskGenerator(model_name=args.model)
    
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
    
    logger.info("🚀 Starting Complete Response Noise Data Generation")
    
    # Send start notification
    script_name = 'response-noise'
    notify_start(script_name)
    
    start_time = datetime.now()
    
    try:
        # Initialize S3 manager
        s3_manager = S3ConfigManager()
        
        # Configuration
        model_name = "sy_simple"
        noise_levels = [0, 0.05, 0.1, 0.2, 0.3, 0.5]
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
        
        # Get kinetic parameters
        kinetic_parameters = model_builder.get_parameters()
        
        # Simulation parameters (matching sy_simple-make-data-v1.py)
        simulation_params = {'start': 0, 'end': 10000, 'points': 101}
        
        # Generate clean dataset using make_data_extended (once for all noise levels)
        logger.info("Generating clean dataset using make_data_extended...")
        clean_dataset = make_data_extended(
            initial_values=initial_values,
            perturbation_type="lognormal",
            perturbation_params={"shape": 0.5},
            parameter_values=kinetic_parameters,
            param_perturbation_type="lognormal",
            param_perturbation_params={"shape": 0.5},
            n_samples=n_samples,
            model_spec=model_spec,
            solver=solver,
            simulation_params=simulation_params,
            seed=seed,
            outcome_var="Oa",
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
        for noise_level in noise_levels:
            logger.info(f"Processing noise level: {noise_level}")
            
            # Apply response noise to target data
            logger.info(f"Applying Gaussian noise (level={noise_level}) to target data...")
            noisy_target_data = apply_response_noise(y, noise_level, seed)
            
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
        
        # Calculate execution time
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Send success notification
        notify_success(script_name, duration, processed_count=len(noise_levels))
        
        logger.info("✅ Complete response noise data generation finished successfully")
        logger.info(f"Generated {len(noise_levels)} noise levels")
        logger.info(f"Total datasets created: {total_datasets}")
        logger.info(f"Total execution time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        
        # List files for verification
        folder_name = f"{model_name}_response_noise_v1"
        item_list = s3_manager.list_files_from_path(f"{s3_manager.save_result_path}/data/{folder_name}/")
        
        logger.info("Files in S3:")
        for item in item_list:
            logger.info(f"  - {item}")
        
    except Exception as e:
        logger.error(f"❌ Response noise data generation failed: {e}")
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
