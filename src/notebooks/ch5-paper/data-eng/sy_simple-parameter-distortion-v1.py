"""
Simplified Parameter Distortion Data Generation Script

Generates distorted parameter sets with Gaussian noise.
Follows the simple S3 storage pattern of sy_simple-make-data-v1.py.
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
from scripts.ntfy_notifier import notify_start, notify_success, notify_failure
from numpy.random import default_rng

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_gaussian_distortion(original_params, distortion_factor, seed=42):
    """
    Apply Gaussian noise distortion to parameters
    
    Args:
        original_params: Dictionary of original parameters from model_builder
        distortion_factor: Controls strength of distortion (0 = no distortion)
        seed: Random seed for reproducible distortion
        noise_std: Standard deviation of Gaussian noise
    
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
        distorted_params[key] = max(value + noise_amount, 1e-8) # Ensure non-negative
    
    return distorted_params


def generate_distorted_parameter_sets(model_builder, distortion_factors, n_samples=2000, seed=42):
    """
    Generate multiple parameter sets with different distortion levels
    
    Args:
        model_builder: ModelBuilder instance
        distortion_factors: List of distortion factors to apply
        n_samples: Number of parameter sets per distortion factor
        seed: Random seed for reproducibility
        noise_std: Standard deviation of Gaussian noise
    
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


def save_distorted_parameter_sets(all_distorted_sets, model_name, s3_manager):
    """
    Save distorted parameter sets to S3 following sy_simple-make-data-v1.py pattern
    
    Args:
        all_distorted_sets: Dictionary mapping distortion_factor -> list of parameter dicts
        model_name: Name of the model
        s3_manager: S3ConfigManager instance
    """
    gen_path = s3_manager.save_result_path
    folder_name = f"{model_name}_parameter_distortion_simple_v1"
    
    # Create DataFrames and save each distortion level
    for distortion_factor, parameter_sets in all_distorted_sets.items():
        # Convert list of dicts to DataFrame
        params_df = pd.DataFrame(parameter_sets)
        
        # Add metadata columns
        params_df['distortion_factor'] = distortion_factor
        params_df['sample_id'] = range(len(parameter_sets))
        params_df['generation_timestamp'] = datetime.now().isoformat()
        
        # Create filename
        file_name = f"distortion_{distortion_factor}_parameter_sets.pkl"
        s3_path = f"{gen_path}/data/{folder_name}/{file_name}"
        
        # Save to S3
        s3_manager.save_data_from_path(s3_path, params_df, data_format="pkl")
        logger.info(f"✅ Saved {len(parameter_sets)} parameter sets to S3: {s3_path}")
    
    # Save metadata file
    metadata = {
        'model_name': model_name,
        'distortion_factors': list(all_distorted_sets.keys()),
        'n_samples_per_factor': len(next(iter(all_distorted_sets.values()))),
        'total_parameter_sets': sum(len(sets) for sets in all_distorted_sets.values()),
        'generation_timestamp': datetime.now().isoformat(),
        'script_version': 'simple_v1'
    }
    
    metadata_path = f"{gen_path}/data/{folder_name}/metadata.pkl"
    s3_manager.save_data_from_path(metadata_path, metadata, data_format="pkl")
    logger.info(f"✅ Saved metadata to S3: {metadata_path}")
    
    # List files for verification
    item_list = s3_manager.list_files_from_path(f"{gen_path}/data/{folder_name}/")
    
    logger.info("Files saved to S3:")
    for item in item_list:
        logger.info(f"  - {item}")


def load_model_from_s3(model_name, s3_manager):
    """
    Load model builder from S3
    
    Args:
        model_name: Name of the model (e.g., 'sy_simple')
        s3_manager: S3ConfigManager instance
    
    Returns:
        model_builder: ModelBuilder instance
    """
    gen_path = s3_manager.save_result_path
    
    logger.info(f"Loading model builder for: {model_name}")
    model_builder = s3_manager.load_data_from_path(
        f"{gen_path}/models/{model_name}/model_builder.pkl", 
        data_format='pkl'
    )
    
    logger.info(f"✅ Loaded model builder for {model_name}")
    return model_builder


def main():
    """Main execution function"""
    logger.info("🚀 Starting Simplified Parameter Distortion Data Generation")
    
    # Send start notification
    script_name = 'parameter-distortion'
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
        
        # Load model
        model_builder = load_model_from_s3(model_name, s3_manager)
        
        # Generate distorted parameter sets
        all_distorted_sets = generate_distorted_parameter_sets(
            model_builder, distortion_factors, n_samples, seed
        )
        
        # Save to S3
        save_distorted_parameter_sets(all_distorted_sets, model_name, s3_manager)
        
        # Calculate execution time
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Send success notification
        notify_success(script_name, duration, processed_count=len(distortion_factors))
        
        logger.info("✅ Parameter distortion data generation completed successfully")
        logger.info(f"Generated {len(distortion_factors)} distortion levels")
        logger.info(f"Total parameter sets: {sum(len(sets) for sets in all_distorted_sets.values())}")
        logger.info(f"Total execution time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        
    except Exception as e:
        logger.error(f"❌ Parameter distortion data generation failed: {e}")
        # Send failure notification
        duration = (datetime.now() - start_time).total_seconds()
        notify_failure(script_name, e, duration_seconds=duration)
        raise


if __name__ == "__main__":
    main()
