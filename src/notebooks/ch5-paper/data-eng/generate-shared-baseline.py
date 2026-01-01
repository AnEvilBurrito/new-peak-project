"""
Standalone Shared Baseline Generation Script - Configuration Version

Generates shared baseline virtual models for all noise/distortion experiments.
This script should be run ONCE before running expression-noise-v1.py,
parameter-distortion-v2.py, or response-noise-v1.py.

Follows the configuration-based style of other data generation scripts.
Uses the shared baseline_generator module for actual generation.
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===== CONFIGURATION SECTION =====
# MODIFY THESE VARIABLES FOR YOUR BATCH JOB
MODEL_NAME = ["sy_simple", "v1"]  # Can be string: "sy_simple" or list: ["sy_simple", "model_v2"]
N_SAMPLES = 2000
SEED = 42
SIMULATION_PARAMS = {'start': 0, 'end': 10000, 'points': 101}
OUTCOME_VAR = "Oa"
UPLOAD_S3 = True
SEND_NOTIFICATIONS = True
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


def check_existing_baseline(model_name, s3_manager):
    """
    Check if baseline already exists in S3 for this model.
    
    Args:
        model_name: Name of the model
        s3_manager: S3ConfigManager instance
    
    Returns:
        True if baseline exists, False otherwise
    """
    try:
        from baseline_generator import load_baseline_from_s3
        baseline_data = load_baseline_from_s3(model_name, s3_manager)
        
        # Check if we have the essential components
        if (baseline_data.get('features') is not None and 
            baseline_data.get('targets') is not None):
            n_samples = len(baseline_data['features'])
            logger.info(f"Existing baseline found for {model_name} with {n_samples} samples")
            return True
        else:
            logger.warning(f"Incomplete baseline found for {model_name}")
            return False
            
    except Exception as e:
        logger.debug(f"No existing baseline found for {model_name}: {e}")
        return False


def process_single_model(model_name, s3_manager):
    """
    Process a single model through the shared baseline generation pipeline.
    
    Args:
        model_name: Name of the model to process
        s3_manager: S3ConfigManager instance
    
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"🚀 Starting shared baseline generation for model: {model_name}")
        
        # Step 1: Check if baseline already exists
        if check_existing_baseline(model_name, s3_manager):
            logger.warning(f"⚠️ Baseline already exists for {model_name}. Skipping generation.")
            logger.warning(f"   If you want to regenerate, delete the existing baseline from S3.")
            return True  # Treat as success since baseline exists
        
        # Step 2: Load model objects
        model_spec, model_builder, model_tuner = load_model_objects(model_name, s3_manager)
        
        # Step 3: Setup solver
        from models.Solver.RoadrunnerSolver import RoadrunnerSolver
        solver = RoadrunnerSolver()
        solver.compile(model_builder.get_sbml_model())
        
        # Step 4: Generate baseline virtual models
        logger.info(f"Generating shared baseline with {N_SAMPLES} samples...")
        from baseline_generator import generate_baseline_virtual_models, save_baseline_to_s3
        
        baseline_data = generate_baseline_virtual_models(
            model_spec=model_spec,
            model_builder=model_builder,
            solver=solver,
            n_samples=N_SAMPLES,
            seed=SEED,
            simulation_params=SIMULATION_PARAMS
        )
        
        logger.info(f"✅ Generated baseline with {len(baseline_data['features'])} virtual models")
        
        # Step 5: Save to S3
        if UPLOAD_S3:
            baseline_paths = save_baseline_to_s3(
                baseline_data, model_name, s3_manager, upload=UPLOAD_S3
            )
            logger.info(f"✅ Saved shared baseline to S3")
            
            # Log the paths for reference
            for key, path in baseline_paths.items():
                logger.info(f"   {key}: {path}")
        
        logger.info(f"✅ Successfully processed model {model_name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to process model {model_name}: {e}")
        return False


def main():
    """Main execution function - configuration-based version"""
    
    # Send start notification if enabled
    if SEND_NOTIFICATIONS:
        script_name = 'generate-shared-baseline'
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
    
    # Calculate execution time
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Send notifications if enabled
    if SEND_NOTIFICATIONS:
        if failed_models:
            error_msg = f"Failed models: {failed_models}"
            notify_failure('generate-shared-baseline', error_msg, duration_seconds=duration)
        else:
            notify_success('generate-shared-baseline', duration, processed_count=len(successful_models))
    
    # Summary
    logger.info("=" * 60)
    logger.info("SHARED BASELINE GENERATION SUMMARY")
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
