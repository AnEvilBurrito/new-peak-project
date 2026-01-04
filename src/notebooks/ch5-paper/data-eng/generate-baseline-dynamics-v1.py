"""
Baseline Dynamics Data Generation Script - Configuration Version

Loads baseline virtual models from generate-shared-baseline.py, calculates dynamic features
from timecourses, and generates CSV task list for ML batch evaluation.

This script should be run AFTER generate-shared-baseline.py to create baseline dynamics
datasets for ML evaluation.

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
from models.utils.dynamic_calculations import dynamic_features_method, last_time_point_method
from scripts.ntfy_notifier import notify_start, notify_success, notify_failure

# Import shared utilities for CSV generation
from ml_task_utils import BaseTaskGenerator, save_task_csv, print_task_summary
from baseline_dynamics_task_generator import BaselineDynamicsTaskGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===== CONFIGURATION SECTION =====
# MODIFY THESE VARIABLES FOR YOUR BATCH JOB
MODEL_NAME = ["sy_simple", "v1"]  # Can be string: "sy_simple" or list: ["sy_simple", "model_v2"]
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


def load_baseline_data(model_name, s3_manager):
    """
    Load baseline virtual models from S3.
    
    Args:
        model_name: Name of the model
        s3_manager: S3ConfigManager instance
    
    Returns:
        Dictionary containing baseline data components
    """
    try:
        from baseline_generator import load_baseline_from_s3
        baseline_data = load_baseline_from_s3(model_name, s3_manager)
        
        # Check if we have the essential components
        if (baseline_data.get('features') is None or 
            baseline_data.get('targets') is None):
            raise RuntimeError(f"Incomplete baseline found for {model_name}. Missing essential components.")
        
        baseline_samples = len(baseline_data['features'])
        logger.info(f"✅ Loaded baseline for {model_name} with {baseline_samples} samples")
        
        return baseline_data
        
    except Exception as e:
        if "Could not load" in str(e) or "No such file" in str(e):
            raise RuntimeError(
                f"❌ Shared baseline not found for model {model_name}. "
                f"Please run generate-shared-baseline.py first to create the baseline."
            )
        else:
            raise RuntimeError(f"❌ Error loading baseline for {model_name}: {e}")


def calculate_dynamic_features(timecourse_data, model_builder):
    """
    Calculate dynamic features from timecourse data (both with and without outcome variable).
    
    Args:
        timecourse_data: Timecourse data from baseline virtual models
        model_builder: ModelBuilder instance for state variable information
    
    Returns:
        Tuple of (dynamic_features_with_outcome, last_time_points_with_outcome,
                 dynamic_features_no_outcome, last_time_points_no_outcome)
    """
    logger.info("Calculating dynamic features from timecourses (with and without outcome)...")
    
    # Get all state variables ending with 'a' (active state variables)
    all_state_vars = {k: v for k, v in model_builder.get_state_variables().items() if k.endswith('a')}
    
    # Calculate dynamic features for ALL variables (including outcome)
    logger.info("Calculating dynamic features...")
    all_dynamic_features = dynamic_features_method(
        timecourse_data, 
        selected_features=list(all_state_vars.keys()), 
        n_cores=1, 
        verbose=False
    )
    
    # Calculate last time points for ALL variables (including outcome)
    all_last_time_points = last_time_point_method(
        timecourse_data, 
        selected_species=list(all_state_vars.keys())
    )
    
    # Extract base variable names from dynamic features columns
    # Split on the last underscore to get base variable name
    def get_base_variable_name(column_name):
        # Split on underscore and remove the suffix (last part after last underscore)
        parts = column_name.split('_')
        if len(parts) > 1:
            # Join all parts except the last one (the suffix)
            return '_'.join(parts[:-1])
        return column_name
    
    # Get unique base variables from dynamic features columns
    dynamic_features_columns = all_dynamic_features.columns
    base_variables = set(get_base_variable_name(col) for col in dynamic_features_columns)
    
    # Filter for variables WITHOUT outcome
    base_variables_no_outcome = {var for var in base_variables 
                                 if var not in ['O', 'Oa']}
    
    # Filter columns for no outcome version
    columns_with_outcome = [col for col in dynamic_features_columns]
    columns_no_outcome = [col for col in dynamic_features_columns 
                         if get_base_variable_name(col) in base_variables_no_outcome]
    
    # Filter last time points columns
    last_time_points_columns = all_last_time_points.columns
    last_time_points_columns_no_outcome = [col for col in last_time_points_columns 
                                         if col not in ['O', 'Oa']]
    
    # Create filtered DataFrames
    dynamic_features_with_outcome = all_dynamic_features[columns_with_outcome]
    dynamic_features_no_outcome = all_dynamic_features[columns_no_outcome]
    
    last_time_points_with_outcome = all_last_time_points
    last_time_points_no_outcome = all_last_time_points[last_time_points_columns_no_outcome]
    
    logger.info(f"Dynamic features (with outcome) shape: {dynamic_features_with_outcome.shape}")
    logger.info(f"Last time points (with outcome) shape: {last_time_points_with_outcome.shape}")
    logger.info(f"Dynamic features (no outcome) shape: {dynamic_features_no_outcome.shape}")
    logger.info(f"Last time points (no outcome) shape: {last_time_points_no_outcome.shape}")
    
    return (dynamic_features_with_outcome, last_time_points_with_outcome,
            dynamic_features_no_outcome, last_time_points_no_outcome)


def load_model_objects(model_name, s3_manager):
    """
    Load model builder from S3 for state variable information.
    
    Args:
        model_name: Name of the model (e.g., 'sy_simple')
        s3_manager: S3ConfigManager instance
    
    Returns:
        ModelBuilder instance
    """
    gen_path = s3_manager.save_result_path
    
    logger.info(f"Loading model builder for: {model_name}")
    
    model_builder = s3_manager.load_data_from_path(
        f"{gen_path}/models/{model_name}/model_builder.pkl", 
        data_format='pkl'
    )
    
    logger.info(f"✅ Loaded model builder for {model_name}")
    return model_builder


def create_combined_feature_sets(original_features, dynamic_features_with_outcome, 
                                last_time_points_with_outcome, dynamic_features_no_outcome,
                                last_time_points_no_outcome):
    """
    Create combined feature sets from individual feature sets.
    
    Args:
        original_features: DataFrame of original features
        dynamic_features_with_outcome: DataFrame of dynamic features with outcome
        last_time_points_with_outcome: DataFrame of last time points with outcome
        dynamic_features_no_outcome: DataFrame of dynamic features without outcome
        last_time_points_no_outcome: DataFrame of last time points without outcome
    
    Returns:
        Dictionary of combined feature DataFrames
    """
    logger.info("Creating combined feature sets...")
    
    # Validate all DataFrames have same shape and indices
    feature_sets = {
        'original': original_features,
        'dynamic_with_outcome': dynamic_features_with_outcome,
        'last_with_outcome': last_time_points_with_outcome,
        'dynamic_no_outcome': dynamic_features_no_outcome,
        'last_no_outcome': last_time_points_no_outcome
    }
    
    # Check consistency
    for name, df in feature_sets.items():
        if df is None or len(df) == 0:
            logger.warning(f"Empty DataFrame for {name}, skipping combinations")
            return {}
    
    # Get reference shape and index
    ref_shape = original_features.shape
    ref_index = original_features.index
    
    for name, df in feature_sets.items():
        if df.shape[0] != ref_shape[0]:
            raise ValueError(f"Row count mismatch: {name} has {df.shape[0]} rows, original has {ref_shape[0]}")
        if not df.index.equals(ref_index):
            raise ValueError(f"Index mismatch for {name}")
    
    combined_sets = {}
    
    # Helper function to concatenate with smart suffixing
    # Dynamic features already have descriptive suffixes (_auc, _median, etc.)
    # Last time points need suffixes to identify them
    # Original features need suffixes to distinguish from dynamic features
    def concat_with_smart_suffix(df1, df2, df1_type="original", df2_type="dynamic_no_outcome"):
        """
        Concatenate DataFrames with intelligent suffixing based on feature type.
        
        Args:
            df1: First DataFrame
            df2: Second DataFrame  
            df1_type: Type of features in df1 ('original', 'dynamic', 'last')
            df2_type: Type of features in df2 ('original', 'dynamic', 'last')
        """
        # Map feature types to suffix strategies
        suffix_strategies = {
            'original': '_original',  # Original features need suffix
            'dynamic_with_outcome': '',  # Dynamic features already have descriptive suffixes
            'dynamic_no_outcome': '',    # Dynamic features already have descriptive suffixes
            'last_with_outcome': '_last',  # Last time points need suffix
            'last_no_outcome': '_last'      # Last time points need suffix
        }
        
        suffix1 = suffix_strategies.get(df1_type, '')
        suffix2 = suffix_strategies.get(df2_type, '')
        
        # Only add suffix if not empty
        if suffix1:
            df1_suffixed = df1.add_suffix(suffix1)
        else:
            df1_suffixed = df1
            
        if suffix2:
            df2_suffixed = df2.add_suffix(suffix2)
        else:
            df2_suffixed = df2
        
        # Concatenate
        combined = pd.concat([df1_suffixed, df2_suffixed], axis=1)
        return combined
    
    # Create combined feature sets
    # 1. Original + dynamic features without outcome (user's example)
    combined_sets['original_plus_dynamic_no_outcome'] = concat_with_smart_suffix(
        original_features, dynamic_features_no_outcome,
        df1_type="original", df2_type="dynamic_no_outcome"
    )
    
    # 2. Original + last time points without outcome
    combined_sets['original_plus_last_no_outcome'] = concat_with_smart_suffix(
        original_features, last_time_points_no_outcome,
        df1_type="original", df2_type="last_no_outcome"
    )
    
    # 3. Original + dynamic features with outcome
    combined_sets['original_plus_dynamic_with_outcome'] = concat_with_smart_suffix(
        original_features, dynamic_features_with_outcome,
        df1_type="original", df2_type="dynamic_with_outcome"
    )
    
    # 4. Original + last time points with outcome
    combined_sets['original_plus_last_with_outcome'] = concat_with_smart_suffix(
        original_features, last_time_points_with_outcome,
        df1_type="original", df2_type="last_with_outcome"
    )
    
    logger.info(f"Created {len(combined_sets)} combined feature sets")
    for name, df in combined_sets.items():
        logger.info(f"  {name}: shape {df.shape}")
    
    return combined_sets


def save_baseline_dynamics_data(baseline_data, dynamic_features_tuple, model_name, s3_manager):
    """
    Save baseline dynamics datasets to S3 (with and without outcome versions).
    
    Args:
        baseline_data: Baseline data from generate-shared-baseline.py
        dynamic_features_tuple: Tuple of (dynamic_features_with_outcome, last_time_points_with_outcome,
                                         dynamic_features_no_outcome, last_time_points_no_outcome)
        model_name: Name of the model
        s3_manager: S3ConfigManager instance
    """
    if not UPLOAD_S3:
        logger.info(f"Skipping S3 upload for model {model_name}")
        return
    
    # Unpack the tuple
    (dynamic_features_with_outcome, last_time_points_with_outcome,
     dynamic_features_no_outcome, last_time_points_no_outcome) = dynamic_features_tuple
    
    gen_path = s3_manager.save_result_path
    folder_name = f"{model_name}_baseline_dynamics_v1"
    full_path = f"{gen_path}/data/{folder_name}"
    
    # Save original features
    original_features = baseline_data['features']
    s3_manager.save_data_from_path(
        f"{full_path}/original_features.pkl", 
        original_features, 
        data_format="pkl"
    )
    logger.info(f"✅ Saved original features to S3: {full_path}/original_features.pkl")
    
    # Save baseline targets
    baseline_targets = baseline_data['targets']
    s3_manager.save_data_from_path(
        f"{full_path}/baseline_targets.pkl", 
        baseline_targets, 
        data_format="pkl"
    )
    logger.info(f"✅ Saved baseline targets to S3: {full_path}/baseline_targets.pkl")
    
    # Save dynamic features WITH outcome
    s3_manager.save_data_from_path(
        f"{full_path}/dynamic_features_with_outcome.pkl", 
        dynamic_features_with_outcome, 
        data_format="pkl"
    )
    logger.info(f"✅ Saved dynamic features (with outcome) to S3: {full_path}/dynamic_features_with_outcome.pkl")
    
    # Save last time points WITH outcome
    s3_manager.save_data_from_path(
        f"{full_path}/last_time_points_with_outcome.pkl", 
        last_time_points_with_outcome, 
        data_format="pkl"
    )
    logger.info(f"✅ Saved last time points (with outcome) to S3: {full_path}/last_time_points_with_outcome.pkl")
    
    # Save dynamic features WITHOUT outcome
    s3_manager.save_data_from_path(
        f"{full_path}/dynamic_features_no_outcome.pkl", 
        dynamic_features_no_outcome, 
        data_format="pkl"
    )
    logger.info(f"✅ Saved dynamic features (no outcome) to S3: {full_path}/dynamic_features_no_outcome.pkl")
    
    # Save last time points WITHOUT outcome
    s3_manager.save_data_from_path(
        f"{full_path}/last_time_points_no_outcome.pkl", 
        last_time_points_no_outcome, 
        data_format="pkl"
    )
    logger.info(f"✅ Saved last time points (no outcome) to S3: {full_path}/last_time_points_no_outcome.pkl")
    
    # Save timecourses (optional, for reference)
    if baseline_data.get('timecourses') is not None:
        s3_manager.save_data_from_path(
            f"{full_path}/timecourses.pkl", 
            baseline_data['timecourses'], 
            data_format="pkl"
        )
        logger.info(f"✅ Saved timecourses to S3: {full_path}/timecourses.pkl")
    
    # Create and save combined feature sets
    combined_features = create_combined_feature_sets(
        original_features=original_features,
        dynamic_features_with_outcome=dynamic_features_with_outcome,
        last_time_points_with_outcome=last_time_points_with_outcome,
        dynamic_features_no_outcome=dynamic_features_no_outcome,
        last_time_points_no_outcome=last_time_points_no_outcome
    )
    
    # Save combined feature sets
    for combined_name, combined_df in combined_features.items():
        s3_path = f"{full_path}/{combined_name}.pkl"
        s3_manager.save_data_from_path(s3_path, combined_df, data_format="pkl")
        logger.info(f"✅ Saved combined feature set to S3: {s3_path}")
    
    # Save metadata
    all_feature_types = [
        'original_features', 
        'dynamic_features_with_outcome', 
        'last_time_points_with_outcome',
        'dynamic_features_no_outcome', 
        'last_time_points_no_outcome'
    ]
    
    # Add combined feature types to metadata
    if combined_features:
        combined_names = list(combined_features.keys())
        all_feature_types.extend(combined_names)
    
    metadata = {
        'model_name': model_name,
        'n_samples': len(original_features),
        'generation_timestamp': datetime.now().isoformat(),
        'feature_types': all_feature_types,
        'combined_feature_sets_generated': bool(combined_features),
        'combined_feature_set_names': list(combined_features.keys()) if combined_features else [],
        'source_baseline': f"{model_name}_baseline_virtual_models",
        'script_version': 'generate-baseline-dynamics-v1.py',
        'outcome_variable_included': 'Oa' if 'Oa' in dynamic_features_with_outcome.columns.str.split('_').str[0].unique() else 'unknown'
    }
    
    s3_manager.save_data_from_path(
        f"{full_path}/metadata.pkl", 
        metadata, 
        data_format="pkl"
    )
    logger.info(f"✅ Saved metadata to S3: {full_path}/metadata.pkl")


def generate_csv_task_list(model_names, s3_manager):
    """
    Generate CSV task list for baseline dynamics experiments.
    
    Args:
        model_names: List of model names
        s3_manager: S3ConfigManager instance
    
    Returns:
        DataFrame containing the task list
    """
    if not GENERATE_ML_TASK_LIST:
        logger.info("Skipping ML task list generation (GENERATE_ML_TASK_LIST=False)")
        return pd.DataFrame()
    
    # Check if S3 upload is enabled
    if not UPLOAD_S3:
        logger.info("Skipping ML task list generation (UPLOAD_S3=False)")
        return pd.DataFrame()
    
    logger.info("🚀 Generating CSV task list for baseline-dynamics-v1")
    
    all_task_rows = []
    
    for model_name in model_names:
        logger.info(f"Generating tasks for model: {model_name}")
        
        # Create task generator
        generator = BaselineDynamicsTaskGenerator(model_name=model_name)
        
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
            # Use the first model's folder name for CSV storage
            folder_name = f"{model_names[0]}_baseline_dynamics_v1"
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
    Process a single model: load baseline, calculate dynamics, save data.
    
    Args:
        model_name: Name of the model to process
        s3_manager: S3ConfigManager instance
    
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"🚀 Starting baseline dynamics generation for model: {model_name}")
        
        # Step 1: Load baseline data
        baseline_data = load_baseline_data(model_name, s3_manager)
        
        # Step 2: Load model builder for state variable information
        model_builder = load_model_objects(model_name, s3_manager)
        
        # Step 3: Calculate dynamic features from timecourses
        timecourse_data = baseline_data.get('timecourses')
        if timecourse_data is None:
            logger.warning(f"No timecourse data found for {model_name}. Skipping dynamic feature calculation.")
            # Create empty tuple for consistency
            dynamic_features_tuple = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        else:
            # Now returns 4-tuple: (with_outcome_dyn, with_outcome_last, no_outcome_dyn, no_outcome_last)
            dynamic_features_tuple = calculate_dynamic_features(timecourse_data, model_builder)
        
        # Step 4: Save baseline dynamics datasets to S3
        save_baseline_dynamics_data(
            baseline_data, 
            dynamic_features_tuple, 
            model_name, 
            s3_manager
        )
        
        logger.info(f"✅ Successfully processed model {model_name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to process model {model_name}: {e}")
        return False


def main():
    """Main execution function - configuration-based version"""
    
    # Send start notification if enabled
    if SEND_NOTIFICATIONS:
        script_name = 'baseline-dynamics'
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
        task_df = generate_csv_task_list(model_names, s3_manager)
        if not task_df.empty:
            logger.info(f"✅ Generated task list with {len(task_df)} rows")
    
    # Calculate execution time
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Send notifications if enabled
    if SEND_NOTIFICATIONS:
        if failed_models:
            error_msg = f"Failed models: {failed_models}"
            notify_failure('baseline-dynamics', error_msg, duration_seconds=duration)
        else:
            notify_success('baseline-dynamics', duration, processed_count=len(successful_models))
    
    # Summary
    logger.info("=" * 60)
    logger.info("BASELINE DYNAMICS GENERATION SUMMARY")
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