"""
ML Batch Loader Generator - Configuration Version

Creates a CSV task list for ML batch evaluation based on data generation patterns.
The CSV can be loaded by a BatchLoader class that interfaces with ml.Workflow.batch_eval functions.

Supports multiple experiment types:
- expression-noise-v1.py
- parameter-distortion-v2.py  
- response-noise-v1.py

CONFIGURATION-BASED VERSION:
For remote batch job execution where modifying script variables is more practical than CLI arguments.
Supports single model (string) or multiple models (list) for multiplexing.
"""

import sys
import os
import pandas as pd
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import logging

# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "../../..")
sys.path.insert(0, src_dir)

from models.utils.s3_config_manager import S3ConfigManager

# Import task generator classes from individual scripts using importlib to handle hyphens
import importlib.util

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===== CONFIGURATION SECTION =====
# MODIFY THESE VARIABLES FOR YOUR BATCH JOB
MODEL_NAME = "sy_simple"  # Can be string: "sy_simple" or list: ["sy_simple", "model_v2"]
EXPERIMENT_TYPES = ["expression-noise-v1", "parameter-distortion-v2", "response-noise-v1"]
OUTPUT_CSV = "ml_batch_tasks.csv"
VERIFY_EXISTS = True
GENERATE_ONLY = False
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


# Helper function to import from hyphenated filenames
def import_from_hyphenated_file(filepath, class_name):
    """Import a class from a Python file with hyphenated name"""
    module_name = filepath.replace('-', '_').replace('.py', '')
    file_dir = os.path.dirname(filepath)
    
    # Add the file's directory to sys.path so it can find local imports
    original_sys_path = sys.path.copy()
    if file_dir not in sys.path:
        sys.path.insert(0, file_dir)
    
    try:
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return getattr(module, class_name)
    finally:
        # Restore original sys.path
        sys.path = original_sys_path


# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Import task generator classes
ExpressionNoiseTaskGenerator = import_from_hyphenated_file(
    os.path.join(current_dir, "expression-noise-v1.py"),
    "ExpressionNoiseTaskGenerator"
)

ParameterDistortionTaskGenerator = import_from_hyphenated_file(
    os.path.join(current_dir, "parameter-distortion-v2.py"),
    "ParameterDistortionTaskGenerator"
)

ResponseNoiseTaskGenerator = import_from_hyphenated_file(
    os.path.join(current_dir, "response-noise-v1.py"),
    "ResponseNoiseTaskGenerator"
)


class BatchTaskGenerator:
    """Generates CSV task lists for ML batch evaluation"""
    
    def __init__(self, s3_manager: Optional[S3ConfigManager] = None):
        self.s3_manager = s3_manager or S3ConfigManager()
        self.experiment_generators = {
            "expression-noise-v1": ExpressionNoiseTaskGenerator,
            "parameter-distortion-v2": ParameterDistortionTaskGenerator,
            "response-noise-v1": ResponseNoiseTaskGenerator
        }
        
    def register_generator(self, name: str, generator_class):
        """Register a new experiment generator for extensibility"""
        self.experiment_generators[name] = generator_class
        
    def generate_task_list(
        self, 
        experiment_types: List[str],
        output_csv: str,
        model_names: List[str],
        verify_exists: bool = False
    ) -> pd.DataFrame:
        """
        Generate CSV task list for specified experiment types and models
        
        Args:
            experiment_types: List of experiment type names to include
            output_csv: Path to output CSV file
            model_names: List of model names to include
            verify_exists: If True, verify files exist in S3 before adding to list
            
        Returns:
            DataFrame containing the task list
        """
        task_rows = []
        
        for model_name in model_names:
            logger.info(f"Generating tasks for model: {model_name}")
            
            for exp_type in experiment_types:
                if exp_type not in self.experiment_generators:
                    logger.warning(f"Unknown experiment type: {exp_type}. Skipping.")
                    continue
                    
                generator_class = self.experiment_generators[exp_type]
                generator = generator_class(model_name=model_name)
                
                for level in generator.get_levels():
                    feature_files = generator.get_feature_files(level)
                    target_files = generator.get_target_files(level)
                    
                    # Create all combinations of feature and target files
                    for feature in feature_files:
                        for target in target_files:
                            # Verify file exists in S3 if requested
                            if verify_exists:
                                try:
                                    # Check if file exists in S3
                                    full_feature_path = f"{self.s3_manager.save_result_path}/data/{feature['path']}"
                                    full_target_path = f"{self.s3_manager.save_result_path}/data/{target['path']}"
                                    
                                    # Try to load metadata to check existence
                                    self.s3_manager.load_data_from_path(full_feature_path, data_format="pkl")
                                    self.s3_manager.load_data_from_path(full_target_path, data_format="pkl")
                                    
                                except Exception as e:
                                    logger.warning(f"File not found or error loading: {feature['path']} or {target['path']}. Skipping.")
                                    continue
                            
                            task_rows.append({
                                "feature_data": feature["path"],
                                "feature_data_label": feature["label"],
                                "target_data": target["path"],
                                "target_data_label": target["label"],
                                "experiment_type": exp_type,
                                "level": level,
                                "model_name": model_name
                            })
        
        task_df = pd.DataFrame(task_rows)
        
        # Save to CSV
        if output_csv:
            task_df.to_csv(output_csv, index=False)
            logger.info(f"✅ Generated task list with {len(task_df)} rows to: {output_csv}")
        
        return task_df


class BatchLoader:
    """Loads batch tasks and provides DataFrames for ML workflow integration"""
    
    def __init__(self, s3_manager: Optional[S3ConfigManager] = None):
        self.s3_manager = s3_manager or S3ConfigManager()
        self.task_df = None
        self.loaded_data = {}
        
    def load_task_list(self, csv_path: str):
        """Load task list from CSV file"""
        self.task_df = pd.read_csv(csv_path)
        logger.info(f"Loaded task list with {len(self.task_df)} tasks from: {csv_path}")
        return self.task_df
        
    def get_feature_target_pairs(self) -> List[Tuple[pd.DataFrame, str, pd.DataFrame, str]]:
        """
        Get feature/target pairs as DataFrames for batch_eval integration
        
        Returns:
            List of tuples: (feature_df, feature_label, target_df, target_label)
        """
        if self.task_df is None:
            raise ValueError("No task list loaded. Call load_task_list() first.")
            
        pairs = []
        
        for _, row in self.task_df.iterrows():
            try:
                # Load feature data from S3
                feature_path = f"{self.s3_manager.save_result_path}/data/{row['feature_data']}"
                feature_df = self.s3_manager.load_data_from_path(feature_path, data_format="pkl")
                
                # Load target data from S3
                target_path = f"{self.s3_manager.save_result_path}/data/{row['target_data']}"
                target_df = self.s3_manager.load_data_from_path(target_path, data_format="pkl")
                
                pairs.append((feature_df, row["feature_data_label"], target_df, row["target_data_label"]))
                
                # Cache loaded data
                self.loaded_data[row["feature_data_label"]] = {
                    "feature": feature_df,
                    "target": target_df
                }
                
            except Exception as e:
                logger.error(f"Error loading data for task {row['feature_data_label']}: {e}")
                continue
                
        logger.info(f"Successfully loaded {len(pairs)} feature/target pairs")
        return pairs
        
    def prepare_for_batch_eval(self, target_column: str = "Oa"):
        """
        Prepare data in format required by ml.Workflow.batch_eval
        
        Args:
            target_column: Name of target column to extract from target DataFrames
            
        Returns:
            Tuple of (feature_data_list, feature_data_names, target_data, target_name)
        """
        pairs = self.get_feature_target_pairs()
        
        feature_data_list = []
        feature_data_names = []
        target_data = None
        target_name = target_column
        
        for feature_df, feature_name, target_df, _ in pairs:
            feature_data_list.append(feature_df)
            feature_data_names.append(feature_name)
            
            # Use first target DataFrame (should all be the same for a given experiment)
            if target_data is None:
                target_data = target_df
                
                # Verify target column exists
                if target_column not in target_data.columns:
                    logger.warning(f"Target column '{target_column}' not found. Using first column.")
                    target_name = target_data.columns[0]
        
        return feature_data_list, feature_data_names, target_data, target_name


def print_task_summary(task_df: pd.DataFrame):
    """
    Print summary of task list.
    
    Args:
        task_df: DataFrame containing task list
    """
    print("\n" + "="*60)
    print("TASK LIST SUMMARY")
    print("="*60)
    
    print(f"Total tasks: {len(task_df)}")
    
    if len(task_df) > 0:
        print("\nExperiment types:")
        for exp_type, count in task_df['experiment_type'].value_counts().items():
            print(f"  - {exp_type}: {count} tasks")
        
        print("\nModels:")
        for model, count in task_df['model_name'].value_counts().items():
            print(f"  - {model}: {count} tasks")
        
        print("\nSample task rows:")
        print(task_df.head(3).to_string(index=False))
    
    print("="*60)


def main():
    """Main execution function - configuration-based version"""
    
    # Process model configuration
    model_names = process_model_config(MODEL_NAME)
    logger.info(f"Processing {len(model_names)} model(s): {model_names}")
    logger.info(f"Experiment types: {EXPERIMENT_TYPES}")
    
    # Generate task list
    generator = BatchTaskGenerator()
    task_df = generator.generate_task_list(
        experiment_types=EXPERIMENT_TYPES,
        output_csv=OUTPUT_CSV,
        model_names=model_names,
        verify_exists=VERIFY_EXISTS
    )
    
    if not GENERATE_ONLY and len(task_df) > 0:
        # Create example usage
        print("\n" + "="*60)
        print("EXAMPLE USAGE FOR ML WORKFLOW INTEGRATION")
        print("="*60)
        
        example_code = f'''
# Example: Using the generated task list with ml.Workflow

from src.notebooks.ch5_paper.data_eng.create_ml_loader_v1 import BatchLoader
from src.ml.Workflow import batch_eval_standard

# 1. Load the task list
loader = BatchLoader()
loader.load_task_list("{OUTPUT_CSV}")

# 2. Prepare data for batch_eval
feature_data_list, feature_data_names, target_data, target_name = loader.prepare_for_batch_eval()

# 3. Run batch evaluation
results = batch_eval_standard(
    feature_data_list=feature_data_list,
    feature_data_names=feature_data_names,
    target_data=target_data,
    target_name=target_name,
    num_repeats=10,
    test_size=0.2,
    o_random_seed=42,
    n_jobs=-1
)

print(results.head())
'''
        print(example_code)
        
        # Also show how to load specific experiment
        print("\nAlternative: Load specific experiment type")
        alt_code = f'''
# To load only specific experiment types, filter the task list first:
task_df = pd.read_csv("{OUTPUT_CSV}")
filtered_df = task_df[task_df["experiment_type"].isin({EXPERIMENT_TYPES})]
filtered_df.to_csv("filtered_tasks.csv", index=False)

loader = BatchLoader()
loader.load_task_list("filtered_tasks.csv")
# ... continue with batch_eval as above
'''
        print(alt_code)
    
    # Print summary
    print_task_summary(task_df)


if __name__ == "__main__":
    main()
