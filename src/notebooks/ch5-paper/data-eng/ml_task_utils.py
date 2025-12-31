"""
Shared utilities for ML task CSV generation.

Provides common functionality for generating CSV task lists that can be loaded
by BatchLoader for ML workflow integration. Used by individual experiment scripts
(expression-noise-v1.py, parameter-distortion-v2.py, response-noise-v1.py) to
generate their own CSV files without coupling to a central generator.
"""

import pandas as pd
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Standard CSV column names for ML task lists
STANDARD_CSV_COLUMNS = [
    'feature_data',
    'feature_data_label', 
    'target_data',
    'target_data_label',
    'experiment_type',
    'level',
    'model_name'
]

def validate_task_rows(task_rows: List[Dict[str, Any]]) -> bool:
    """
    Validate that all task rows have required columns.
    
    Args:
        task_rows: List of dictionaries representing task rows
        
    Returns:
        True if all rows are valid, False otherwise
    """
    if not task_rows:
        logger.warning("No task rows to validate")
        return False
    
    for i, row in enumerate(task_rows):
        missing = [col for col in STANDARD_CSV_COLUMNS if col not in row]
        if missing:
            logger.error(f"Row {i} missing required columns: {missing}")
            return False
    
    return True

def save_task_csv(
    task_rows: List[Dict[str, Any]], 
    output_path: str,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Save task list to CSV file.
    
    Args:
        task_rows: List of dictionaries representing task rows
        output_path: Path to output CSV file
        columns: Column names to use (defaults to STANDARD_CSV_COLUMNS)
        
    Returns:
        DataFrame containing the saved task list
        
    Raises:
        ValueError: If task rows fail validation
    """
    if not validate_task_rows(task_rows):
        raise ValueError("Task rows failed validation")
    
    columns = columns or STANDARD_CSV_COLUMNS
    task_df = pd.DataFrame(task_rows, columns=columns)
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    task_df.to_csv(output_path, index=False)
    logger.info(f"✅ Saved task list with {len(task_df)} rows to: {output_path}")
    
    return task_df

def construct_s3_path(base_folder: str, subfolder: str, filename: str) -> str:
    """
    Construct consistent S3 path for experiment files.
    
    Args:
        base_folder: Base folder name (e.g., 'sy_simple_expression_noise_v1')
        subfolder: Subfolder for specific level (e.g., 'noise_0.1')
        filename: Filename (e.g., 'features.pkl')
        
    Returns:
        Constructed S3 path
    """
    return f"{base_folder}/{subfolder}/{filename}"

def generate_task_rows_from_pattern(
    base_folder: str,
    experiment_type: str,
    model_name: str,
    levels: List[Any],
    get_feature_files_func,
    get_target_files_func,
    verify_exists: bool = False,
    s3_manager = None
) -> List[Dict[str, Any]]:
    """
    Generate task rows from experiment pattern functions.
    
    This is a generic function that can be used by individual scripts
    to generate their task rows based on their specific patterns.
    
    Args:
        base_folder: Base folder name for this experiment
        experiment_type: Type of experiment (e.g., 'expression-noise-v1')
        model_name: Name of the model (e.g., 'sy_simple')
        levels: List of levels for this experiment
        get_feature_files_func: Function that returns feature files for a level
        get_target_files_func: Function that returns target files for a level
        verify_exists: If True, verify files exist (requires s3_manager)
        s3_manager: S3ConfigManager instance for verification
        
    Returns:
        List of task row dictionaries
    """
    task_rows = []
    
    for level in levels:
        feature_files = get_feature_files_func(level)
        target_files = get_target_files_func(level)
        
        # Create all combinations of feature and target files
        for feature in feature_files:
            for target in target_files:
                # Verify file exists in S3 if requested
                if verify_exists and s3_manager is not None:
                    try:
                        full_feature_path = f"{s3_manager.save_result_path}/data/{feature['path']}"
                        full_target_path = f"{s3_manager.save_result_path}/data/{target['path']}"
                        
                        # Try to load metadata to check existence
                        s3_manager.load_data_from_path(full_feature_path, data_format="pkl")
                        s3_manager.load_data_from_path(full_target_path, data_format="pkl")
                        
                    except Exception as e:
                        logger.warning(f"File not found or error loading: {feature['path']} or {target['path']}. Skipping.")
                        continue
                
                task_rows.append({
                    "feature_data": feature["path"],
                    "feature_data_label": feature["label"],
                    "target_data": target["path"],
                    "target_data_label": target["label"],
                    "experiment_type": experiment_type,
                    "level": level,
                    "model_name": model_name
                })
    
    return task_rows

class BaseTaskGenerator:
    """
    Base class for task generators in individual scripts.
    
    Individual scripts can inherit from this class and implement
    the pattern-specific methods.
    """
    
    def __init__(self, model_name: str = "sy_simple"):
        self.model_name = model_name
        self.experiment_type = None  # Should be set by subclasses
        
    def get_levels(self) -> List[Any]:
        """Return list of levels for this experiment."""
        raise NotImplementedError("Subclasses must implement get_levels")
        
    def get_base_folder(self) -> str:
        """Return base folder name for this experiment."""
        raise NotImplementedError("Subclasses must implement get_base_folder")
        
    def get_feature_files(self, level: Any) -> List[Dict[str, str]]:
        """Return list of feature file patterns for a given level."""
        raise NotImplementedError("Subclasses must implement get_feature_files")
        
    def get_target_files(self, level: Any) -> List[Dict[str, str]]:
        """Return list of target file patterns for a given level."""
        raise NotImplementedError("Subclasses must implement get_target_files")
    
    def generate_task_list(
        self,
        output_csv: str,
        verify_exists: bool = False,
        s3_manager = None
    ) -> pd.DataFrame:
        """
        Generate CSV task list for this experiment pattern.
        
        Args:
            output_csv: Path to output CSV file
            verify_exists: If True, verify files exist in S3 before adding to list
            s3_manager: S3ConfigManager instance for verification
            
        Returns:
            DataFrame containing the task list
        """
        if not self.experiment_type:
            raise ValueError("experiment_type must be set in subclass")
            
        task_rows = generate_task_rows_from_pattern(
            base_folder=self.get_base_folder(),
            experiment_type=self.experiment_type,
            model_name=self.model_name,
            levels=self.get_levels(),
            get_feature_files_func=self.get_feature_files,
            get_target_files_func=self.get_target_files,
            verify_exists=verify_exists,
            s3_manager=s3_manager
        )
        
        return save_task_csv(task_rows, output_csv)

def read_task_csv(csv_path: str) -> pd.DataFrame:
    """
    Read task list from CSV file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame containing the task list
    """
    task_df = pd.read_csv(csv_path)
    logger.info(f"Loaded task list with {len(task_df)} tasks from: {csv_path}")
    return task_df

def filter_tasks_by_experiment(
    task_df: pd.DataFrame,
    experiment_types: List[str]
) -> pd.DataFrame:
    """
    Filter task DataFrame to include only specified experiment types.
    
    Args:
        task_df: DataFrame containing task list
        experiment_types: List of experiment types to include
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = task_df[task_df['experiment_type'].isin(experiment_types)]
    logger.info(f"Filtered to {len(filtered_df)} tasks for experiment types: {experiment_types}")
    return filtered_df

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
