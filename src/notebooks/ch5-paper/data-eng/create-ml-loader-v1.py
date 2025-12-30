"""
ML Batch Loader Generator (v1)

Creates a CSV task list for ML batch evaluation based on data generation patterns.
The CSV can be loaded by a BatchLoader class that interfaces with ml.Workflow.batch_eval functions.

Supports multiple experiment types:
- expression-noise-v1.py
- parameter-distortion-v2.py  
- response-noise-v1.py
"""

import sys
import os
import pandas as pd
import argparse
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import logging

# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "../../..")
sys.path.insert(0, src_dir)

from models.utils.s3_config_manager import S3ConfigManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentPattern:
    """Base class for experiment pattern definitions"""
    
    def __init__(self, name: str, model_name: str = "sy_simple"):
        self.name = name
        self.model_name = model_name
        
    def get_feature_files(self, level: Any) -> List[Dict[str, str]]:
        """Return list of feature file patterns for a given level"""
        raise NotImplementedError
        
    def get_target_files(self, level: Any) -> List[Dict[str, str]]:
        """Return list of target file patterns for a given level"""
        raise NotImplementedError
        
    def get_levels(self) -> List[Any]:
        """Return list of levels for this experiment"""
        raise NotImplementedError
        
    def get_base_folder(self) -> str:
        """Return base folder name for this experiment"""
        raise NotImplementedError


class ExpressionNoisePattern(ExperimentPattern):
    """Pattern for expression-noise-v1.py datasets"""
    
    def __init__(self, model_name: str = "sy_simple"):
        super().__init__("expression-noise-v1", model_name)
        self.noise_levels = [0, 0.1, 0.2, 0.3, 0.5, 1.0]
        
    def get_levels(self) -> List[float]:
        return self.noise_levels
        
    def get_base_folder(self) -> str:
        return f"{self.model_name}_expression_noise_v1"
        
    def get_feature_files(self, noise_level: float) -> List[Dict[str, str]]:
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
        
    def get_target_files(self, noise_level: float) -> List[Dict[str, str]]:
        """Get target files for a given noise level"""
        base_path = f"{self.get_base_folder()}/noise_{noise_level}"
        
        # For expression noise, we have both original_targets and targets
        # Based on user requirements, use original_targets
        return [
            {
                "path": f"{base_path}/original_targets.pkl",
                "label": "original_targets"
            }
        ]


class ParameterDistortionPattern(ExperimentPattern):
    """Pattern for parameter-distortion-v2.py datasets"""
    
    def __init__(self, model_name: str = "sy_simple"):
        super().__init__("parameter-distortion-v2", model_name)
        self.distortion_factors = [0, 1.1, 1.3, 1.5, 2.0, 3.0]
        
    def get_levels(self) -> List[float]:
        return self.distortion_factors
        
    def get_base_folder(self) -> str:
        return f"{self.model_name}_parameter_distortion_v2"
        
    def get_feature_files(self, distortion_factor: float) -> List[Dict[str, str]]:
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
        
    def get_target_files(self, distortion_factor: float) -> List[Dict[str, str]]:
        """Get target files for a given distortion factor"""
        base_path = f"{self.get_base_folder()}/distortion_{distortion_factor}"
        
        return [
            {
                "path": f"{base_path}/targets.pkl",
                "label": "original_targets"
            }
        ]


class ResponseNoisePattern(ExperimentPattern):
    """Pattern for response-noise-v1.py datasets"""
    
    def __init__(self, model_name: str = "sy_simple"):
        super().__init__("response-noise-v1", model_name)
        self.noise_levels = [0, 0.05, 0.1, 0.2, 0.3, 0.5]
        
    def get_levels(self) -> List[float]:
        return self.noise_levels
        
    def get_base_folder(self) -> str:
        return f"{self.model_name}_response_noise_v1"
        
    def get_feature_files(self, noise_level: float) -> List[Dict[str, str]]:
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
        
    def get_target_files(self, noise_level: float) -> List[Dict[str, str]]:
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


class BatchTaskGenerator:
    """Generates CSV task lists for ML batch evaluation"""
    
    def __init__(self, s3_manager: Optional[S3ConfigManager] = None):
        self.s3_manager = s3_manager or S3ConfigManager()
        self.experiment_patterns = {
            "expression-noise-v1": ExpressionNoisePattern,
            "parameter-distortion-v2": ParameterDistortionPattern,
            "response-noise-v1": ResponseNoisePattern
        }
        
    def register_pattern(self, name: str, pattern_class):
        """Register a new experiment pattern for extensibility"""
        self.experiment_patterns[name] = pattern_class
        
    def generate_task_list(
        self, 
        experiment_types: List[str],
        output_csv: str,
        model_name: str = "sy_simple",
        verify_exists: bool = False
    ) -> pd.DataFrame:
        """
        Generate CSV task list for specified experiment types
        
        Args:
            experiment_types: List of experiment type names to include
            output_csv: Path to output CSV file
            model_name: Name of the model (default: 'sy_simple')
            verify_exists: If True, verify files exist in S3 before adding to list
            
        Returns:
            DataFrame containing the task list
        """
        task_rows = []
        
        for exp_type in experiment_types:
            if exp_type not in self.experiment_patterns:
                logger.warning(f"Unknown experiment type: {exp_type}. Skipping.")
                continue
                
            pattern_class = self.experiment_patterns[exp_type]
            pattern = pattern_class(model_name=model_name)
            
            for level in pattern.get_levels():
                feature_files = pattern.get_feature_files(level)
                target_files = pattern.get_target_files(level)
                
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
                
                pairs.append((
                    feature_df,
                    row["feature_data_label"],
                    target_df,
                    row["target_data_label"]
                ))
                
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


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Generate ML batch task lists")
    parser.add_argument("--experiments", nargs="+", required=True,
                       choices=["expression-noise-v1", "parameter-distortion-v2", "response-noise-v1"],
                       help="Experiment types to include")
    parser.add_argument("--output", "-o", required=True, help="Output CSV path")
    parser.add_argument("--model", default="sy_simple", help="Model name")
    parser.add_argument("--verify", action="store_true", 
                       help="Verify files exist in S3 before adding to list")
    parser.add_argument("--generate-only", action="store_true",
                       help="Only generate CSV, don't create loader example")
    
    args = parser.parse_args()
    
    # Generate task list
    generator = BatchTaskGenerator()
    task_df = generator.generate_task_list(
        experiment_types=args.experiments,
        output_csv=args.output,
        model_name=args.model,
        verify_exists=args.verify
    )
    
    if not args.generate_only and len(task_df) > 0:
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
loader.load_task_list("{args.output}")

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
task_df = pd.read_csv("{args.output}")
filtered_df = task_df[task_df["experiment_type"].isin({args.experiments})]
filtered_df.to_csv("filtered_tasks.csv", index=False)

loader = BatchLoader()
loader.load_task_list("filtered_tasks.csv")
# ... continue with batch_eval as above
'''
        print(alt_code)


if __name__ == "__main__":
    main()
