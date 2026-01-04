"""
Baseline Dynamics Task Generator for ML workflow integration.

Generates CSV task lists for baseline data with dynamic features.
This module provides the BaselineDynamicsTaskGenerator class that follows
the BaseTaskGenerator pattern used by other experiment scripts.

Usage:
    generator = BaselineDynamicsTaskGenerator(model_name="sy_simple")
    task_df = generator.generate_task_list("baseline_tasks.csv", verify_exists=True)
"""

import pandas as pd
import logging
from typing import List, Dict, Any

from ml_task_utils import BaseTaskGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineDynamicsTaskGenerator(BaseTaskGenerator):
    """
    Task generator for baseline dynamics experiment pattern.
    
    This class encapsulates the pattern-specific logic for generating
    CSV task lists for baseline data with dynamic features.
    """
    
    def __init__(self, model_name: str = "sy_simple"):
        super().__init__(model_name)
        self.experiment_type = "baseline-dynamics-v1"
        
    def get_levels(self) -> List[Any]:
        """Return list of levels for this experiment."""
        # Baseline has only one level (0) - no noise/distortion
        return [0]
        
    def get_base_folder(self) -> str:
        """Return base folder name for this experiment."""
        return f"{self.model_name}_baseline_dynamics_v1"
        
    def get_feature_files(self, level: Any) -> List[Dict[str, str]]:
        """
        Get feature files for baseline data.
        
        Returns five individual feature types plus four combined feature types.
        
        Individual feature types:
        1. Original features from baseline virtual models
        2. Dynamic features with outcome variable
        3. Last time points with outcome variable
        4. Dynamic features without outcome variable
        5. Last time points without outcome variable
        
        Combined feature types:
        6. Original + dynamic features without outcome
        7. Original + last time points without outcome
        8. Original + dynamic features with outcome
        9. Original + last time points with outcome
        """
        base_path = self.get_base_folder()
        
        individual_files = [
            {
                "path": f"{base_path}/original_features.pkl",
                "label": "original_features"
            },
            {
                "path": f"{base_path}/dynamic_features_with_outcome.pkl",
                "label": "dynamic_features_with_outcome"
            },
            {
                "path": f"{base_path}/last_time_points_with_outcome.pkl",
                "label": "last_time_points_with_outcome"
            },
            {
                "path": f"{base_path}/dynamic_features_no_outcome.pkl",
                "label": "dynamic_features_no_outcome"
            },
            {
                "path": f"{base_path}/last_time_points_no_outcome.pkl",
                "label": "last_time_points_no_outcome"
            }
        ]
        
        combined_files = [
            {
                "path": f"{base_path}/original_plus_dynamic_no_outcome.pkl",
                "label": "original_plus_dynamic_no_outcome"
            },
            {
                "path": f"{base_path}/original_plus_last_no_outcome.pkl",
                "label": "original_plus_last_no_outcome"
            },
            {
                "path": f"{base_path}/original_plus_dynamic_with_outcome.pkl",
                "label": "original_plus_dynamic_with_outcome"
            },
            {
                "path": f"{base_path}/original_plus_last_with_outcome.pkl",
                "label": "original_plus_last_with_outcome"
            }
        ]
        
        return individual_files + combined_files
        
    def get_target_files(self, level: Any) -> List[Dict[str, str]]:
        """Get target files for baseline data."""
        base_path = self.get_base_folder()
        
        return [
            {
                "path": f"{base_path}/baseline_targets.pkl",
                "label": "baseline_targets"
            }
        ]
