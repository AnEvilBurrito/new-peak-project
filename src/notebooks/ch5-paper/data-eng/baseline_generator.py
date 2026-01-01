"""
Shared baseline generator for parameter distortion and expression noise experiments.

This module provides a unified way to generate baseline "virtual models" using
the same approach as sy_simple-make-data-v1.py (lognormal perturbation for both
features and parameters).
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


def generate_baseline_virtual_models(
    model_spec,
    model_builder,
    solver,
    n_samples: int = 2000,
    seed: int = 42,
    simulation_params: Dict[str, Any] = None,
    perturbation_type: str = "lognormal",
    perturbation_params: Dict[str, Any] = {"shape": 0.5},
    param_perturbation_type: str = "lognormal",
    param_perturbation_params: Dict[str, Any] = {"shape": 0.5},
    param_seed_offset: int = 1,
    require_all_successful: bool = False,
    resample_size: int = 10,
    max_retries: int = 3,
    outcome_var: str = "Oa"
) -> Dict[str, Any]:
    """
    Generate baseline virtual models using make_data_extended with configurable perturbation.
    
    This creates a population of "virtual models" with natural biological variation
    using the same approach as sy_simple-make-data-v1.py.
    
    Args:
        model_spec: ModelSpecification instance
        model_builder: ModelBuilder instance
        solver: Solver instance
        n_samples: Number of virtual models to generate
        seed: Random seed for reproducibility
        simulation_params: Simulation parameters (default: {'start': 0, 'end': 10000, 'points': 101})
        perturbation_type: Type of perturbation for initial values (e.g., "lognormal", "uniform")
        perturbation_params: Parameters for initial value perturbation
        param_perturbation_type: Type of perturbation for kinetic parameters
        param_perturbation_params: Parameters for kinetic parameter perturbation
        param_seed_offset: Offset to create different seed for parameter perturbation
        require_all_successful: Whether to require all samples to succeed
        resample_size: Number of extra samples to generate for resampling
        max_retries: Maximum number of retry attempts
        outcome_var: Outcome variable to capture from simulations
        
    Returns:
        Dictionary containing:
            - 'features': DataFrame of perturbed initial values (shape: n_samples × n_features)
            - 'targets': DataFrame of baseline targets (ground truth)
            - 'parameters': DataFrame of perturbed kinetic parameters (shape: n_samples × n_params)
            - 'timecourses': Timecourse data from baseline simulations
            - 'metadata': Generation metadata
    """
    from models.utils.data_generation_helpers import make_data_extended
    
    # Get initial values (inactive state variables)
    state_variables = model_builder.get_state_variables()
    initial_values = {k: v for k, v in state_variables.items() if not k.endswith('a')}
    if 'O' in initial_values:
        del initial_values['O']
    
    # Get kinetic parameters
    kinetic_parameters = model_builder.get_parameters()
    
    # Default simulation parameters
    if simulation_params is None:
        simulation_params = {'start': 0, 'end': 10000, 'points': 101}
    
    logger.info(f"Generating baseline virtual models with {n_samples} samples...")
    logger.info(f"  Perturbation type (initial values): {perturbation_type}")
    logger.info(f"  Perturbation type (parameters): {param_perturbation_type}")
    
    # Generate baseline dataset using configurable perturbation
    baseline_data = make_data_extended(
        initial_values=initial_values,
        perturbation_type=perturbation_type,
        perturbation_params=perturbation_params,
        parameter_values=kinetic_parameters,
        param_perturbation_type=param_perturbation_type,
        param_perturbation_params=param_perturbation_params,
        n_samples=n_samples,
        model_spec=model_spec,
        solver=solver,
        simulation_params=simulation_params,
        seed=seed,
        param_seed=seed + param_seed_offset,
        outcome_var=outcome_var,
        capture_all_species=True,
        require_all_successful=require_all_successful,
        resample_size=resample_size,
        max_retries=max_retries
    )
    
    # Extract components
    features = baseline_data['features']
    targets = baseline_data['targets']
    parameters = baseline_data['parameters']
    timecourses = baseline_data['timecourse']
    metadata = baseline_data['metadata']
    
    # Log success rate
    success_rate = metadata.get('success_rate', 1.0)
    failed_indices = metadata.get('failed_indices', [])
    
    if len(failed_indices) > 0:
        logger.warning(f"Baseline generation: {len(failed_indices)}/{n_samples} samples failed")
        logger.info(f"Success rate: {success_rate:.1%}")
        
        # Remove failed samples to ensure clean baseline
        if len(failed_indices) > 0:
            success_mask = pd.Series([True] * n_samples)
            success_mask.iloc[failed_indices] = False
            
            features = features[success_mask].reset_index(drop=True)
            targets = targets[success_mask].reset_index(drop=True)
            if parameters is not None:
                parameters = parameters[success_mask].reset_index(drop=True)
            if timecourses is not None:
                timecourses = timecourses[success_mask].reset_index(drop=True)
            
            logger.info(f"Removed {len(failed_indices)} failed samples, remaining: {len(features)}")
    
    logger.info(f"✅ Generated baseline with {len(features)} virtual models")
    logger.info(f"  Features shape: {features.shape}")
    logger.info(f"  Parameters shape: {parameters.shape if parameters is not None else 'N/A'}")
    logger.info(f"  Targets shape: {targets.shape}")
    
    return {
        'features': features,
        'targets': targets,
        'parameters': parameters,
        'timecourses': timecourses,
        'metadata': metadata
    }


def save_baseline_to_s3(
    baseline_data: Dict[str, Any],
    model_name: str,
    s3_manager,
    upload: bool = True
) -> Dict[str, str]:
    """
    Save baseline virtual models to S3 for reuse.
    
    Args:
        baseline_data: Baseline data dictionary from generate_baseline_virtual_models
        model_name: Name of the model
        s3_manager: S3ConfigManager instance
        upload: Whether to actually upload to S3
        
    Returns:
        Dictionary mapping component names to S3 paths
    """
    if not upload:
        logger.info("Skipping S3 upload for baseline data")
        return {}
    
    gen_path = s3_manager.save_result_path
    folder_name = f"{model_name}_baseline_virtual_models"
    full_path = f"{gen_path}/data/{folder_name}"
    
    file_mappings = {
        'features': f"{full_path}/baseline_features.pkl",
        'targets': f"{full_path}/baseline_targets.pkl",
        'parameters': f"{full_path}/baseline_parameters.pkl",
        'timecourses': f"{full_path}/baseline_timecourses.pkl",
        'metadata': f"{full_path}/baseline_metadata.pkl"
    }
    
    s3_paths = {}
    
    for key, s3_path in file_mappings.items():
        if key in baseline_data and baseline_data[key] is not None:
            s3_manager.save_data_from_path(s3_path, baseline_data[key], data_format="pkl")
            logger.info(f"✅ Saved {key} to S3: {s3_path}")
            s3_paths[key] = s3_path
    
    return s3_paths


def load_baseline_from_s3(
    model_name: str,
    s3_manager
) -> Dict[str, Any]:
    """
    Load baseline virtual models from S3.
    
    Args:
        model_name: Name of the model
        s3_manager: S3ConfigManager instance
        
    Returns:
        Dictionary containing baseline data components
    """
    gen_path = s3_manager.save_result_path
    folder_name = f"{model_name}_baseline_virtual_models"
    full_path = f"{gen_path}/data/{folder_name}"
    
    file_mappings = {
        'features': f"{full_path}/baseline_features.pkl",
        'targets': f"{full_path}/baseline_targets.pkl",
        'parameters': f"{full_path}/baseline_parameters.pkl",
        'timecourses': f"{full_path}/baseline_timecourses.pkl",
        'metadata': f"{full_path}/baseline_metadata.pkl"
    }
    
    baseline_data = {}
    
    for key, s3_path in file_mappings.items():
        try:
            data = s3_manager.load_data_from_path(s3_path, data_format="pkl")
            baseline_data[key] = data
            logger.info(f"✅ Loaded {key} from S3: {s3_path}")
        except Exception as e:
            logger.warning(f"Could not load {key} from S3: {e}")
            baseline_data[key] = None
    
    return baseline_data
