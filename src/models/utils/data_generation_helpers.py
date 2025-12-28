"""
Shared helper functions for data generation utilities.
"""

import warnings
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from models.Solver.Solver import Solver
from models.Utils import ModelSpecification


def validate_simulation_params(simulation_params: Dict[str, Any]) -> None:
    """
    Validate simulation parameters.
    
    Args:
        simulation_params: Dictionary with 'start', 'end', 'points' keys
        
    Raises:
        ValueError: If parameters are invalid
    """
    if 'start' not in simulation_params:
        raise ValueError('Simulation parameters must contain "start" key')
    if 'end' not in simulation_params:
        raise ValueError('Simulation parameters must contain "end" key')
    if 'points' not in simulation_params:
        raise ValueError('Simulation parameters must contain "points" key')
    
    if simulation_params['start'] >= simulation_params['end']:
        raise ValueError('Start time must be less than end time')
    if simulation_params['points'] <= 0:
        raise ValueError('Number of points must be positive')


def extract_species_from_model_spec(
    model_spec,
    include_phospho: bool = True
) -> List[str]:
    """
    Extract species list from model specification.
    
    Args:
        model_spec: ModelSpecification object
        include_phospho: Whether to include phosphorylated versions
        
    Returns:
        List of species names
    """
    species = []
    
    # Try to access species attributes (handling different model spec versions)
    if hasattr(model_spec, 'A_species'):
        species.extend(model_spec.A_species)
    if hasattr(model_spec, 'B_species'):
        species.extend(model_spec.B_species)
    if hasattr(model_spec, 'C_species'):
        species.extend(model_spec.C_species)
    
    if include_phospho:
        # Add phosphorylated versions
        species_with_phospho = []
        for s in species:
            species_with_phospho.append(s)
            species_with_phospho.append(s + 'p')
        return species_with_phospho
    
    return species


def create_default_simulation_params(
    start: float = 0,
    end: float = 500,
    points: int = 100
) -> Dict[str, Any]:
    """
    Create default simulation parameters.
    
    Args:
        start: Start time
        end: End time
        points: Number of points
        
    Returns:
        Dictionary with simulation parameters
    """
    return {
        'start': start,
        'end': end,
        'points': points
    }


def prepare_perturbation_values(
    feature_df_row: pd.Series
) -> Dict[str, float]:
    """
    Prepare perturbation values dictionary from DataFrame row.
    
    Args:
        feature_df_row: Single row from feature DataFrame
        
    Returns:
        Dictionary of perturbation values
    """
    return feature_df_row.to_dict()


def check_parameter_set_compatibility(
    parameter_set: List[Dict[str, float]],
    feature_df: pd.DataFrame
) -> None:
    """
    Check compatibility between parameter set and feature DataFrame.
    
    Args:
        parameter_set: List of parameter dictionaries
        feature_df: Feature DataFrame
        
    Raises:
        ValueError: If incompatible
    """
    if len(parameter_set) != feature_df.shape[0]:
        raise ValueError(
            f'Parameter set length ({len(parameter_set)}) must match '
            f'feature dataframe rows ({feature_df.shape[0]})'
        )


def create_feature_target_pipeline(
    make_feature_data_func,
    make_target_data_func,
    initial_values: Dict[str, float],
    perturbation_params: Dict[str, Any],
    n_samples: int,
    model_spec=None,
    solver=None,
    simulation_params: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a pipeline that generates both feature and target data.
    
    Args:
        make_feature_data_func: Function to generate feature data
        make_target_data_func: Function to generate target data
        initial_values: Dictionary of initial values
        perturbation_params: Parameters for perturbation
        n_samples: Number of samples
        model_spec: Model specification (optional)
        solver: Solver object (optional)
        simulation_params: Simulation parameters (optional)
        seed: Random seed for feature generation
        **kwargs: Additional arguments for target data generation
        
    Returns:
        Tuple of (feature_df, target_df)
    """
    # Generate feature data
    feature_df = make_feature_data_func(
        initial_values=initial_values,
        perturbation_params=perturbation_params,
        n_samples=n_samples,
        seed=seed
    )
    
    # Generate target data if model_spec and solver are provided
    if model_spec is not None and solver is not None:
        target_df, _ = make_target_data_func(
            model_spec=model_spec,
            solver=solver,
            feature_df=feature_df,
            simulation_params=simulation_params,
            **kwargs
        )
    else:
        # Return empty target DataFrame if no model/solver provided
        target_df = pd.DataFrame()
    
    return feature_df, target_df


def make_target_data_with_params(
    model_spec: ModelSpecification,
    solver: Solver,
    feature_df: pd.DataFrame,
    parameter_df: pd.DataFrame = None,
    simulation_params: Dict[str, Any] = None,
    n_cores: int = 1,
    outcome_var: str = 'Cp',
    verbose: bool = False
) -> Tuple[pd.DataFrame, List[np.ndarray]]:
    """
    Generate target data with optional kinetic parameter perturbation.
    
    Args:
        model_spec: ModelSpecification object
        solver: Solver object (ScipySolver or RoadrunnerSolver)
        feature_df: DataFrame of perturbed initial values
        parameter_df: DataFrame of perturbed kinetic parameters (optional)
        simulation_params: Dictionary with 'start', 'end', 'points' keys
        n_cores: Number of cores for parallel processing (-1 for all cores)
        outcome_var: Variable to extract as target
        verbose: Whether to show progress bar
        
    Returns:
        Tuple of (target_df, time_course_data)
    """
    # Set default simulation parameters
    if simulation_params is None:
        simulation_params = {'start': 0, 'end': 500, 'points': 100}
    
    # Validate simulation parameters
    if 'start' not in simulation_params or 'end' not in simulation_params or 'points' not in simulation_params:
        raise ValueError('Simulation parameters must contain "start", "end" and "points" keys')
    
    start = simulation_params['start']
    end = simulation_params['end']
    points = simulation_params['points']
    
    def simulate_perturbation(i: int) -> Tuple[float, np.ndarray]:
        """Simulate a single perturbation with optional kinetic parameters."""
        perturbed_values = feature_df.iloc[i].to_dict()
        
        # Set perturbed initial values into solver
        solver.set_state_values(perturbed_values)
        
        # Set perturbed kinetic parameters if provided
        if parameter_df is not None:
            parameter_values = parameter_df.iloc[i].to_dict()
            solver.set_parameter_values(parameter_values)
        
        # Run simulation
        res = solver.simulate(start, end, points)
        
        # Extract target value and time course
        target_value = res[outcome_var].iloc[-1]
        time_course = res[outcome_var].values
        
        return target_value, time_course
    
    # Use parallel processing if requested
    if n_cores > 1 or n_cores == -1:
        results = Parallel(n_jobs=n_cores)(
            delayed(simulate_perturbation)(i)
            for i in tqdm(range(feature_df.shape[0]), 
                         desc='Simulating perturbations', 
                         disable=not verbose)
        )
        all_targets, time_course_data = zip(*results)
        all_targets = list(all_targets)
        time_course_data = list(time_course_data)
    else:
        # Sequential processing
        all_targets = []
        time_course_data = []
        
        for i in tqdm(range(feature_df.shape[0]), 
                     desc='Simulating perturbations', 
                     disable=not verbose):
            target_value, time_course = simulate_perturbation(i)
            all_targets.append(target_value)
            time_course_data.append(time_course)
    
    # Create target DataFrame
    target_df = pd.DataFrame(all_targets, columns=[outcome_var])
    
    return target_df, time_course_data


def generate_batch_alternatives(base_values: Dict[str, float], 
                               perturbation_type: str,
                               perturbation_params: Dict[str, Any],
                               batch_size: int,
                               base_seed: int,
                               attempt: int) -> pd.DataFrame:
    """
    Generate a batch of alternative values for resampling.
    
    Args:
        base_values: Dictionary of base values to perturb
        perturbation_type: Type of perturbation ('uniform', 'gaussian', 'lognormal', 'lhs')
        perturbation_params: Parameters for perturbation
        batch_size: Number of alternative samples to generate
        base_seed: Base random seed
        attempt: Resampling attempt number (used to generate unique seeds)
        
    Returns:
        DataFrame with batch_size alternative samples
    """
    from .make_feature_data import make_feature_data
    
    # Use a unique seed for each resampling attempt
    alt_seed = base_seed + 1000 * attempt + batch_size
    return make_feature_data(
        initial_values=base_values,
        perturbation_type=perturbation_type,
        perturbation_params=perturbation_params,
        n_samples=batch_size,
        seed=alt_seed
    )


# Unified function that returns both feature and target data
def make_data(
    initial_values: Dict[str, float],
    perturbation_type: str,
    perturbation_params: Dict[str, Any],
    n_samples: int,
    model_spec=None,
    solver=None,
    parameter_values: Optional[Dict[str, float]] = None,
    param_perturbation_type: str = 'none',
    param_perturbation_params: Optional[Dict[str, Any]] = None,
    simulation_params: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    param_seed: Optional[int] = None,
    resample_size: int = 10,
    max_retries: int = 3,
    require_all_successful: bool = False,
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate both feature and target data in one call with robust error handling.
    
    Args:
        initial_values: Dictionary of initial values
        perturbation_type: Type of perturbation ('uniform', 'gaussian', 'lognormal', 'lhs')
        perturbation_params: Parameters for perturbation
        n_samples: Number of samples
        model_spec: Model specification (required for target generation)
        solver: Solver object (required for target generation)
        parameter_values: Dictionary of kinetic parameter values (optional)
        param_perturbation_type: Type of perturbation for kinetic parameters ('none', 'uniform', 'gaussian', 'lognormal', 'lhs')
        param_perturbation_params: Parameters for kinetic parameter perturbation (optional)
        simulation_params: Simulation parameters (optional)
        seed: Random seed for feature generation
        param_seed: Random seed for parameter generation (optional, uses seed if not provided)
        resample_size: Number of alternative samples to generate when a simulation fails (default: 10)
        max_retries: Maximum number of resampling attempts per failed index (default: 3)
        require_all_successful: Whether to require all samples to succeed (default: False)
        **kwargs: Additional arguments for target data generation
        
    Returns:
        Tuple of (feature_df, target_df) where target_df may contain NaN values for failed simulations
        
    Examples:
        >>> X, y = make_data(
        ...     initial_values={'A': 10.0, 'B': 20.0},
        ...     perturbation_type='gaussian',
        ...     perturbation_params={'std': 2.0},
        ...     n_samples=100,
        ...     model_spec=model_spec,
        ...     solver=solver,
        ...     seed=42
        ... )
        
        >>> X, y = make_data(
        ...     initial_values=inactive_state_variables,
        ...     perturbation_type='lognormal',
        ...     perturbation_params={'shape': 0.5},
        ...     parameter_values=kinetic_parameters,
        ...     param_perturbation_type='uniform',
        ...     param_perturbation_params={'min': 0.8, 'max': 1.2},
        ...     n_samples=1000,
        ...     model_spec=degree_spec,
        ...     solver=solver,
        ...     simulation_params={'start': 0, 'end': 10000, 'points': 101},
        ...     seed=42,
        ...     outcome_var='Oa',
        ...     resample_size=10,
        ...     max_retries=3
        ... )
    """
    from .make_feature_data import make_feature_data
    from tqdm import tqdm
    from joblib import Parallel, delayed
    from models.Solver.Solver import Solver
    from models.Utils import ModelSpecification
    
    # Generate feature data (initial value perturbations)
    feature_df = make_feature_data(
        initial_values=initial_values,
        perturbation_type=perturbation_type,
        perturbation_params=perturbation_params,
        n_samples=n_samples,
        seed=seed
    )
    
    # Generate kinetic parameter perturbations if provided
    parameter_df = None
    if parameter_values is not None and param_perturbation_type != 'none':
        parameter_df = make_feature_data(
            initial_values=parameter_values,
            perturbation_type=param_perturbation_type,
            perturbation_params=param_perturbation_params,
            n_samples=n_samples,
            seed=param_seed if param_seed is not None else seed
        )
    
    # Generate target data if model_spec and solver are provided
    if model_spec is not None and solver is not None:
        # Set default simulation parameters
        if simulation_params is None:
            simulation_params = {'start': 0, 'end': 500, 'points': 100}
        
        # Validate simulation parameters
        if 'start' not in simulation_params or 'end' not in simulation_params or 'points' not in simulation_params:
            raise ValueError('Simulation parameters must contain "start", "end" and "points" keys')
        
        start = simulation_params['start']
        end = simulation_params['end']
        points = simulation_params['points']
        outcome_var = kwargs.get('outcome_var', 'Cp')
        n_cores = kwargs.get('n_cores', 1)
        verbose = kwargs.get('verbose', False)
        
        def simulate_perturbation(i: int) -> float:
            """Simulate a single perturbation with optional kinetic parameters."""
            perturbed_values = feature_df.iloc[i].to_dict()
            
            # Set perturbed initial values into solver
            solver.set_state_values(perturbed_values)
            
            # Set perturbed kinetic parameters if provided
            if parameter_df is not None:
                parameter_dict = parameter_df.iloc[i].to_dict()
                solver.set_parameter_values(parameter_dict)
            
            # Run simulation
            res = solver.simulate(start, end, points)
            
            # Extract target value
            return res[outcome_var].iloc[-1]
        
        def simulate_with_values(feature_values: Dict[str, float], param_values: Optional[Dict[str, float]] = None) -> Optional[float]:
            """Simulate with given values and return result or None on failure."""
            try:
                # Set perturbed initial values into solver
                solver.set_state_values(feature_values)
                
                # Set perturbed kinetic parameters if provided
                if param_values is not None:
                    solver.set_parameter_values(param_values)
                
                # Run simulation
                res = solver.simulate(start, end, points)
                
                # Extract target value
                return res[outcome_var].iloc[-1]
            except RuntimeError as e:
                # Check for CVODE errors
                if "CV_TOO_MUCH_WORK" in str(e) or "CVODE" in str(e):
                    return None
                else:
                    raise  # Re-raise unexpected errors
            except Exception as e:
                # Catch other solver errors
                return None
        
        # Sequential processing with error handling and resampling
        all_targets = []
        failed_indices = []
        
        # Create progress bar
        pbar = tqdm(range(feature_df.shape[0]), desc='Simulating perturbations', disable=not verbose)
        
        for i in pbar:
            success = False
            target_value = None
            
            # Get original values
            original_feature_values = feature_df.iloc[i].to_dict()
            original_param_values = parameter_df.iloc[i].to_dict() if parameter_df is not None else None
            
            # Try original values first
            target_value = simulate_with_values(original_feature_values, original_param_values)
            
            if target_value is not None:
                success = True
            else:
                # Try resampling up to max_retries times
                for attempt in range(max_retries):
                    pbar.set_description(f'Simulating perturbations (resampling {i}, attempt {attempt+1}/{max_retries})')
                    
                    # Generate batch alternatives for both feature and parameter values
                    feature_alternatives = generate_batch_alternatives(
                        initial_values, perturbation_type, perturbation_params,
                        resample_size, seed, attempt
                    )
                    
                    if parameter_df is not None:
                        param_alternatives = generate_batch_alternatives(
                            parameter_values, param_perturbation_type, param_perturbation_params,
                            resample_size, param_seed if param_seed is not None else seed, attempt
                        )
                    else:
                        param_alternatives = None
                    
                    # Test each alternative in the batch
                    for j in range(resample_size):
                        alt_feature_values = feature_alternatives.iloc[j].to_dict()
                        alt_param_values = param_alternatives.iloc[j].to_dict() if param_alternatives is not None else None
                        
                        target_value = simulate_with_values(alt_feature_values, alt_param_values)
                        if target_value is not None:
                            success = True
                            # Update the feature and parameter dataframes with successful alternative
                            feature_df.iloc[i] = pd.Series(alt_feature_values)
                            if parameter_df is not None and param_alternatives is not None:
                                parameter_df.iloc[i] = pd.Series(alt_param_values)
                            break
                    
                    if success:
                        break
            
            if success:
                all_targets.append(target_value)
            else:
                all_targets.append(np.nan)
                failed_indices.append(i)
                pbar.set_description(f'Simulating perturbations (failed: {len(failed_indices)})')
        
        # Update progress bar final message
        if failed_indices:
            pbar.set_description(f'Simulating perturbations (completed, {len(failed_indices)} failed)')
        else:
            pbar.set_description('Simulating perturbations (completed)')
        
        # Handle require_all_successful option
        if require_all_successful and failed_indices:
            raise RuntimeError(
                f"Failed to simulate {len(failed_indices)} samples after {max_retries} retries "
                f"with resample_size={resample_size}. Failed indices: {failed_indices[:10]}{'...' if len(failed_indices) > 10 else ''}"
            )
        
        # Create target DataFrame
        target_df = pd.DataFrame(all_targets, columns=[outcome_var])
    else:
        # Return empty target DataFrame if no model/solver provided
        target_df = pd.DataFrame()
    
    return feature_df, target_df


def add_deprecation_warning(
    old_function_name: str,
    new_function_name: str,
    stacklevel: int = 2
):
    """
    Add deprecation warning to a function.
    
    Args:
        old_function_name: Name of deprecated function
        new_function_name: Name of new function to use instead
        stacklevel: Stack level for warning
    """
    warnings.warn(
        f"{old_function_name} is deprecated. Use {new_function_name} instead.",
        DeprecationWarning,
        stacklevel=stacklevel
    )
