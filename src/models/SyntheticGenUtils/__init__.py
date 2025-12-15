"""
Utility functions for SyntheticGen module refactoring.

This package contains reusable utility functions extracted from SyntheticGen.py
to eliminate code duplication while maintaining backward compatibility.
"""

# Version information
__version__ = "1.0.0"
__author__ = "Auto-generated refactoring"

# Import key functions for easy access
from .ValidationUtils import (
    validate_simulation_params,
    validate_perturbation_params,
    validate_perturbation_type
)

from .ParallelUtils import (
    run_parallel_simulation,
    run_sequential_simulation,
    handle_simulation_error
)

from .PerturbationUtils import (
    generate_uniform_perturbation,
    generate_gaussian_perturbation,
    generate_lhs_perturbation,
    get_all_species
)

from .SimulationUtils import (
    compile_solver,
    set_simulation_values,
    extract_simulation_results
)

from .DataProcessingUtils import (
    create_feature_dataframe,
    create_target_dataframe,
    process_time_course_data
)

# Expose main utility categories
__all__ = [
    # Validation
    'validate_simulation_params',
    'validate_perturbation_params',
    'validate_perturbation_type',
    
    # Parallel processing
    'run_parallel_simulation', 
    'run_sequential_simulation',
    'handle_simulation_error',
    
    # Perturbation generation
    'generate_uniform_perturbation',
    'generate_gaussian_perturbation', 
    'generate_lhs_perturbation',
    'get_all_species',
    
    # Simulation
    'compile_solver',
    'set_simulation_values',
    'extract_simulation_results',
    
    # Data processing
    'create_feature_dataframe',
    'create_target_dataframe',
    'process_time_course_data'
]
