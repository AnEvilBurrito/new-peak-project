"""
Example demonstrating extended data generation with intermediate datasets.
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
try:
    from models.utils.data_generation_helpers import make_data_extended, make_data
    print("✓ Successfully imported data generation functions")
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory.")
    sys.exit(1)

def example_extended_data_generation():
    """
    Example showing how to use the extended data generation functionality.
    """
    print("=" * 70)
    print("Extended Data Generation Example")
    print("=" * 70)
    
    # Create mock objects for demonstration
    class MockSolver:
        def __init__(self):
            self.call_count = 0
        
        def set_state_values(self, values):
            pass
        
        def set_parameter_values(self, values):
            pass
        
        def simulate(self, start, end, points):
            self.call_count += 1
            time_points = np.linspace(start, end, points)
            # Create realistic simulation results
            result = pd.DataFrame({
                'time': time_points,
                'Cp': 100 * np.exp(-0.01 * time_points) + np.random.normal(0, 0.5, points),
                'A': 50 * (1 - np.exp(-0.02 * time_points)) + np.random.normal(0, 0.3, points),
                'B': 30 * np.sin(0.05 * time_points) + np.random.normal(0, 0.2, points),
                'C': 10 * np.cos(0.03 * time_points) + np.random.normal(0, 0.1, points)
            })
            return result
    
    class MockModelSpec:
        def __init__(self):
            self.A_species = ['A', 'B']
            self.B_species = ['C']
    
    mock_solver = MockSolver()
    mock_spec = MockModelSpec()
    
    # Define initial values and kinetic parameters
    initial_values = {'A': 100.0, 'B': 50.0, 'C': 0.0}
    kinetic_parameters = {
        'k1': 1.0,    # Forward rate constant
        'k2': 0.5,    # Reverse rate constant  
        'k3': 0.2,    # Degradation rate
        'Vmax': 10.0  # Maximum velocity
    }
    
    print("\n1. Traditional make_data() call (returns only features and targets):")
    print("-" * 60)
    
    feature_df, target_df = make_data(
        initial_values=initial_values,
        perturbation_type='gaussian',
        perturbation_params={'rsd': 0.1},  # 10% relative standard deviation
        n_samples=10,
        model_spec=mock_spec,
        solver=mock_solver,
        seed=42,
        outcome_var='Cp',
        simulation_params={'start': 0, 'end': 100, 'points': 101},
        verbose=True
    )
    
    print(f"Traditional output:")
    print(f"  Feature dataframe shape: {feature_df.shape}")
    print(f"  Target dataframe shape: {target_df.shape}")
    print(f"  Features columns: {list(feature_df.columns)}")
    print(f"  Target column: {list(target_df.columns)}")
    
    print("\n2. Extended make_data() call with return_details=True:")
    print("-" * 60)
    
    result = make_data(
        initial_values=initial_values,
        perturbation_type='gaussian',
        perturbation_params={'rsd': 0.1},
        n_samples=10,
        model_spec=mock_spec,
        solver=mock_solver,
        parameter_values=kinetic_parameters,
        param_perturbation_type='uniform',
        param_perturbation_params={'min': 0.8, 'max': 1.2},  # ±20% uniform perturbation
        param_seed=123,
        seed=42,
        return_details=True,  # Key parameter for extended output
        outcome_var='Cp',
        simulation_params={'start': 0, 'end': 100, 'points': 101},
        verbose=False
    )
    
    print("Extended output contains:")
    for key, value in result.items():
        if key == 'metadata':
            print(f"  {key}: {list(value.keys())}")
        elif value is None:
            print(f"  {key}: None")
        elif isinstance(value, pd.DataFrame):
            print(f"  {key}: DataFrame with shape {value.shape}")
        elif isinstance(value, list):
            print(f"  {key}: List with {len(value)} elements")
        else:
            print(f"  {key}: {type(value).__name__}")
    
    print("\n3. Using make_data_extended() convenience function (default: capture_all_species=True):")
    print("-" * 60)
    
    extended_result = make_data_extended(
        initial_values=initial_values,
        perturbation_type='lognormal',
        perturbation_params={'shape': 0.2},  # Lognormal shape parameter
        n_samples=8,
        model_spec=mock_spec,
        solver=mock_solver,
        parameter_values=kinetic_parameters,
        param_perturbation_type='gaussian',
        param_perturbation_params={'std': 0.1},  # 10% standard deviation
        seed=42,
        param_seed=456,
        outcome_var='Cp',
        simulation_params={'start': 0, 'end': 50, 'points': 51},
        verbose=False
    )
    
    print("Extended result structure (capture_all_species=True by default):")
    print(f"  Features: {extended_result['features'].shape}")
    print(f"  Targets: {extended_result['targets'].shape}")
    print(f"  Parameters: {extended_result['parameters'].shape if extended_result['parameters'] is not None else 'None'}")
    print(f"  Timecourse data: DataFrame with shape {extended_result['timecourse'].shape}")
    
    # Show species captured in timecourse DataFrame
    timecourse_df = extended_result['timecourse']
    print(f"  Species captured: {list(timecourse_df.columns)}")
    
    print("\n4. Using make_data_extended() with capture_all_species=False:")
    print("-" * 60)
    
    extended_result_single = make_data_extended(
        initial_values=initial_values,
        perturbation_type='lognormal',
        perturbation_params={'shape': 0.2},
        n_samples=8,
        model_spec=mock_spec,
        solver=mock_solver,
        parameter_values=kinetic_parameters,
        param_perturbation_type='gaussian',
        param_perturbation_params={'std': 0.1},
        seed=42,
        param_seed=456,
        outcome_var='Cp',
        capture_all_species=False,  # Only capture outcome variable timecourse
        simulation_params={'start': 0, 'end': 50, 'points': 51},
        verbose=False
    )
    
    print("Extended result structure (capture_all_species=False):")
    print(f"  Features: {extended_result_single['features'].shape}")
    print(f"  Targets: {extended_result_single['targets'].shape}")
    print(f"  Parameters: {extended_result_single['parameters'].shape if extended_result_single['parameters'] is not None else 'None'}")
    print(f"  Timecourse data: List with {len(extended_result_single['timecourse'])} arrays")
    
    metadata = extended_result['metadata']
    print(f"\nMetadata (from capture_all_species=True run):")
    print(f"  Number of samples: {metadata['n_samples']}")
    print(f"  Perturbation type: {metadata['perturbation_type']}")
    print(f"  Capture all species: {metadata['capture_all_species']}")
    print(f"  Success rate: {metadata['success_rate']:.2%}")
    print(f"  Failed indices: {metadata['failed_indices']}")
    print(f"  Resampling used: {metadata['resampling_used']}")
    
    print("\n5. Accessing and using the intermediate datasets:")
    print("-" * 60)
    
    # Extract datasets from result with capture_all_species=True
    features = extended_result['features']
    targets = extended_result['targets']
    parameters = extended_result['parameters']
    timecourse_df = extended_result['timecourse']  # DataFrame with all species
    
    # Extract datasets from result with capture_all_species=False
    timecourse_list = extended_result_single['timecourse']  # List of arrays for outcome variable
    
    print("Example uses of intermediate datasets:")
    print("a) Feature statistics:")
    for col in features.columns:
        print(f"   {col}: mean = {features[col].mean():.2f}, std = {features[col].std():.2f}")
    
    if parameters is not None:
        print("\nb) Parameter statistics:")
        for col in parameters.columns:
            print(f"   {col}: mean = {parameters[col].mean():.2f}, std = {parameters[col].std():.2f}")
    
    print("\nc) Timecourse analysis with capture_all_species=True (DataFrame format):")
    print(f"   Timecourse DataFrame shape: {timecourse_df.shape}")
    print(f"   Species captured: {list(timecourse_df.columns)}")
    
    print("\nd) Timecourse analysis with capture_all_species=False (list format):")
    print(f"   Number of timecourse arrays: {len(timecourse_list)}")
    print(f"   Timecourse length: {len(timecourse_list[0]) if timecourse_list[0] is not None else 'N/A'}")
    
    # Visualize the data if matplotlib is available
    try:
        # Plot example timecourses from both formats
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Timecourses from capture_all_species=False (list format)
        for i in range(min(3, len(timecourse_list))):
            if timecourse_list[i] is not None:
                time_points = np.linspace(0, 50, len(timecourse_list[i]))
                axes[0, 0].plot(time_points, timecourse_list[i], label=f'Sample {i}')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Cp')
        axes[0, 0].set_title('Timecourses (capture_all_species=False) - Outcome Variable Only')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Timecourses from capture_all_species=True (DataFrame format)
        if timecourse_df is not None and not timecourse_df.empty:
            # Plot timecourses for first sample
            sample_idx = 0
            for species in ['A', 'B', 'C']:
                if species in timecourse_df.columns and timecourse_df.iloc[sample_idx][species] is not None:
                    time_points = np.linspace(0, 50, len(timecourse_df.iloc[sample_idx][species]))
                    axes[0, 1].plot(time_points, timecourse_df.iloc[sample_idx][species], label=f'{species}')
            axes[0, 1].set_xlabel('Time')
            axes[0, 1].set_ylabel('Concentration')
            axes[0, 1].set_title('Timecourses (capture_all_species=True) - All Species')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Feature vs Target correlation
        axes[1, 0].scatter(features['A'], targets['Cp'], alpha=0.6)
        axes[1, 0].set_xlabel('Initial A')
        axes[1, 0].set_ylabel('Final Cp')
        axes[1, 0].set_title('Feature-Target Correlation')
        axes[1, 0].grid(True, alpha=0.3)
        
        if parameters is not None:
            # Plot 4: Parameter vs Target correlation
            axes[1, 1].scatter(parameters['k1'], targets['Cp'], alpha=0.6)
            axes[1, 1].set_xlabel('Parameter k1')
            axes[1, 1].set_ylabel('Final Cp')
            axes[1, 1].set_title('Parameter-Target Correlation')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"\nNote: Could not display plots: {e}")
        print("Make sure matplotlib is installed for visualization.")
    
    print("\n6. Practical applications of intermediate data:")
    print("-" * 60)
    print("The extended data generation enables:")
    print("  • Sensitivity analysis using parameter perturbations")
    print("  • Timecourse analysis for dynamic behavior studies")
    print("  • Correlation studies between features, parameters, and outcomes")
    print("  • Debugging failed simulations with detailed metadata")
    print("  • Resampling analysis to understand problematic parameter regions")
    
    return extended_result

def main():
    """Main example execution."""
    print("This example demonstrates extended data generation with intermediate datasets.")
    print("\nKey features:")
    print("  1. Traditional make_data() returns only (features, targets)")
    print("  2. make_data(return_details=True) returns comprehensive dictionary")
    print("  3. make_data_extended() convenience function for extended output")
    print("  4. Captures kinetic parameters, timecourse data, and metadata")
    print("  5. Default behavior: capture_all_species=True (DataFrame format)")
    print("  6. Optional: capture_all_species=False for backward compatibility (list format)")
    print()
    
    try:
        result = example_extended_data_generation()
        print("\n" + "=" * 70)
        print("✓ Example completed successfully!")
        print("=" * 70)
        
        # Demonstrate backward compatibility
        print("\nBackward compatibility demonstration:")
        print("-" * 40)
        print("Existing code continues to work unchanged:")
        print(">>> feature_df, target_df = make_data(...)")
        print("\nNew code can use extended features:")
        print(">>> result = make_data_extended(...)")
        print(">>> # Returns DataFrame for timecourses (capture_all_species=True by default)")
        print("\n>>> result = make_data_extended(..., capture_all_species=False)")
        print(">>> # Returns list for backward compatibility")
        print("\n>>> result = make_data(..., return_details=True, capture_all_species=True)")
        print(">>> # Direct access to extended data structure")
        
    except Exception as e:
        print(f"\nError running example: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
