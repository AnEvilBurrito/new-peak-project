"""
Example demonstrating robust data generation with error handling and resampling.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import required modules
from models.Solver.ScipySolver import ScipySolver
from models.Specs.ModelSpecification import ModelSpecification
from models.utils.data_generation_helpers import make_data

def example_robust_data_generation():
    """
    Example showing how to use the robust data generation with error handling.
    """
    print("=" * 70)
    print("Robust Data Generation Example")
    print("=" * 70)
    
    # Create a simple model specification
    class ExampleSpec(ModelSpecification):
        def __init__(self):
            super().__init__()
            self.A_species = ['A', 'B', 'C']
            self.B_species = ['D', 'E']
            self.phosphorylation_reactions = []
    
    # Create a solver (ScipySolver in this case)
    solver = ScipySolver()
    
    # Build a simple model
    from models.ModelBuilder import ModelBuilder
    from models.ReactionArchtype import SimpleReaction
    
    builder = ModelBuilder()
    spec = ExampleSpec()
    
    # Add some simple reactions
    builder.add_reaction(SimpleReaction('A + B -> C', k=1.0))
    builder.add_reaction(SimpleReaction('C -> D + E', k=0.5))
    
    model = builder.build_model(spec)
    solver.load_model(model)
    
    # Set initial values
    initial_values = {'A': 100.0, 'B': 50.0, 'C': 0.0, 'D': 0.0, 'E': 0.0}
    
    print("\n1. Generating data WITHOUT robust error handling (old behavior):")
    print("-" * 60)
    
    try:
        # This would use the old make_data function if called without new parameters
        # For demonstration, we'll show what happens with problematic parameters
        feature_df, target_df = make_data(
            initial_values=initial_values,
            perturbation_type='gaussian',
            perturbation_params={'rsd': 0.5},  # 50% relative standard deviation
            n_samples=50,
            model_spec=spec,
            solver=solver,
            seed=42,
            outcome_var='D',
            simulation_params={'start': 0, 'end': 100, 'points': 101},
            verbose=True
        )
        
        success_rate = (1 - target_df['D'].isna().sum() / len(target_df)) * 100
        print(f"Success rate: {success_rate:.1f}%")
        
    except Exception as e:
        print(f"Error encountered (expected with large perturbations): {e}")
    
    print("\n2. Generating data WITH robust error handling (new behavior):")
    print("-" * 60)
    
    # Now use the new robust features
    feature_df, target_df = make_data(
        initial_values=initial_values,
        perturbation_type='gaussian',
        perturbation_params={'rsd': 0.5},  # Same 50% RSD
        n_samples=50,
        model_spec=spec,
        solver=solver,
        seed=42,
        # New robust parameters
        resample_size=10,      # Generate 10 alternative samples per failed simulation
        max_retries=3,         # Try up to 3 times per failed index
        require_all_successful=False,  # Allow NaN values for persistent failures
        # Standard parameters
        outcome_var='D',
        simulation_params={'start': 0, 'end': 100, 'points': 101},
        verbose=True
    )
    
    # Analyze results
    failed_indices = target_df['D'].isna()
    success_rate = (1 - failed_indices.sum() / len(target_df)) * 100
    
    print(f"\nResults:")
    print(f"  Total samples: {len(target_df)}")
    print(f"  Successful simulations: {len(target_df) - failed_indices.sum()}")
    print(f"  Failed simulations (NaN): {failed_indices.sum()}")
    print(f"  Success rate: {success_rate:.1f}%")
    
    if failed_indices.any():
        print(f"\nFailed indices: {list(target_df[failed_indices].index[:5])}{'...' if failed_indices.sum() > 5 else ''}")
        print("Note: These samples have NaN values in the target dataframe.")
    
    # Visualize successful vs failed samples
    successful_targets = target_df[~failed_indices]['D']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Distribution of successful target values
    axes[0].hist(successful_targets.values, bins=20, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Target Value (D at t=100)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Distribution of Successful Simulations\n({len(successful_targets)} samples)')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Feature values for successful vs failed samples
    if failed_indices.any():
        successful_features = feature_df[~failed_indices].mean()
        failed_features = feature_df[failed_indices].mean()
        
        # Compare feature means
        comparison_df = pd.DataFrame({
            'Successful': successful_features,
            'Failed': failed_features
        })
        
        comparison_df.plot(kind='bar', ax=axes[1])
        axes[1].set_xlabel('Feature')
        axes[1].set_ylabel('Mean Value')
        axes[1].set_title('Mean Feature Values: Successful vs Failed')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'All simulations successful!', 
                    ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
        axes[1].set_title('No Failed Simulations')
    
    plt.tight_layout()
    plt.show()
    
    print("\n3. Example with kinetic parameter perturbation:")
    print("-" * 60)
    
    # Define kinetic parameters to perturb
    kinetic_parameters = {'k1': 1.0, 'k2': 0.5}
    
    feature_df, target_df = make_data(
        initial_values=initial_values,
        perturbation_type='uniform',
        perturbation_params={'min': 0.8, 'max': 1.2},  # ±20% uniform perturbation
        n_samples=30,
        model_spec=spec,
        solver=solver,
        # Kinetic parameter perturbation
        parameter_values=kinetic_parameters,
        param_perturbation_type='lognormal',
        param_perturbation_params={'shape': 0.2},  # Lognormal shape parameter
        param_seed=123,  # Separate seed for parameter generation
        # Robust parameters
        resample_size=8,
        max_retries=2,
        require_all_successful=True,  # Require all samples to succeed
        # Standard parameters
        outcome_var='E',
        simulation_params={'start': 0, 'end': 50, 'points': 51},
        verbose=False
    )
    
    print(f"Generated {len(target_df)} samples with BOTH initial value and kinetic parameter perturbations")
    print(f"All samples successful (require_all_successful=True)")
    
    return feature_df, target_df

if __name__ == "__main__":
    print("This example demonstrates robust data generation with error handling.")
    print("The implementation includes:")
    print("  - Automatic detection of CVODE/solver failures")
    print("  - Batch resampling of both initial values AND kinetic parameters")
    print("  - Configurable retry limits (max_retries)")
    print("  - Configurable resample batch size (resample_size)")
    print("  - Option to require all samples to succeed (require_all_successful)")
    print("  - Progress tracking with informative messages")
    print()
    
    try:
        feature_df, target_df = example_robust_data_generation()
        print("\n✓ Example completed successfully!")
    except ImportError as e:
        print(f"\nImport error: {e}")
        print("Make sure you're running from the project root directory.")
    except Exception as e:
        print(f"\nError running example: {e}")
        import traceback
        traceback.print_exc()
