"""
Test script for the kinetic parameter tuner.
"""
import os
import sys

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.Specs.DegreeInteractionSpec import DegreeInteractionSpec
from models.Specs.Drug import Drug
from models.utils.kinetic_tuner import generate_parameters, KineticParameterTuner

def create_test_model():
    """Create a simple test model using DegreeInteractionSpec."""
    # Create a small network with 2 degrees
    degree_spec = DegreeInteractionSpec(degree_cascades=[1, 2])
    degree_spec.generate_specifications(random_seed=42, feedback_density=0.5)
    
    # Add a drug
    drug = Drug(
        name="D",
        start_time=500.0,
        default_value=10.0,
        regulation=["R1_1"],
        regulation_type=["down"]
    )
    degree_spec.add_drug(drug)
    
    # Generate the model
    model = degree_spec.generate_network(
        network_name="TestKineticTuner",
        mean_range_species=(50, 150),
        rangeScale_params=(0.8, 1.2),
        rangeMultiplier_params=(0.9, 1.1),
        random_seed=42,
        receptor_basal_activation=True
    )
    
    return model

def test_basic_functionality():
    """Test basic functionality of the kinetic tuner."""
    print("Testing KineticParameterTuner...")
    
    # Create test model
    model = create_test_model()
    print(f"Created model: {model.name}")
    print(f"Total states: {len(model.get_state_variables())}")
    print(f"Total parameters: {len(model.get_parameters())}")
    
    # Test parameter generation
    try:
        parameter_dict = generate_parameters(
            model=model,
            active_percentage_range=(0.3, 0.7),
            X_total_multiplier=5.0,
            ki_val=100.0,
            v_max_f_random_range=(5.0, 10.0),
            random_seed=42
        )
        
        print(f"\nGenerated {len(parameter_dict)} parameters")
        
        # Check some basic properties
        if parameter_dict:
            print("✓ Parameter generation successful")
            
            # Check that all values are positive
            negative_values = [(k, v) for k, v in parameter_dict.items() if v <= 0]
            if negative_values:
                print(f"⚠ Warning: Found {len(negative_values)} non-positive parameter values")
                for param, value in negative_values[:3]:
                    print(f"  {param} = {value}")
            else:
                print("✓ All parameter values are positive")
            
            # Show a few example parameters
            print("\nExample parameters:")
            for i, (param, value) in enumerate(list(parameter_dict.items())[:5]):
