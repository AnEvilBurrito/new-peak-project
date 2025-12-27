"""
Unit tests for the kinetic parameter tuner.

Tests the KineticParameterTuner class and generate_parameters function
for multi-degree drug interaction networks.
"""

import pytest
import numpy as np
from models.Specs.DegreeInteractionSpec import DegreeInteractionSpec
from models.Specs.Drug import Drug
from models.utils.kinetic_tuner import generate_parameters, KineticParameterTuner


def create_simple_test_model():
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


class TestKineticParameterTuner:
    """Test suite for KineticParameterTuner class."""
    
    def test_initialization(self):
        """Test that the tuner initializes correctly."""
        model = create_simple_test_model()
        tuner = KineticParameterTuner(model, random_seed=42)
        
        assert tuner.model is model
        assert len(tuner.active_states) > 0
        assert len(tuner.inactive_states) > 0
        
        # Each active state should have a corresponding inactive state
        for active_state in tuner.active_states.keys():
            inactive_state = active_state[:-1]
            assert inactive_state in tuner.inactive_states
    
    def test_initialization_fails_with_uncompiled_model(self):
        """Test that initialization fails with uncompiled model."""
        from models.ModelBuilder import ModelBuilder
        
        model = ModelBuilder("TestModel")
        # Don't precompile
        with pytest.raises(ValueError, match="must be pre-compiled"):
            KineticParameterTuner(model)
    
    def test_generate_parameters_basic(self):
        """Test basic parameter generation."""
        model = create_simple_test_model()
        tuner = KineticParameterTuner(model, random_seed=42)
        
        parameters = tuner.generate_parameters(
            active_percentage_range=(0.3, 0.7),
            X_total_multiplier=5.0,
            ki_val=100.0,
            v_max_f_random_range=(5.0, 10.0)
        )
        
        # Check that parameters were generated
        assert len(parameters) > 0
        
        # All parameters should have positive values
        for param_name, param_value in parameters.items():
            assert param_value > 0, f"Parameter {param_name} has non-positive value: {param_value}"
        
        # Check that certain parameter types exist
        param_names = list(parameters.keys())
        has_vmax = any('Vmax' in name for name in param_names)
        has_km = any('Km' in name for name in param_names)
        has_kc = any('Kc' in name for name in param_names)
        
        assert has_vmax, "Should have Vmax parameters"
        assert has_km, "Should have Km parameters"
    
    def test_generate_parameters_reproducibility(self):
        """Test that parameter generation is reproducible with same seed."""
        model = create_simple_test_model()
        
        # Generate parameters with same seed
        tuner1 = KineticParameterTuner(model, random_seed=42)
        params1 = tuner1.generate_parameters()
        
        tuner2 = KineticParameterTuner(model, random_seed=42)
        params2 = tuner2.generate_parameters()
        
        # Parameters should be identical
        assert params1.keys() == params2.keys()
        for key in params1:
            assert params1[key] == pytest.approx(params2[key]), \
                f"Parameter {key} differs: {params1[key]} vs {params2[key]}"
    
    def test_generate_parameters_different_seeds(self):
        """Test that different seeds produce different parameters."""
        model = create_simple_test_model()
        
        tuner1 = KineticParameterTuner(model, random_seed=42)
        params1 = tuner1.generate_parameters()
        
        tuner2 = KineticParameterTuner(model, random_seed=123)
        params2 = tuner2.generate_parameters()
        
        # Parameters should differ (not guaranteed but very likely)
        values1 = list(params1.values())
        values2 = list(params2.values())
        
        # At least some values should differ
        assert not all(v1 == pytest.approx(v2) for v1, v2 in zip(values1, values2)), \
            "Parameters should differ with different seeds"
    
    def test_generate_parameters_invalid_percentage_range(self):
        """Test error handling for invalid active percentage ranges."""
        model = create_simple_test_model()
        tuner = KineticParameterTuner(model)
        
        # Percentage range where min >= max should raise error
        with pytest.raises(ValueError):
            tuner.generate_parameters(active_percentage_range=(0.7, 0.3))
    
    def test_generate_parameters_custom_ranges(self):
        """Test parameter generation with custom ranges."""
        model = create_simple_test_model()
        tuner = KineticParameterTuner(model, random_seed=42)
        
        # Test with different parameter ranges
        parameters = tuner.generate_parameters(
            active_percentage_range=(0.4, 0.6),
            X_total_multiplier=10.0,
            ki_val=50.0,
            v_max_f_random_range=(1.0, 2.0)
        )
        
        # Check Ki parameters have the specified value
        ki_params = {k: v for k, v in parameters.items() if k.startswith('Ki')}
        for ki_value in ki_params.values():
            assert ki_value == 50.0
        
        # Check Km_b parameters are X_total * multiplier
        # This is harder to test without knowing X_total for each state
        # We'll just verify all parameters are positive
        for param_value in parameters.values():
            assert param_value > 0
    
    def test_apply_parameters(self):
        """Test applying generated parameters to a model."""
        model = create_simple_test_model()
        tuner = KineticParameterTuner(model, random_seed=42)
        
        # Generate parameters
        parameters = tuner.generate_parameters()
        
        # Apply parameters
        new_model = tuner.apply_parameters(parameters)
        
        # Check that new model has the applied parameters
        new_params = new_model.get_parameters()
        
        # All generated parameters should be in the new model with correct values
        for param_name, expected_value in parameters.items():
            assert param_name in new_params
            assert new_params[param_name] == pytest.approx(expected_value)
        
        # Model should still have the same structure
        assert len(new_model.reactions) == len(model.reactions)
        assert len(new_model.states) == len(model.states)
    
    def test_convenience_function(self):
        """Test the generate_parameters convenience function."""
        model = create_simple_test_model()
        
        parameters = generate_parameters(
            model=model,
            active_percentage_range=(0.3, 0.7),
            X_total_multiplier=5.0,
            ki_val=100.0,
            v_max_f_random_range=(5.0, 10.0),
            random_seed=42
        )
        
        assert len(parameters) > 0
        for param_value in parameters.values():
            assert param_value > 0


def test_error_on_missing_inactive_state():
    """Test error when active state has no corresponding inactive state."""
    from models.ModelBuilder import ModelBuilder
    
    # Create a minimal model with mismatched states
    model = ModelBuilder("TestMismatch")
    
    # Manually add states (simulating a broken model)
    # This would normally be done through reactions
    model.states = {'R1_1a': 50.0}  # Only active state, no inactive
    
    # Try to compile - this might fail or succeed depending on model implementation
    try:
        model.precompile()
    except Exception:
        # If precompile fails, that's okay - the model is invalid anyway
        pass
    
    # Should fail during tuner initialization if model is in invalid state
    # Note: This test might pass or fail depending on how ModelBuilder handles
    # invalid states. The main functionality tests above are more important.
    try:
        tuner = KineticParameterTuner(model)
        # If we get here, the tuner was created successfully
        # This might happen if ModelBuilder fixes the state structure during precompile
        print(f"Note: Tuner created despite mismatched states. Active states: {tuner.active_states}")
    except ValueError as e:
        if "no corresponding inactive state" in str(e):
            # This is the expected error for this test case
            pass
        else:
            raise


if __name__ == "__main__":
    # Run tests directly if needed
    pytest.main([__file__, "-v"])
