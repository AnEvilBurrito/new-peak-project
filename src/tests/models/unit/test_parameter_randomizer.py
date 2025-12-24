"""
Unit tests for parameter_randomizer module.
"""
import pytest
import numpy as np
from models.utils.parameter_randomizer import ParameterRandomizer


class TestParameterRandomizer:
    """Test parameter randomization utilities."""
    
    def test_initialization(self, multi_reaction_model):
        """Test ParameterRandomizer initialization."""
        randomizer = ParameterRandomizer(multi_reaction_model)
        
        # Should have model and default ranges
        assert randomizer.model is multi_reaction_model
        assert len(randomizer.parameter_ranges) > 0
        
        # Check default ranges exist
        assert 'vmax' in randomizer.parameter_ranges
        assert 'km' in randomizer.parameter_ranges
    
    def test_initialization_uncompiled_model(self):
        """Test that uncompiled models raise error."""
        uncompiled_model = pytest.importorskip('models.ModelBuilder').ModelBuilder("uncompiled")
        
        with pytest.raises(ValueError, match="must be pre-compiled"):
            ParameterRandomizer(uncompiled_model)
    
    def test_set_range_for_param_type(self, multi_reaction_model):
        """Test setting parameter ranges."""
        randomizer = ParameterRandomizer(multi_reaction_model)
        
        # Set new range for Vmax
        randomizer.set_range_for_param_type('Vmax', 0.5, 5.0)
        assert randomizer.parameter_ranges['vmax'] == (0.5, 5.0)
        
        # Set range for Km
        randomizer.set_range_for_param_type('Km', 10.0, 100.0)
        assert randomizer.parameter_ranges['km'] == (10.0, 100.0)
        
        # Case-insensitive
        randomizer.set_range_for_param_type('Kc', 0.01, 0.1)
        assert randomizer.parameter_ranges['kc'] == (0.01, 0.1)
    
    def test_set_range_for_param_type_invalid(self, multi_reaction_model):
        """Test invalid range settings."""
        randomizer = ParameterRandomizer(multi_reaction_model)
        
        # min_val >= max_val
        with pytest.raises(ValueError, match="must be less than"):
            randomizer.set_range_for_param_type('Vmax', 5.0, 0.5)
        
        # non-positive min_val
        with pytest.raises(ValueError, match="must be positive"):
            randomizer.set_range_for_param_type('Vmax', 0.0, 5.0)
        
        with pytest.raises(ValueError, match="must be positive"):
            randomizer.set_range_for_param_type('Vmax', -1.0, 5.0)
    
    def test_get_param_type_from_name(self, multi_reaction_model):
        """Test extracting parameter types from names."""
        randomizer = ParameterRandomizer(multi_reaction_model)
        
        # Test common parameter types
        assert randomizer.get_param_type_from_name('Vmax_J0') == 'vmax'
        assert randomizer.get_param_type_from_name('Km_J1') == 'km'
        assert randomizer.get_param_type_from_name('Kc_J2') == 'kc'
        assert randomizer.get_param_type_from_name('Ka0_J0') == 'ka'
        assert randomizer.get_param_type_from_name('Ki0_J1') == 'ki'
        assert randomizer.get_param_type_from_name('Kic0_J2') == 'kic'
        
        # Generic K parameters
        assert randomizer.get_param_type_from_name('Kf_J0') == 'k'
        assert randomizer.get_param_type_from_name('Kd_J1') == 'k'
        
        # Unknown parameters
        assert randomizer.get_param_type_from_name('Unknown_J0') == 'unknown'
        assert randomizer.get_param_type_from_name('J0') == 'unknown'  # No underscore
    
    def test_randomize_all_parameters(self, multi_reaction_model, rng_seed):
        """Test randomizing all parameters."""
        randomizer = ParameterRandomizer(multi_reaction_model)
        
        # Set ranges that are far from original values to ensure change
        # Original values: Vmax_J0=8.0, Vmax_J1=12.0, Vmax_J2=6.0, Km_J0=100.0, Km_J1=80.0, Km_J2=120.0
        randomizer.set_range_for_param_type('Vmax', 1.0, 2.0)  # Much lower than originals (6-12)
        randomizer.set_range_for_param_type('Km', 10.0, 20.0)  # Much lower than originals (80-120)
        
        # Randomize with seed
        randomized_model = randomizer.randomize_all_parameters(seed=rng_seed)
        
        # Should return new model
        assert randomized_model is not multi_reaction_model
        assert randomized_model.name == multi_reaction_model.name
        
        # Check parameters were randomized
        original_params = multi_reaction_model.get_parameters()
        randomized_params = randomized_model.get_parameters()
        
        assert len(original_params) == len(randomized_params)
        
        # Check all parameters are within specified ranges
        for param_name in original_params:
            rand_val = randomized_params[param_name]
            param_type = randomizer.get_param_type_from_name(param_name)
            if param_type == 'vmax':
                assert 1.0 <= rand_val <= 2.0
            elif param_type == 'km':
                assert 10.0 <= rand_val <= 20.0
    
    def test_randomize_all_parameters_reproducible(self, multi_reaction_model, rng_seed):
        """Test that randomization is reproducible with same seed."""
        randomizer1 = ParameterRandomizer(multi_reaction_model)
        randomizer2 = ParameterRandomizer(multi_reaction_model)
        
        # Use same ranges for both
        randomizer1.set_range_for_param_type('Vmax', 1.0, 10.0)
        randomizer1.set_range_for_param_type('Km', 50.0, 150.0)
        randomizer2.set_range_for_param_type('Vmax', 1.0, 10.0)
        randomizer2.set_range_for_param_type('Km', 50.0, 150.0)
        
        # Randomize with same seed
        model1 = randomizer1.randomize_all_parameters(seed=rng_seed)
        model2 = randomizer2.randomize_all_parameters(seed=rng_seed)
        
        # Parameters should be identical
        params1 = model1.get_parameters()
        params2 = model2.get_parameters()
        
        for param_name in params1:
            assert params1[param_name] == pytest.approx(params2[param_name])
    
    def test_randomize_parameters_for_state(self, multi_reaction_model, rng_seed):
        """Test randomizing parameters for specific state."""
        randomizer = ParameterRandomizer(multi_reaction_model)
        
        # Set ranges far from original values
        randomizer.set_range_for_param_type('Vmax', 1.0, 2.0)
        randomizer.set_range_for_param_type('Km', 10.0, 20.0)
        
        # Randomize parameters affecting O
        randomized_model = randomizer.randomize_parameters_for_state('O', seed=rng_seed)
        
        # Get original and randomized parameters
        original_params = multi_reaction_model.get_parameters()
        randomized_params = randomized_model.get_parameters()
        
        # Only parameters affecting O should change
        # O is affected by Km_J2 and Vmax_J2 (reaction 2: O -> Oa)
        for param_name in original_params:
            orig_val = original_params[param_name]
            rand_val = randomized_params[param_name]
            
            if '_J2' in param_name:  # Parameters for reaction 2 (O -> Oa)
                param_type = randomizer.get_param_type_from_name(param_name)
                if param_type == 'vmax':
                    assert 1.0 <= rand_val <= 2.0
                    assert rand_val != pytest.approx(orig_val)  # Should be different given different range
                elif param_type == 'km':
                    assert 10.0 <= rand_val <= 20.0
                    assert rand_val != pytest.approx(orig_val)  # Should be different given different range
            else:
                # Other parameters should remain unchanged
                assert orig_val == rand_val
    
    def test_randomize_parameters_for_state_invalid(self, multi_reaction_model):
        """Test error when randomizing parameters for nonexistent state."""
        randomizer = ParameterRandomizer(multi_reaction_model)
        
        with pytest.raises(ValueError, match="not found"):
            randomizer.randomize_parameters_for_state('Nonexistent')
    
    def test_randomize_parameters_by_role(self, multi_reaction_model, rng_seed):
        """Test randomizing parameters by role."""
        randomizer = ParameterRandomizer(multi_reaction_model)
        
        # Set Vmax range far from original values
        randomizer.set_range_for_param_type('Vmax', 1.0, 2.0)  # Original: 8.0, 12.0, 6.0
        
        # Randomize only Vmax parameters
        randomized_model = randomizer.randomize_parameters_by_role('Vmax', seed=rng_seed)
        
        # Get original and randomized parameters
        original_params = multi_reaction_model.get_parameters()
        randomized_params = randomized_model.get_parameters()
        
        # Only Vmax parameters should change
        for param_name in original_params:
            orig_val = original_params[param_name]
            rand_val = randomized_params[param_name]
            
            if 'Vmax' in param_name:
                assert 1.0 <= rand_val <= 2.0
                assert rand_val != pytest.approx(orig_val)  # Should be different given different range
            else:
                # Other parameters should remain unchanged
                assert orig_val == rand_val
    
    def test_randomize_parameters_by_role_invalid(self, multi_reaction_model):
        """Test randomizing parameters with non-existent role."""
        randomizer = ParameterRandomizer(multi_reaction_model)
        
        with pytest.raises(ValueError, match="No parameters found"):
            randomizer.randomize_parameters_by_role('Nonexistent')
    
    def test_validate_parameter_ranges(self, multi_reaction_model):
        """Test parameter range validation."""
        randomizer = ParameterRandomizer(multi_reaction_model)
        
        # Set tight ranges that likely exclude current values
        randomizer.set_range_for_param_type('Vmax', 20.0, 30.0)  # Current Vmax: 8.0, 12.0, 6.0
        randomizer.set_range_for_param_type('Km', 200.0, 300.0)  # Current Km: 100.0, 80.0, 120.0
        
        validation_results = randomizer.validate_parameter_ranges()
        
        # All parameters should be invalid with these ranges
        for param_name, is_valid in validation_results.items():
            assert not is_valid, f"Parameter {param_name} should be invalid"
    
    def test_get_parameter_statistics(self, multi_reaction_model):
        """Test parameter statistics."""
        randomizer = ParameterRandomizer(multi_reaction_model)
        
        stats = randomizer.get_parameter_statistics()
        
        # Should have stats for vmax and km
        assert 'vmax' in stats
        assert 'km' in stats
        
        # Check vmax stats
        vmax_stats = stats['vmax']
        assert vmax_stats['count'] == 3
        assert vmax_stats['min'] == pytest.approx(6.0)  # Minimum Vmax is 6.0
        assert vmax_stats['max'] == pytest.approx(12.0)  # Maximum Vmax is 12.0
        assert vmax_stats['mean'] == pytest.approx((8.0 + 12.0 + 6.0) / 3)
    
    def test_model_spec4_integration(self, model_spec4_example):
        """Test parameter randomization with real ModelSpec4 model."""
        randomizer = ParameterRandomizer(model_spec4_example)
        
        # First, ensure current parameters are within reasonable ranges
        # Set ranges that should include current parameters
        randomizer.set_range_for_param_type('Vmax', 0.1, 50.0)
        randomizer.set_range_for_param_type('Km', 1.0, 500.0)  # Make range wide enough
        
        # Validate current parameters are within these wide ranges
        validation_results = randomizer.validate_parameter_ranges()
        
        # All current parameters should be valid with these wide ranges
        for param_name, is_valid in validation_results.items():
            assert is_valid, f"Parameter {param_name} should be valid with wide ranges"
        
        # Now randomize with a narrower range
        randomizer.set_range_for_param_type('Vmax', 0.1, 20.0)
        randomizer.set_range_for_param_type('Km', 1.0, 200.0)
        
        randomized_model = randomizer.randomize_all_parameters(seed=42)
        
        # Should return new model
        assert randomized_model is not model_spec4_example
        
        # Check that randomized parameters are within the narrower ranges
        randomized_randomizer = ParameterRandomizer(randomized_model)
        randomized_randomizer.parameter_ranges = randomizer.parameter_ranges
        rand_validation = randomized_randomizer.validate_parameter_ranges()
        
        # All randomized parameters should be valid
        for param_name, is_valid in rand_validation.items():
            assert is_valid, f"Randomized parameter {param_name} should be within range"
