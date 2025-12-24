"""
Integration tests for parameter utilities.
"""
import pytest
import numpy as np
from models.utils.parameter_mapper import (
    get_parameter_reaction_map,
    find_parameter_by_role,
    get_parameters_for_state,
    explain_reaction_parameters
)
from models.utils.parameter_randomizer import ParameterRandomizer
from models.utils.initial_condition_randomizer import InitialConditionRandomizer


class TestParameterUtilsIntegration:
    """Integration tests for parameter utilities."""
    
    def test_complete_parameter_analysis_workflow(self, multi_reaction_model):
        """Test complete parameter analysis workflow."""
        # 1. Get parameter-reaction map
        param_map = get_parameter_reaction_map(multi_reaction_model)
        assert len(param_map) > 0
        
        # 2. Find parameters by role
        vmax_params = find_parameter_by_role(multi_reaction_model, 'Vmax')
        km_params = find_parameter_by_role(multi_reaction_model, 'Km')
        
        assert len(vmax_params) > 0
        assert len(km_params) > 0
        
        # 3. Get parameters for specific state
        state_params = get_parameters_for_state(multi_reaction_model, 'O')
        assert len(state_params['all']) > 0
        
        # 4. Explain reaction parameters
        for i in range(len(multi_reaction_model.reactions)):
            explanation = explain_reaction_parameters(multi_reaction_model, i)
            # explanation returns a string, check it contains expected content
            assert isinstance(explanation, str)
            assert 'Parameters:' in explanation or 'parameters' in explanation.lower()
    
    def test_parameter_mapping_and_randomization(self, multi_reaction_model, rng_seed):
        """Test parameter mapping followed by randomization."""
        # Get parameter statistics before randomization
        randomizer = ParameterRandomizer(multi_reaction_model)
        orig_stats = randomizer.get_parameter_statistics()
        
        # Set ranges that guarantee different values
        randomizer.set_range_for_param_type('Vmax', 20.0, 30.0)  # Original: 8.0, 12.0, 6.0
        randomizer.set_range_for_param_type('Km', 200.0, 300.0)  # Original: 100.0, 80.0, 120.0
        
        # Randomize all parameters
        randomized_model = randomizer.randomize_all_parameters(seed=rng_seed)
        
        # Get parameter statistics after randomization
        randomized_randomizer = ParameterRandomizer(randomized_model)
        rand_stats = randomized_randomizer.get_parameter_statistics()
        
        # Statistics should be different
        assert rand_stats['vmax']['mean'] != pytest.approx(orig_stats['vmax']['mean'])
        assert rand_stats['km']['mean'] != pytest.approx(orig_stats['km']['mean'])
        
        # Verify parameters are within specified ranges
        # Create a new randomizer for the randomized model and copy the ranges
        randomized_randomizer = ParameterRandomizer(randomized_model)
        randomized_randomizer.parameter_ranges = randomizer.parameter_ranges.copy()
        validation_results = randomized_randomizer.validate_parameter_ranges()
        for param_name, is_valid in validation_results.items():
            assert is_valid, f"Randomized parameter {param_name} should be within range"
    
    def test_targeted_parameter_randomization(self, multi_reaction_model, rng_seed):
        """Test targeted parameter randomization."""
        # Get original parameters for state O
        from models.utils.parameter_mapper import get_parameters_for_state
        state_params_info = get_parameters_for_state(multi_reaction_model, 'O')
        params_for_o = state_params_info['all']
        
        assert len(params_for_o) > 0
        
        # Create randomizer and set ranges far from original values
        randomizer = ParameterRandomizer(multi_reaction_model)
        randomizer.set_range_for_param_type('Vmax', 20.0, 30.0)  # Original Vmax for O: 6.0
        randomizer.set_range_for_param_type('Km', 200.0, 300.0)  # Original Km for O: 120.0
        
        # Randomize only parameters affecting O
        randomized_model = randomizer.randomize_parameters_for_state('O', seed=rng_seed)
        
        # Get original and randomized parameters
        original_params = multi_reaction_model.get_parameters()
        randomized_params = randomized_model.get_parameters()
        
        # Only parameters affecting O should be in different range
        for param_name in original_params:
            orig_val = original_params[param_name]
            rand_val = randomized_params[param_name]
            
            if param_name in params_for_o:
                param_type = randomizer.get_param_type_from_name(param_name)
                if param_type == 'vmax':
                    assert 20.0 <= rand_val <= 30.0
                    assert rand_val != pytest.approx(orig_val)  # Different range ensures different values
                elif param_type == 'km':
                    assert 200.0 <= rand_val <= 300.0
                    assert rand_val != pytest.approx(orig_val)  # Different range ensures different values
            else:
                # Other parameters should remain unchanged
                assert orig_val == rand_val
    
    def test_combined_parameter_and_initial_condition_randomization(self, multi_reaction_model, rng_seed):
        """Test combined parameter and initial condition randomization."""
        # Create parameter randomizer
        param_randomizer = ParameterRandomizer(multi_reaction_model)
        
        # Set parameter ranges
        param_randomizer.set_range_for_param_type('Vmax', 20.0, 30.0)
        param_randomizer.set_range_for_param_type('Km', 200.0, 300.0)
        
        # Randomize parameters
        param_randomized_model = param_randomizer.randomize_all_parameters(seed=rng_seed)
        
        # Verify parameters were randomized before adding initial condition randomization
        param_randomized_params = param_randomized_model.get_parameters()
        for param_name, value in param_randomized_params.items():
            param_type = param_randomizer.get_param_type_from_name(param_name)
            if param_type == 'vmax':
                assert 20.0 <= value <= 30.0, f"Parameter {param_name}={value} not in range [20, 30]"
            elif param_type == 'km':
                assert 200.0 <= value <= 300.0, f"Parameter {param_name}={value} not in range [200, 300]"
        
        # Create initial condition randomizer on parameter-randomized model
        ic_randomizer = InitialConditionRandomizer(param_randomized_model)
        
        # Set initial condition ranges
        ic_randomizer.set_range_for_state('R1', 50.0, 150.0)
        ic_randomizer.set_range_for_state('O', 200.0, 400.0)
        
        # Randomize initial conditions
        final_model = ic_randomizer.randomize_initial_conditions(seed=rng_seed)
        
        # Get final parameters and states
        final_params = final_model.get_parameters()
        final_states = final_model.get_state_variables()
        
        # Verify parameters are still randomized (initial condition randomization shouldn't affect parameters)
        for param_name, value in final_params.items():
            param_type = param_randomizer.get_param_type_from_name(param_name)
            if param_type == 'vmax':
                assert 20.0 <= value <= 30.0, f"Parameter {param_name}={value} not in range [20, 30] after IC randomization"
            elif param_type == 'km':
                assert 200.0 <= value <= 300.0, f"Parameter {param_name}={value} not in range [200, 300] after IC randomization"
        
        # Verify initial condition randomization
        assert 50.0 <= final_states['R1'] <= 150.0, f"State R1={final_states['R1']} not in range [50, 150]"
        assert 200.0 <= final_states['O'] <= 400.0, f"State O={final_states['O']} not in range [200, 400]"
    
    def test_reproducible_workflow(self, multi_reaction_model, rng_seed):
        """Test that complete workflow is reproducible with same seed."""
        # First run
        randomizer1 = ParameterRandomizer(multi_reaction_model)
        randomizer1.set_range_for_param_type('Vmax', 1.0, 10.0)
        randomizer1.set_range_for_param_type('Km', 50.0, 150.0)
        model1 = randomizer1.randomize_all_parameters(seed=rng_seed)
        
        ic_randomizer1 = InitialConditionRandomizer(model1)
        ic_randomizer1.set_range_for_state('R1', 50.0, 150.0)
        final1 = ic_randomizer1.randomize_initial_conditions(seed=rng_seed)
        
        # Second run with same seed
        randomizer2 = ParameterRandomizer(multi_reaction_model)
        randomizer2.set_range_for_param_type('Vmax', 1.0, 10.0)
        randomizer2.set_range_for_param_type('Km', 50.0, 150.0)
        model2 = randomizer2.randomize_all_parameters(seed=rng_seed)
        
        ic_randomizer2 = InitialConditionRandomizer(model2)
        ic_randomizer2.set_range_for_state('R1', 50.0, 150.0)
        final2 = ic_randomizer2.randomize_initial_conditions(seed=rng_seed)
        
        # Both runs should produce identical results
        params1 = final1.get_parameters()
        params2 = final2.get_parameters()
        
        states1 = final1.get_state_variables()
        states2 = final2.get_state_variables()
        
        for param_name in params1:
            assert params1[param_name] == pytest.approx(params2[param_name])
        
        for state_name in states1:
            assert states1[state_name] == pytest.approx(states2[state_name])
    
    def test_model_spec4_complete_workflow(self, model_spec4_example, rng_seed):
        """Test complete parameter and initial condition workflow with ModelSpec4."""
        # Create parameter randomizer
        param_randomizer = ParameterRandomizer(model_spec4_example)
        
        # Set reasonable parameter ranges that should include current values
        param_randomizer.set_range_for_param_type('Vmax', 0.1, 50.0)
        param_randomizer.set_range_for_param_type('Km', 1.0, 500.0)  # Wide enough for current values
        
        # Verify current parameters are within these ranges
        current_validation = param_randomizer.validate_parameter_ranges()
        for param_name, is_valid in current_validation.items():
            assert is_valid, f"Current parameter {param_name} should be within wide ranges"
        
        # Randomize parameters with narrower, controlled ranges
        param_randomizer.set_range_for_param_type('Vmax', 0.1, 20.0)
        param_randomizer.set_range_for_param_type('Km', 1.0, 200.0)
        
        param_randomized_model = param_randomizer.randomize_all_parameters(seed=rng_seed)
        
        # Verify randomized parameters are within specified ranges
        param_randomized_randomizer = ParameterRandomizer(param_randomized_model)
        param_randomized_randomizer.parameter_ranges = param_randomizer.parameter_ranges
        param_validation = param_randomized_randomizer.validate_parameter_ranges()
        
        for param_name, is_valid in param_validation.items():
            assert is_valid, f"Randomized parameter {param_name} should be within range"
        
        # Now randomize initial conditions
        ic_randomizer = InitialConditionRandomizer(param_randomized_model)
        
        # Set ranges for initial conditions
        states = param_randomized_model.get_state_variables()
        for state_name, current_val in states.items():
            # Skip states with current_val = 0 (activated forms are typically 0 initially)
            if current_val == 0.0:
                continue
            ic_randomizer.set_range_for_state(state_name, current_val * 0.5, current_val * 1.5)
        
        # Randomize initial conditions
        final_model = ic_randomizer.randomize_initial_conditions(seed=rng_seed)
        
        # Verify initial conditions are within ranges
        final_states = final_model.get_state_variables()
        for state_name, value in final_states.items():
            min_val, max_val = ic_randomizer.get_range_for_state(state_name)
            assert min_val <= value <= max_val, \
                f"State {state_name} = {value} should be in range [{min_val}, {max_val}]"
    
    def test_error_handling_integration(self, multi_reaction_model):
        """Test error handling across utilities."""
        # Test invalid state in parameter_mapper
        with pytest.raises(ValueError, match="not found"):
            get_parameters_for_state(multi_reaction_model, 'NonexistentState')
        
        # Test invalid reaction index
        with pytest.raises(IndexError):
            explain_reaction_parameters(multi_reaction_model, 999)
        
        # Test invalid role in find_parameter_by_role
        result = find_parameter_by_role(multi_reaction_model, 'NonexistentRole')
        assert len(result) == 0
        
        # Test invalid state in parameter_randomizer
        randomizer = ParameterRandomizer(multi_reaction_model)
        with pytest.raises(ValueError, match="not found"):
            randomizer.randomize_parameters_for_state('NonexistentState')
        
        # Test invalid role in parameter_randomizer
        with pytest.raises(ValueError, match="No parameters found"):
            randomizer.randomize_parameters_by_role('NonexistentRole')
        
        # Test invalid state in initial_condition_randomizer
        ic_randomizer = InitialConditionRandomizer(multi_reaction_model)
        with pytest.raises(ValueError, match="not found"):
            ic_randomizer.set_range_for_state('NonexistentState', 0, 100)
        
        with pytest.raises(ValueError, match="not found"):
            ic_randomizer.get_range_for_state('NonexistentState')
    
    def test_statistics_and_validation_integration(self, multi_reaction_model):
        """Test statistics and validation integration."""
        # Get parameter statistics
        param_randomizer = ParameterRandomizer(multi_reaction_model)
        param_stats = param_randomizer.get_parameter_statistics()
        
        assert 'vmax' in param_stats
        assert 'km' in param_stats
        
        # Validate parameter ranges with current default ranges
        param_validation = param_randomizer.validate_parameter_ranges()
        for param_name, is_valid in param_validation.items():
            assert is_valid, f"Parameter {param_name} should be within default ranges"
        
        # Get initial condition statistics
        ic_randomizer = InitialConditionRandomizer(multi_reaction_model)
        ic_stats = ic_randomizer.get_initial_condition_statistics()
        
        assert len(ic_stats) > 0
        
        # Validate initial condition ranges
        ic_validation = ic_randomizer.validate_initial_condition_ranges()
        for state_name, is_valid in ic_validation.items():
            assert is_valid, f"State {state_name} should be within default ranges"
