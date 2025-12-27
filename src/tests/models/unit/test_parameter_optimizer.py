"""
Unit tests for the parameter optimizer.

Tests the ParameterOptimizer class and optimize_parameters function
for multi-degree drug interaction networks with pre/post drug targets.
"""

import pytest
import numpy as np
from models.Specs.DegreeInteractionSpec import DegreeInteractionSpec
from models.Specs.Drug import Drug
from models.optimisation import ParameterOptimizer, optimize_parameters


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
        network_name="TestParameterOptimizer",
        mean_range_species=(50, 150),
        rangeScale_params=(0.8, 1.2),
        rangeMultiplier_params=(0.9, 1.1),
        random_seed=42,
        receptor_basal_activation=True
    )
    
    return model


class TestParameterOptimizer:
    """Test suite for ParameterOptimizer class."""
    
    def test_initialization(self):
        """Test that the optimizer initializes correctly."""
        model = create_simple_test_model()
        optimizer = ParameterOptimizer(model, random_seed=42)
        
        assert optimizer.model is model
        assert len(optimizer.all_states) > 0
        assert len(optimizer.active_states) > 0
        assert len(optimizer.all_params) > 0
        
        # Each active state should have a corresponding inactive state
        for active_state in optimizer.active_states.keys():
            inactive_state = active_state[:-1]
            assert inactive_state in optimizer.inactive_states
        
        # Should have drug info
        assert optimizer.drug_info is not None
        assert 'start_time' in optimizer.drug_info
    
    def test_initialization_fails_with_uncompiled_model(self):
        """Test that initialization fails with uncompiled model."""
        from models.ModelBuilder import ModelBuilder
        
        model = ModelBuilder("TestModel")
        # Don't precompile
        with pytest.raises(ValueError, match="must be pre-compiled"):
            ParameterOptimizer(model)
    
    def test_parameter_bounds(self):
        """Test that parameter bounds are estimated correctly."""
        model = create_simple_test_model()
        optimizer = ParameterOptimizer(model)
        
        bounds = optimizer.param_bounds
        assert len(bounds.lb) == len(optimizer.all_params)
        assert len(bounds.ub) == len(optimizer.all_params)
        
        # All lower bounds should be less than upper bounds
        for lb, ub in zip(bounds.lb, bounds.ub):
            assert lb < ub
        
        # Check specific parameter type bounds
        for param_name in optimizer.all_params:
            idx = optimizer.all_params.index(param_name)
            lb = bounds.lb[idx]
            ub = bounds.ub[idx]
            
            if 'vmax' in param_name.lower() or 'kc' in param_name.lower():
                assert lb == 0.01
                assert ub == 1000.0
            elif 'km' in param_name.lower():
                assert lb == 0.1
                assert ub == 10000.0
            elif 'ki' in param_name.lower():
                assert lb == 0.01
                assert ub == 1000.0
    
    def test_identify_degree1_species(self):
        """Test identification of degree 1 species."""
        model = create_simple_test_model()
        optimizer = ParameterOptimizer(model)
        
        degree1_species = optimizer._identify_degree1_species()
        
        # Should include R1_1, R1_1a, I1_1, I1_1a, O, Oa
        expected = ['R1_1', 'R1_1a', 'I1_1', 'I1_1a', 'O', 'Oa']
        for species in expected:
            assert species in degree1_species
        
        # Should not include degree 2 species
        assert 'R2_1' not in degree1_species
        assert 'R2_2' not in degree1_species
    
    def test_generate_targets(self):
        """Test generation of pre-drug and post-drug targets."""
        model = create_simple_test_model()
        optimizer = ParameterOptimizer(model, random_seed=42)
        
        pre_targets, post_targets = optimizer.generate_targets(
            pre_drug_range=(0.5, 0.7),
            post_drug_range=(0.2, 0.5)
        )
        
        # Check pre-drug targets
        assert len(pre_targets) > 0
        for species, target in pre_targets.items():
            assert species.endswith('a')  # Only active species
            assert target > 0
        
        # Check post-drug targets
        assert len(post_targets) > 0
        for species, target in post_targets.items():
            assert species.endswith('a')  # Only active species
            assert target > 0
            # Should be degree 1 species
            assert species in optimizer._identify_degree1_species()
        
        # Post-drug targets should be lower than pre-drug targets for same species
        for species in post_targets:
            if species in pre_targets:
                # Convert to fractions for comparison
                inactive_state = species[:-1]
                total = optimizer.all_states[inactive_state] + optimizer.all_states[species]
                pre_fraction = pre_targets[species] / total
                post_fraction = post_targets[species] / total
                
                assert post_fraction < pre_fraction, \
                    f"Post-drug fraction ({post_fraction}) should be lower than pre-drug ({pre_fraction})"
                assert 0.2 <= post_fraction <= 0.5, \
                    f"Post-drug fraction ({post_fraction}) should be in range [0.2, 0.5]"
                assert 0.5 <= pre_fraction <= 0.7, \
                    f"Pre-drug fraction ({pre_fraction}) should be in range [0.5, 0.7]"
    
    def test_objective_function(self):
        """Test that objective function returns a scalar value."""
        model = create_simple_test_model()
        optimizer = ParameterOptimizer(model, random_seed=42)
        
        # Generate targets first
        optimizer.generate_targets()
        
        # Get initial guess
        initial_guess = optimizer._get_initial_guess()
        
        # Evaluate objective function
        error = optimizer._objective_function(initial_guess)
        
        # Should return a non-negative float
        assert isinstance(error, float)
        assert error >= 0
    
    def test_get_initial_guess(self):
        """Test getting initial guess from KineticParameterTuner."""
        model = create_simple_test_model()
        optimizer = ParameterOptimizer(model, random_seed=42)
        
        initial_guess = optimizer._get_initial_guess()
        
        # Should return numpy array with correct length
        assert isinstance(initial_guess, np.ndarray)
        assert len(initial_guess) == len(optimizer.all_params)
        
        # All values should be within bounds
        bounds = optimizer.param_bounds
        for i, value in enumerate(initial_guess):
            assert bounds.lb[i] <= value <= bounds.ub[i]
    
    def test_optimize_basic(self):
        """Test basic optimization (with reduced iterations for speed)."""
        model = create_simple_test_model()
        optimizer = ParameterOptimizer(model, random_seed=42)
        
        # Run optimization with few iterations for speed
        result = optimizer.optimize(
            pre_drug_range=(0.5, 0.7),
            post_drug_range=(0.2, 0.5),
            max_iterations=5,  # Small number for speed
            tolerance=1e-1     # Loose tolerance for speed
        )
        
        # Check result structure
        assert 'success' in result
        assert 'optimized_parameters' in result
        assert 'initial_parameters' in result
        assert 'final_error' in result
        assert 'pre_drug_targets' in result
        assert 'post_drug_targets' in result
        
        # Optimized parameters should have correct structure
        optimized_params = result['optimized_parameters']
        assert len(optimized_params) == len(optimizer.all_params)
        
        for param_name, param_value in optimized_params.items():
            assert param_name in optimizer.all_params
            assert isinstance(param_value, float)
        
        # Final error should be non-negative
        assert result['final_error'] >= 0
    
    def test_optimize_without_initial_guess(self):
        """Test optimization without using KineticParameterTuner initial guess."""
        model = create_simple_test_model()
        optimizer = ParameterOptimizer(model, random_seed=42)
        
        # Run optimization without initial guess
        result = optimizer.optimize(
            use_initial_guess=False,
            max_iterations=3,  # Small number for speed
            tolerance=1e-1     # Loose tolerance for speed
        )
        
        # Should still complete
        assert 'success' in result
        assert 'optimized_parameters' in result
        assert len(result['optimized_parameters']) == len(optimizer.all_params)
    
    def test_convenience_function(self):
        """Test the optimize_parameters convenience function."""
        model = create_simple_test_model()
        
        result = optimize_parameters(
            model=model,
            pre_drug_range=(0.5, 0.7),
            post_drug_range=(0.2, 0.5),
            random_seed=42
        )
        
        # Should return result dictionary
        assert isinstance(result, dict)
        assert 'optimized_parameters' in result
        assert 'final_error' in result
        
        # Optimized parameters should be valid
        optimized_params = result['optimized_parameters']
        assert len(optimized_params) > 0
        
        for param_value in optimized_params.values():
            assert isinstance(param_value, float)
            assert param_value > 0
    
    def test_reproducibility(self):
        """Test that optimization is reproducible with same seed."""
        model = create_simple_test_model()
        
        # Run optimization twice with same seed
        result1 = optimize_parameters(model, random_seed=42)
        result2 = optimize_parameters(model, random_seed=42)
        
        # Optimized parameters should be identical
        params1 = result1['optimized_parameters']
        params2 = result2['optimized_parameters']
        
        assert params1.keys() == params2.keys()
        for key in params1:
            assert params1[key] == pytest.approx(params2[key]), \
                f"Parameter {key} differs: {params1[key]} vs {params2[key]}"
        
        # Final errors should be identical
        assert result1['final_error'] == pytest.approx(result2['final_error'])
    
    def test_different_pre_post_ranges(self):
        """Test optimization with different pre/post drug ranges."""
        model = create_simple_test_model()
        optimizer = ParameterOptimizer(model, random_seed=42)
        
        # Test with narrow ranges
        result = optimizer.optimize(
            pre_drug_range=(0.6, 0.65),
            post_drug_range=(0.25, 0.35),
            max_iterations=3,
            tolerance=1e-1
        )
        
        # Should complete successfully
        assert 'success' in result
        assert 'optimized_parameters' in result
        
        # Check that targets were generated with correct ranges
        pre_targets = result['pre_drug_targets']
        post_targets = result['post_drug_targets']
        
        for species, target in pre_targets.items():
            inactive_state = species[:-1]
            total = optimizer.all_states[inactive_state] + optimizer.all_states[species]
            fraction = target / total
            assert 0.6 <= fraction <= 0.65
        
        for species, target in post_targets.items():
            inactive_state = species[:-1]
            total = optimizer.all_states[inactive_state] + optimizer.all_states[species]
            fraction = target / total
            assert 0.25 <= fraction <= 0.35


def test_error_handling():
    """Test error handling in objective function."""
    from models.ModelBuilder import ModelBuilder
    
    # Create a model without drug
    model = ModelBuilder("TestNoDrug")
    # Add minimal structure to pass pre-compilation
    model.precompile()
    
    # Should raise error when trying to optimize without drug
    optimizer = ParameterOptimizer(model)
    with pytest.raises(RuntimeError, match="Targets not set"):
        # Need to generate targets first
        optimizer._objective_function(np.zeros(len(optimizer.all_params)))


if __name__ == "__main__":
    # Run tests directly if needed
    pytest.main([__file__, "-v"])
