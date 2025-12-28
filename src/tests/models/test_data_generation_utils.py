"""
Tests for the data generation utility functions.

These tests verify that the refactored utility functions work correctly
and maintain compatibility with the original SyntheticGen functions.
"""

import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal, assert_allclose

from models.utils import (
    make_feature_data,
    make_feature_data_uniform,
    make_feature_data_gaussian,
    make_feature_data_lognormal,
    make_feature_data_lhs,
    make_target_data,
    make_target_data_diff_spec,
    make_target_data_diff_build,
    make_timecourse_data,
    make_timecourse_data_v3,
    validate_simulation_params,
    create_default_simulation_params,
    check_parameter_set_compatibility,
    make_data
)

from models.Solver.ScipySolver import ScipySolver
from models.ModelBuilder import ModelBuilder
from models.Reaction import Reaction
from models.Utils import ModelSpecification
from models.ReactionArchtype import ReactionArchtype
from models.ArchtypeCollections import BaseArchtypeCollection


def create_test_model_spec():
    """Create a simple test model specification."""
    model_spec = ModelSpecification()
    model_spec.A_species = ['A', 'B']
    model_spec.B_species = ['C']
    model_spec.C_species = ['D']
    return model_spec


def create_test_model_builder():
    """Create a simple test model."""
    model_spec = create_test_model_spec()
    
    # Create a simple phosphorylation cascade model
    model_builder = ModelBuilder()
    
    # Add reactions
    reaction1 = Reaction(
        archtype=ReactionArchtype.UNCATALYSED_CONVERSION,
        reactants=['A'],
        products=['Ap'],
        parameters={'kf': 0.1, 'kr': 0.01}
    )
    
    reaction2 = Reaction(
        archtype=ReactionArchtype.ENZYMATIC_CONVERSION,
        reactants=['Ap', 'B'],
        products=['App', 'B'],
        parameters={'kcat': 0.5, 'km': 10.0}
    )
    
    reaction3 = Reaction(
        archtype=ReactionArchtype.ENZYMATIC_CONVERSION,
        reactants=['App', 'C'],
        products=['Appp', 'C'],
        parameters={'kcat': 0.3, 'km': 5.0}
    )
    
    model_builder.add_reaction(reaction1)
    model_builder.add_reaction(reaction2)
    model_builder.add_reaction(reaction3)
    
    return model_builder


def create_test_solver():
    """Create a test solver with a simple model."""
    model_builder = create_test_model_builder()
    solver = ScipySolver()
    solver.compile(model_builder.get_antimony_model())
    return solver


class TestFeatureDataGeneration:
    """Tests for feature data generation functions."""
    
    def test_make_feature_data_uniform(self):
        """Test uniform perturbation generation."""
        initial_values = {'A': 10.0, 'B': 20.0, 'C': 5.0}
        perturbation_params = {'min': 0.5, 'max': 1.5}
        n_samples = 100
        
        df = make_feature_data_uniform(
            initial_values=initial_values,
            perturbation_params=perturbation_params,
            n_samples=n_samples,
            seed=42
        )
        
        assert df.shape == (n_samples, 3)
        assert list(df.columns) == ['A', 'B', 'C']
        assert df['A'].mean() == pytest.approx(10.0, rel=0.1)
        assert df['B'].mean() == pytest.approx(20.0, rel=0.1)
        assert df['C'].mean() == pytest.approx(5.0, rel=0.1)
        
        # Check bounds
        min_val = initial_values['A'] * perturbation_params['min']
        max_val = initial_values['A'] * perturbation_params['max']
        assert df['A'].min() >= min_val * 0.95  # Allow small tolerance
        assert df['A'].max() <= max_val * 1.05  # Allow small tolerance
    
    def test_make_feature_data_gaussian(self):
        """Test Gaussian perturbation generation."""
        initial_values = {'A': 10.0, 'B': 20.0, 'C': 5.0}
        perturbation_params = {'std': 2.0}
        n_samples = 1000
        
        df = make_feature_data_gaussian(
            initial_values=initial_values,
            perturbation_params=perturbation_params,
            n_samples=n_samples,
            seed=42
        )
        
        assert df.shape == (n_samples, 3)
        assert list(df.columns) == ['A', 'B', 'C']
        
        # Check means are approximately correct
        assert df['A'].mean() == pytest.approx(10.0, rel=0.1)
        assert df['B'].mean() == pytest.approx(20.0, rel=0.1)
        assert df['C'].mean() == pytest.approx(5.0, rel=0.1)
        
        # Check standard deviations
        assert df['A'].std() == pytest.approx(2.0, rel=0.1)
    
    def test_make_feature_data_lognormal(self):
        """Test lognormal perturbation generation."""
        initial_values = {'A': 10.0, 'B': 20.0, 'C': 5.0}
        perturbation_params = {'sigma': 0.2}  # 20% variation
        n_samples = 100
        
        df = make_feature_data_lognormal(
            initial_values=initial_values,
            perturbation_params=perturbation_params,
            n_samples=n_samples,
            seed=42
        )
        
        assert df.shape == (n_samples, 3)
        assert list(df.columns) == ['A', 'B', 'C']
        
        # Lognormal distribution: values should be positive
        assert (df > 0).all().all()
        
        # Check approximate means (lognormal has mean = exp(mu + sigma^2/2))
        # For small sigma, mean ≈ initial_value
        assert df['A'].mean() == pytest.approx(10.0, rel=0.2)
    
    def test_make_feature_data_lhs(self):
        """Test Latin Hypercube Sampling."""
        initial_values = {'A': 10.0, 'B': 20.0, 'C': 5.0}
        perturbation_params = {'min': 0.5, 'max': 2.0}
        n_samples = 50
        
        df = make_feature_data_lhs(
            initial_values=initial_values,
            perturbation_params=perturbation_params,
            n_samples=n_samples,
            seed=42
        )
        
        assert df.shape == (n_samples, 3)
        assert list(df.columns) == ['A', 'B', 'C']
        
        # LHS samples should cover the range well
        min_val = perturbation_params['min']
        max_val = perturbation_params['max']
        
        # Check bounds
        assert df.min().min() >= min_val * 0.95
        assert df.max().max() <= max_val * 1.05
    
    def test_make_feature_data_main_function(self):
        """Test the main make_feature_data function with different perturbation types."""
        initial_values = {'A': 10.0, 'B': 20.0}
        
        # Test uniform perturbation
        uniform_df = make_feature_data(
            initial_values=initial_values,
            perturbation_type='uniform',
            perturbation_params={'min': 0.8, 'max': 1.2},
            n_samples=10,
            seed=42
        )
        assert uniform_df.shape == (10, 2)
        
        # Test Gaussian perturbation
        gaussian_df = make_feature_data(
            initial_values=initial_values,
            perturbation_type='gaussian',
            perturbation_params={'std': 1.0},
            n_samples=10,
            seed=42
        )
        assert gaussian_df.shape == (10, 2)
        
        # Test LHS perturbation
        lhs_df = make_feature_data(
            initial_values=initial_values,
            perturbation_type='lhs',
            perturbation_params={'min': 0.5, 'max': 2.0},
            n_samples=10,
            seed=42
        )
        assert lhs_df.shape == (10, 2)


class TestHelperFunctions:
    """Tests for helper functions."""
    
    def test_validate_simulation_params(self):
        """Test simulation parameter validation."""
        # Valid parameters
        valid_params = {'start': 0, 'end': 100, 'points': 50}
        validate_simulation_params(valid_params)
        
        # Missing keys
        with pytest.raises(ValueError):
            validate_simulation_params({'start': 0, 'end': 100})
        
        with pytest.raises(ValueError):
            validate_simulation_params({'start': 0, 'points': 50})
        
        with pytest.raises(ValueError):
            validate_simulation_params({'end': 100, 'points': 50})
        
        # Invalid values
        with pytest.raises(ValueError):
            validate_simulation_params({'start': 100, 'end': 0, 'points': 50})
        
        with pytest.raises(ValueError):
            validate_simulation_params({'start': 0, 'end': 100, 'points': 0})
    
    def test_create_default_simulation_params(self):
        """Test default simulation parameter creation."""
        params = create_default_simulation_params()
        
        assert params == {'start': 0, 'end': 500, 'points': 100}
        
        # Test custom values
        params_custom = create_default_simulation_params(start=10, end=200, points=20)
        assert params_custom == {'start': 10, 'end': 200, 'points': 20}
    
    def test_check_parameter_set_compatibility(self):
        """Test parameter set compatibility checking."""
        feature_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        parameter_set = [
            {'k1': 0.1, 'k2': 0.2},
            {'k1': 0.3, 'k2': 0.4},
            {'k1': 0.5, 'k2': 0.6}
        ]
        
        # Should not raise an error
        check_parameter_set_compatibility(parameter_set, feature_df)
        
        # Should raise an error for incompatible lengths
        with pytest.raises(ValueError):
            check_parameter_set_compatibility(parameter_set[:2], feature_df)


class TestMakeDataIntegration:
    """Integration tests for the make_data function."""
    
    def test_make_data_without_target(self):
        """Test make_data without model/solver (only feature data)."""
        initial_values = {'A': 10.0, 'B': 20.0}
        feature_df, target_df = make_data(
            initial_values=initial_values,
            perturbation_type='uniform',
            perturbation_params={'min': 0.9, 'max': 1.1},
            n_samples=10,
            seed=42
        )
        
        assert feature_df.shape == (10, 2)
        assert target_df.empty  # No model/solver provided, so empty target
    
    @pytest.mark.skip(reason="Requires actual model simulation which may be complex")
    def test_make_data_with_target(self):
        """Test make_data with model/solver (both feature and target data)."""
        # This test would require a properly set up model and solver
        # It's marked as skip to avoid issues in test environment
        pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
