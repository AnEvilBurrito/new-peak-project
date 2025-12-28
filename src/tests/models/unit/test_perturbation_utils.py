"""
Unit tests for PerturbationUtils module.
"""
import numpy as np
import pandas as pd
import pytest
from models.SyntheticGenUtils.PerturbationUtils import (
    apply_uniform_perturbation,
    apply_gaussian_perturbation,
    apply_lognormal_perturbation,
    generate_perturbation_samples,
    generate_gaussian_perturbation_dataframe,
    generate_lognormal_perturbation_dataframe,
    generate_uniform_perturbation_dataframe,
    generate_lhs_perturbation_dataframe,
    convert_perturbations_to_dataframe,
    validate_initial_values,
    set_random_seed
)


class TestPerturbationUtils:
    """Test suite for PerturbationUtils functions."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.initial_values = {
            'A': 10.0,
            'B': 20.0,
            'C': 30.0
        }
        self.positive_initial_values = {
            'X': 1.5,
            'Y': 2.0,
            'Z': 3.5
        }
    
    def test_validate_initial_values(self):
        """Test initial values validation."""
        # Valid initial values
        validate_initial_values(self.initial_values)
        
        # Invalid: empty dictionary
        with pytest.raises(ValueError, match='cannot be empty'):
            validate_initial_values({})
        
        # Invalid: non-string key
        with pytest.raises(ValueError, match='must be strings'):
            validate_initial_values({1: 10.0})
        
        # Invalid: non-numeric value
        with pytest.raises(ValueError, match='must be numeric'):
            validate_initial_values({'A': 'not a number'})
        
        # Invalid: negative value
        with pytest.raises(ValueError, match='must be non-negative'):
            validate_initial_values({'A': -5.0})
    
    def test_apply_uniform_perturbation(self):
        """Test uniform perturbation generation."""
        rng = np.random.default_rng(42)
        perturbed = apply_uniform_perturbation(
            self.initial_values, min_=0.5, max_=1.5, rng=rng
        )
        
        assert set(perturbed.keys()) == set(self.initial_values.keys())
        for species in self.initial_values:
            assert isinstance(perturbed[species], float)
            # Values should be between min_ * initial and max_ * initial
            assert 0.5 * self.initial_values[species] <= perturbed[species] <= 1.5 * self.initial_values[species]
    
    def test_apply_gaussian_perturbation_with_std(self):
        """Test Gaussian perturbation with absolute standard deviation."""
        rng = np.random.default_rng(42)
        perturbed = apply_gaussian_perturbation(
            self.initial_values, {'std': 2.0}, rng=rng
        )
        
        assert set(perturbed.keys()) == set(self.initial_values.keys())
        for species in self.initial_values:
            assert isinstance(perturbed[species], float)
    
    def test_apply_gaussian_perturbation_with_rsd(self):
        """Test Gaussian perturbation with relative standard deviation."""
        rng = np.random.default_rng(42)
        perturbed = apply_gaussian_perturbation(
            self.initial_values, {'rsd': 0.1}, rng=rng
        )
        
        assert set(perturbed.keys()) == set(self.initial_values.keys())
        for species in self.initial_values:
            assert isinstance(perturbed[species], float)
    
    def test_apply_lognormal_perturbation_with_shape(self):
        """Test lognormal perturbation with shape parameter."""
        rng = np.random.default_rng(42)
        perturbed = apply_lognormal_perturbation(
            self.positive_initial_values, {'shape': 0.5}, rng=rng
        )
        
        assert set(perturbed.keys()) == set(self.positive_initial_values.keys())
        for species in self.positive_initial_values:
            assert isinstance(perturbed[species], float)
            assert perturbed[species] > 0  # Lognormal produces positive values
    
    def test_apply_lognormal_perturbation_with_rsd_shape(self):
        """Test lognormal perturbation with relative shape parameter."""
        rng = np.random.default_rng(42)
        perturbed = apply_lognormal_perturbation(
            self.positive_initial_values, {'rsd_shape': 0.5}, rng=rng
        )
        
        assert set(perturbed.keys()) == set(self.positive_initial_values.keys())
        for species in self.positive_initial_values:
            assert isinstance(perturbed[species], float)
            assert perturbed[species] > 0
    
    def test_apply_lognormal_perturbation_validation(self):
        """Test lognormal perturbation parameter validation."""
        rng = np.random.default_rng(42)
        
        # Invalid: non-positive initial values
        with pytest.raises(ValueError, match='must be positive'):
            apply_lognormal_perturbation({'A': 0}, {'shape': 0.5}, rng)
        
        with pytest.raises(ValueError, match='must be positive'):
            apply_lognormal_perturbation({'A': -1}, {'shape': 0.5}, rng)
        
        # Invalid: missing parameters
        with pytest.raises(ValueError, match='must contain "shape" or "rsd_shape"'):
            apply_lognormal_perturbation(self.positive_initial_values, {}, rng)
        
        # Invalid: non-positive shape parameter
        with pytest.raises(ValueError, match='must be positive'):
            apply_lognormal_perturbation(self.positive_initial_values, {'shape': 0}, rng)
        
        with pytest.raises(ValueError, match='must be positive'):
            apply_lognormal_perturbation(self.positive_initial_values, {'shape': -0.5}, rng)
        
        # Invalid: rsd_shape with initial_value = 1
        with pytest.raises(ValueError, match='Cannot use rsd_shape with initial_value = 1'):
            apply_lognormal_perturbation({'A': 1.0}, {'rsd_shape': 0.5}, rng)
    
    def test_generate_perturbation_samples_uniform(self):
        """Test generation of multiple uniform perturbation samples."""
        samples = generate_perturbation_samples(
            perturbation_type='uniform',
            initial_values=self.initial_values,
            perturbation_params={'min': 0.8, 'max': 1.2},
            n_samples=5,
            seed=42
        )
        
        assert len(samples) == 5
        for sample in samples:
            assert set(sample.keys()) == set(self.initial_values.keys())
            for species in self.initial_values:
                assert isinstance(sample[species], float)
    
    def test_generate_perturbation_samples_gaussian(self):
        """Test generation of multiple Gaussian perturbation samples."""
        samples = generate_perturbation_samples(
            perturbation_type='gaussian',
            initial_values=self.initial_values,
            perturbation_params={'std': 2.0},
            n_samples=5,
            seed=42
        )
        
        assert len(samples) == 5
        for sample in samples:
            assert set(sample.keys()) == set(self.initial_values.keys())
            for species in self.initial_values:
                assert isinstance(sample[species], float)
    
    def test_generate_perturbation_samples_lognormal(self):
        """Test generation of multiple lognormal perturbation samples."""
        samples = generate_perturbation_samples(
            perturbation_type='lognormal',
            initial_values=self.positive_initial_values,
            perturbation_params={'shape': 0.5},
            n_samples=5,
            seed=42
        )
        
        assert len(samples) == 5
        for sample in samples:
            assert set(sample.keys()) == set(self.positive_initial_values.keys())
            for species in self.positive_initial_values:
                assert isinstance(sample[species], float)
                assert sample[species] > 0
    
    def test_convert_perturbations_to_dataframe(self):
        """Test conversion of perturbation samples to DataFrame."""
        samples = [
            {'A': 1.0, 'B': 2.0, 'C': 3.0},
            {'A': 1.1, 'B': 2.1, 'C': 3.1},
            {'A': 1.2, 'B': 2.2, 'C': 3.2}
        ]
        
        df = convert_perturbations_to_dataframe(samples)
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 3)
        assert list(df.columns) == ['A', 'B', 'C']
        assert df['A'].tolist() == [1.0, 1.1, 1.2]
    
    def test_generate_gaussian_perturbation_dataframe(self):
        """Test direct generation of Gaussian perturbation DataFrame."""
        df = generate_gaussian_perturbation_dataframe(
            initial_values=self.initial_values,
            perturbation_params={'std': 1.0},
            n_samples=10,
            seed=42
        )
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (10, 3)
        assert set(df.columns) == set(self.initial_values.keys())
        
        # Test reproducibility with same seed
        df2 = generate_gaussian_perturbation_dataframe(
            initial_values=self.initial_values,
            perturbation_params={'std': 1.0},
            n_samples=10,
            seed=42
        )
        
        pd.testing.assert_frame_equal(df, df2)
    
    def test_generate_lognormal_perturbation_dataframe(self):
        """Test direct generation of lognormal perturbation DataFrame."""
        df = generate_lognormal_perturbation_dataframe(
            initial_values=self.positive_initial_values,
            perturbation_params={'shape': 0.5},
            n_samples=10,
            seed=42
        )
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (10, 3)
        assert set(df.columns) == set(self.positive_initial_values.keys())
        
        # All values should be positive
        assert (df > 0).all().all()
        
        # Test reproducibility with same seed
        df2 = generate_lognormal_perturbation_dataframe(
            initial_values=self.positive_initial_values,
            perturbation_params={'shape': 0.5},
            n_samples=10,
            seed=42
        )
        
        pd.testing.assert_frame_equal(df, df2)
    
    def test_generate_uniform_perturbation_dataframe(self):
        """Test direct generation of uniform perturbation DataFrame."""
        df = generate_uniform_perturbation_dataframe(
            initial_values=self.initial_values,
            perturbation_params={'min': 0.5, 'max': 1.5},
            n_samples=10,
            seed=42
        )
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (10, 3)
        assert set(df.columns) == set(self.initial_values.keys())
        
        # Test reproducibility with same seed
        df2 = generate_uniform_perturbation_dataframe(
            initial_values=self.initial_values,
            perturbation_params={'min': 0.5, 'max': 1.5},
            n_samples=10,
            seed=42
        )
        
        pd.testing.assert_frame_equal(df, df2)
    
    def test_generate_lhs_perturbation_dataframe(self):
        """Test direct generation of LHS perturbation DataFrame."""
        df = generate_lhs_perturbation_dataframe(
            initial_values=self.initial_values,
            perturbation_params={'min': 0.0, 'max': 1.0},
            n_samples=10,
            seed=42
        )
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (10, 3)
        assert set(df.columns) == set(self.initial_values.keys())
        
        # All values should be between min and max
        assert (df >= 0.0).all().all()
        assert (df <= 1.0).all().all()
    
    def test_dataframe_generators_parameter_validation(self):
        """Test parameter validation in DataFrame generators."""
        # Gaussian: missing parameters
        with pytest.raises(ValueError, match='must contain "std" or "rsd"'):
            generate_gaussian_perturbation_dataframe(
                self.initial_values, {}, n_samples=5
            )
        
        # Gaussian: negative std
        with pytest.raises(ValueError, match='must be non-negative'):
            generate_gaussian_perturbation_dataframe(
                self.initial_values, {'std': -1.0}, n_samples=5
            )
        
        # Lognormal: missing parameters
        with pytest.raises(ValueError, match='must contain "shape" or "rsd_shape"'):
            generate_lognormal_perturbation_dataframe(
                self.positive_initial_values, {}, n_samples=5
            )
        
        # Lognormal: non-positive initial values
        with pytest.raises(ValueError, match='must be positive'):
            generate_lognormal_perturbation_dataframe(
                {'A': 0}, {'shape': 0.5}, n_samples=5
            )
        
        # Uniform: missing parameters
        with pytest.raises(ValueError, match='must contain "min" and "max"'):
            generate_uniform_perturbation_dataframe(
                self.initial_values, {}, n_samples=5
            )
        
        # Uniform: min > max
        with pytest.raises(ValueError, match='Minimum must be less than or equal to maximum'):
            generate_uniform_perturbation_dataframe(
                self.initial_values, {'min': 2.0, 'max': 1.0}, n_samples=5
            )
    
    def test_set_random_seed(self):
        """Test random seed setting."""
        # Test with seed
        rng1 = set_random_seed(42)
        rng2 = set_random_seed(42)
        assert rng1.random() == rng2.random()
        
        # Test without seed (should produce different values)
        rng3 = set_random_seed(None)
        rng4 = set_random_seed(None)
        # Note: we can't assert they're different, but we can verify they're generators
        assert isinstance(rng3, np.random.Generator)
        assert isinstance(rng4, np.random.Generator)
