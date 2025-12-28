"""
Test script for robust data generation with error handling.
"""
import sys
import os
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.utils.data_generation_helpers import make_data, generate_batch_alternatives
from models.Solver.ScipySolver import ScipySolver
from models.Specs.ModelSpecification import ModelSpecification

def test_generate_batch_alternatives():
    """Test the batch alternatives generation function."""
    print("Testing generate_batch_alternatives...")
    
    base_values = {'A': 10.0, 'B': 20.0, 'C': 30.0}
    
    # Test Gaussian perturbation
    df = generate_batch_alternatives(
        base_values=base_values,
        perturbation_type='gaussian',
        perturbation_params={'std': 2.0},
        batch_size=5,
        base_seed=42,
        attempt=1
    )
    
    assert df.shape == (5, 3), f"Expected shape (5, 3), got {df.shape}"
    assert list(df.columns) == ['A', 'B', 'C'], f"Expected columns ['A', 'B', 'C'], got {list(df.columns)}"
    
    # Verify values are perturbed
    for col in df.columns:
        assert not np.allclose(df[col], base_values[col]), f"Values for {col} should be perturbed"
    
    print("✓ generate_batch_alternatives passed")
    return True

def test_make_data_signature():
    """Test that the make_data function accepts new parameters."""
    print("Testing make_data signature...")
    
    # Create a simple model spec for testing
    class SimpleSpec:
        def __init__(self):
            self.A_species = ['A', 'B', 'C']
            self.B_species = []
    
    simple_spec = SimpleSpec()
    
    # Create a mock solver that sometimes fails
    class MockSolver:
        def __init__(self, fail_probability=0.2):
            self.fail_probability = fail_probability
            self.call_count = 0
        
        def set_state_values(self, values):
            pass
        
        def set_parameter_values(self, values):
            pass
        
        def simulate(self, start, end, points):
            self.call_count += 1
            # Simulate occasional failure
            if np.random.random() < self.fail_probability:
                raise RuntimeError("CVODE CV_TOO_MUCH_WORK: At t = 0.001, too many steps taken without reaching tout.")
            
            # Return mock simulation result
            time_points = np.linspace(start, end, points)
            result = pd.DataFrame({
                'time': time_points,
                'Cp': np.random.random(points) * 100,
                'A': np.random.random(points) * 50,
                'B': np.random.random(points) * 50,
                'C': np.random.random(points) * 50
            })
            return result
    
    mock_solver = MockSolver(fail_probability=0.3)
    
    initial_values = {'A': 10.0, 'B': 20.0, 'C': 30.0}
    
    try:
        # Test with new parameters
        feature_df, target_df = make_data(
            initial_values=initial_values,
            perturbation_type='gaussian',
            perturbation_params={'std': 2.0},
            n_samples=10,
            model_spec=simple_spec,
            solver=mock_solver,
            seed=42,
            resample_size=5,
            max_retries=2,
            require_all_successful=False,
            outcome_var='Cp',
            simulation_params={'start': 0, 'end': 100, 'points': 50},
            verbose=False
        )
        
        print(f"✓ make_data with new parameters executed successfully")
        print(f"  Feature shape: {feature_df.shape}")
        print(f"  Target shape: {target_df.shape}")
        print(f"  Target contains NaN: {target_df['Cp'].isna().sum()} out of {len(target_df)}")
        print(f"  Solver calls: {mock_solver.call_count}")
        
        return True
        
    except Exception as e:
        print(f"✗ make_data test failed: {e}")
        return False

def test_require_all_successful():
    """Test the require_all_successful parameter."""
    print("Testing require_all_successful parameter...")
    
    # Create a mock solver that always fails
    class AlwaysFailSolver:
        def set_state_values(self, values):
            pass
        
        def set_parameter_values(self, values):
            pass
        
        def simulate(self, start, end, points):
            raise RuntimeError("CVODE CV_TOO_MUCH_WORK: At t = 0.001, too many steps taken without reaching tout.")
    
    class SimpleSpec:
        def __init__(self):
            self.A_species = ['A', 'B']
            self.B_species = []
    
    initial_values = {'A': 10.0, 'B': 20.0}
    
    # Test with require_all_successful=False (should succeed with NaN values)
    try:
        feature_df, target_df = make_data(
            initial_values=initial_values,
            perturbation_type='gaussian',
            perturbation_params={'std': 1.0},
            n_samples=3,
            model_spec=SimpleSpec(),
            solver=AlwaysFailSolver(),
            seed=42,
            resample_size=2,
            max_retries=1,
            require_all_successful=False,
            outcome_var='Cp',
            simulation_params={'start': 0, 'end': 10, 'points': 5},
            verbose=False
        )
        
        assert target_df['Cp'].isna().all(), "All target values should be NaN when solver always fails"
        print("✓ require_all_successful=False works correctly (returns NaN for failures)")
        
    except Exception as e:
        print(f"✗ require_all_successful=False test failed: {e}")
        return False
    
    # Test with require_all_successful=True (should raise exception)
    try:
        feature_df, target_df = make_data(
            initial_values=initial_values,
            perturbation_type='gaussian',
            perturbation_params={'std': 1.0},
            n_samples=3,
            model_spec=SimpleSpec(),
            solver=AlwaysFailSolver(),
            seed=42,
            resample_size=2,
            max_retries=1,
            require_all_successful=True,
            outcome_var='Cp',
            simulation_params={'start': 0, 'end': 10, 'points': 5},
            verbose=False
        )
        
        print("✗ require_all_successful=True should have raised an exception")
        return False
        
    except RuntimeError as e:
        if "Failed to simulate" in str(e):
            print("✓ require_all_successful=True correctly raises RuntimeError")
            return True
        else:
            print(f"✗ Unexpected error type: {e}")
            return False
    except Exception as e:
        print(f"✗ Unexpected exception: {e}")
        return False

def main():
    """Run all tests."""
    print("Running tests for robust data generation...")
    print("=" * 60)
    
    tests = [
        test_generate_batch_alternatives,
        test_make_data_signature,
        test_require_all_successful
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} raised exception: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Test results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
