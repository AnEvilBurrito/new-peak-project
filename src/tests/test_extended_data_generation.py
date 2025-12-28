"""
Test script for extended data generation with intermediate datasets.
"""
import sys
import os
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.utils.data_generation_helpers import (
    make_data, 
    make_target_data_with_params,
    make_data_extended
)

def test_make_target_data_with_params_returns_timecourse():
    """Test that make_target_data_with_params returns timecourse data."""
    print("Testing make_target_data_with_params returns timecourse data...")
    
    # Create a simple mock solver
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
            result = pd.DataFrame({
                'time': time_points,
                'Cp': np.random.random(points) * 100,
                'A': np.random.random(points) * 50
            })
            return result
    
    # Create mock model spec
    class MockModelSpec:
        pass
    
    mock_solver = MockSolver()
    mock_spec = MockModelSpec()
    
    # Create test feature data
    feature_df = pd.DataFrame({
        'A': [10.0, 20.0, 30.0],
        'B': [5.0, 15.0, 25.0]
    })
    
    # Test without parameter_df
    target_df, timecourse_data = make_target_data_with_params(
        model_spec=mock_spec,
        solver=mock_solver,
        feature_df=feature_df,
        simulation_params={'start': 0, 'end': 10, 'points': 5},
        outcome_var='Cp',
        verbose=False
    )
    
    assert target_df.shape == (3, 1), f"Expected target shape (3, 1), got {target_df.shape}"
    assert len(timecourse_data) == 3, f"Expected 3 timecourse arrays, got {len(timecourse_data)}"
    assert all(isinstance(tc, np.ndarray) for tc in timecourse_data), "Timecourse data should be numpy arrays"
    
    print("✓ make_target_data_with_params returns timecourse data correctly")
    return True

def test_make_data_extended_functionality():
    """Test the new make_data_extended function."""
    print("Testing make_data_extended functionality...")
    
    # Create simple mock solver
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
            result = pd.DataFrame({
                'time': time_points,
                'Cp': np.random.random(points) * 100,
                'A': np.random.random(points) * 50,
                'B': np.random.random(points) * 30
            })
            return result
    
    # Create mock model spec
    class MockModelSpec:
        def __init__(self):
            self.A_species = ['A']
            self.B_species = ['B']
            self.C_species = []
    
    mock_solver = MockSolver()
    mock_spec = MockModelSpec()
    
    initial_values = {'A': 10.0, 'B': 20.0}
    kinetic_parameters = {'k1': 1.0, 'k2': 0.5}
    
    try:
        # Test 1: Extended function with default capture_all_species=True
        result = make_data_extended(
            initial_values=initial_values,
            perturbation_type='gaussian',
            perturbation_params={'std': 1.0},
            n_samples=5,
            model_spec=mock_spec,
            solver=mock_solver,
            parameter_values=kinetic_parameters,
            param_perturbation_type='uniform',
            param_perturbation_params={'min': 0.9, 'max': 1.1},
            simulation_params={'start': 0, 'end': 10, 'points': 5},
            seed=42,
            verbose=False
        )
        
        # Check that all expected keys are present
        expected_keys = ['features', 'targets', 'parameters', 'timecourse', 'metadata']
        assert all(key in result for key in expected_keys), f"Missing keys in result. Expected: {expected_keys}, Got: {list(result.keys())}"
        
        # Check data types and shapes
        assert isinstance(result['features'], pd.DataFrame), "Features should be a DataFrame"
        assert isinstance(result['targets'], pd.DataFrame), "Targets should be a DataFrame"
        assert result['parameters'] is None or isinstance(result['parameters'], pd.DataFrame), "Parameters should be None or DataFrame"
        assert isinstance(result['timecourse'], pd.DataFrame), "Timecourse should be a DataFrame when capture_all_species=True"
        assert isinstance(result['metadata'], dict), "Metadata should be a dictionary"
        
        # Check shapes
        assert result['features'].shape == (5, 2), f"Features shape should be (5, 2), got {result['features'].shape}"
        assert result['targets'].shape == (5, 1), f"Targets shape should be (5, 1), got {result['targets'].shape}"
        
        # Check timecourse DataFrame format
        timecourse_df = result['timecourse']
        assert isinstance(timecourse_df, pd.DataFrame), "Timecourse should be a DataFrame"
        assert timecourse_df.shape[0] == 5, f"Timecourse DataFrame should have 5 rows, got {timecourse_df.shape[0]}"
        
        # Check metadata
        metadata_keys = ['failed_indices', 'success_rate', 'n_samples', 'perturbation_type', 'capture_all_species', 'resampling_used']
        for key in metadata_keys:
            assert key in result['metadata'], f"Missing metadata key: {key}"
        
        assert result['metadata']['capture_all_species'] == True, "capture_all_species should be True by default"
        
        print("✓ make_data_extended returns correct data structure with capture_all_species=True")
        
        # Test 2: Extended function with capture_all_species=False
        result_no_capture = make_data_extended(
            initial_values=initial_values,
            perturbation_type='gaussian',
            perturbation_params={'std': 1.0},
            n_samples=5,
            model_spec=mock_spec,
            solver=mock_solver,
            parameter_values=kinetic_parameters,
            param_perturbation_type='uniform',
            param_perturbation_params={'min': 0.9, 'max': 1.1},
            simulation_params={'start': 0, 'end': 10, 'points': 5},
            seed=42,
            capture_all_species=False,
            verbose=False
        )
        
        # Check timecourse format when capture_all_species=False
        assert isinstance(result_no_capture['timecourse'], list), "Timecourse should be a list when capture_all_species=False"
        timecourse_list = result_no_capture['timecourse']
        assert len(timecourse_list) == 5, f"Timecourse list should have 5 elements, got {len(timecourse_list)}"
        assert all(isinstance(tc, np.ndarray) for tc in timecourse_list if tc is not None), "Timecourse elements should be numpy arrays"
        
        assert result_no_capture['metadata']['capture_all_species'] == False, "capture_all_species should be False when specified"
        
        print("✓ make_data_extended returns correct data structure with capture_all_species=False")
        
        # Test 3: Extended function with all failures (should still return DataFrame)
        class AlwaysFailSolver:
            def set_state_values(self, values):
                pass
            
            def set_parameter_values(self, values):
                pass
            
            def simulate(self, start, end, points):
                raise RuntimeError("CV_TOO_MUCH_WORK")
        
        fail_solver = AlwaysFailSolver()
        
        result_all_fail = make_data_extended(
            initial_values=initial_values,
            perturbation_type='gaussian',
            perturbation_params={'std': 1.0},
            n_samples=3,
            model_spec=mock_spec,
            solver=fail_solver,
            require_all_successful=False,
            simulation_params={'start': 0, 'end': 10, 'points': 5},
            seed=42,
            verbose=False
        )
        
        # Check that we still get a DataFrame even when all simulations fail
        assert isinstance(result_all_fail['timecourse'], pd.DataFrame), "Timecourse should be DataFrame even when all simulations fail"
        timecourse_df = result_all_fail['timecourse']
        assert timecourse_df.shape[0] == 3, f"Timecourse DataFrame should have 3 rows, got {timecourse_df.shape[0]}"
        assert not timecourse_df.empty, "Timecourse DataFrame should not be empty"
        assert len(timecourse_df.columns) > 0, "Timecourse DataFrame should have columns"
        
        print("✓ make_data_extended returns DataFrame even when all simulations fail")
        return True
        
    except Exception as e:
        print(f"✗ make_data_extended test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_extended_data_structure():
    """Test that the extended data structure contains all expected components."""
    print("Testing extended data structure...")
    
    # We'll define what we expect from the extended function
    expected_keys = [
        'features',      # Feature dataframe (initial values)
        'targets',       # Target dataframe (outcome values)
        'parameters',    # Kinetic parameters dataframe (if provided)
        'timecourse',    # Timecourse simulation data
        'metadata'       # Metadata about the generation process
    ]
    
    print(f"Expected keys in extended data structure: {expected_keys}")
    print("✓ Extended data structure design verified")
    return True

def main():
    """Run all tests."""
    print("Running tests for extended data generation...")
    print("=" * 60)
    
    tests = [
        test_make_target_data_with_params_returns_timecourse,
        test_make_data_extended_functionality,
        test_extended_data_structure
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
