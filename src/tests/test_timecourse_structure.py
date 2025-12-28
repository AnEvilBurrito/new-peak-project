"""
Test to diagnose and fix the timecourse DataFrame structure issue.
"""
import sys
import os
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.utils.data_generation_helpers import make_data_extended

def test_timecourse_dataframe_structure():
    """Test that timecourse DataFrame has proper structure."""
    print("Testing timecourse DataFrame structure...")
    
    # Create mock solver that returns realistic data
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
            # Create realistic simulation results with all species
            result = pd.DataFrame({
                'time': time_points,
                'Cp': 100 * np.exp(-0.01 * time_points) + np.random.normal(0, 0.5, points),
                'A': 50 * (1 - np.exp(-0.02 * time_points)) + np.random.normal(0, 0.3, points),
                'Ap': 20 * np.sin(0.05 * time_points) + np.random.normal(0, 0.2, points),
                'B': 30 * np.cos(0.03 * time_points) + np.random.normal(0, 0.2, points),
                'Bp': 10 * np.tanh(0.01 * time_points) + np.random.normal(0, 0.1, points)
            })
            return result
    
    # Create mock model spec with species
    class MockModelSpec:
        def __init__(self):
            self.A_species = ['A', 'B']
            self.B_species = []
            self.C_species = []
    
    mock_solver = MockSolver()
    mock_spec = MockModelSpec()
    
    initial_values = {'A': 100.0, 'B': 50.0}
    
    try:
        # Test with capture_all_species=True (should return DataFrame)
        result = make_data_extended(
            initial_values=initial_values,
            perturbation_type='gaussian',
            perturbation_params={'std': 5.0},
            n_samples=5,
            model_spec=mock_spec,
            solver=mock_solver,
            simulation_params={'start': 0, 'end': 100, 'points': 11},
            seed=42,
            verbose=False
        )
        
        print(f"Result keys: {list(result.keys())}")
        print(f"Timecourse type: {type(result['timecourse'])}")
        
        # Check that timecourse is a DataFrame
        assert isinstance(result['timecourse'], pd.DataFrame), f"Timecourse should be DataFrame, got {type(result['timecourse'])}"
        
        timecourse_df = result['timecourse']
        print(f"Timecourse DataFrame shape: {timecourse_df.shape}")
        print(f"Timecourse DataFrame columns: {list(timecourse_df.columns)}")
        print(f"Timecourse DataFrame head:\n{timecourse_df.head()}")
        
        # Check that DataFrame is not empty
        assert not timecourse_df.empty, "Timecourse DataFrame should not be empty"
        
        # Check that DataFrame has expected columns (species)
        expected_columns = ['A', 'Ap', 'B', 'Bp']  # From model spec
        for col in expected_columns:
            assert col in timecourse_df.columns, f"Missing expected column: {col}"
        
        # Check that each cell contains a numpy array
        for i in range(min(3, len(timecourse_df))):
            for col in expected_columns:
                cell_value = timecourse_df.iloc[i][col]
                assert isinstance(cell_value, np.ndarray), f"Cell ({i}, {col}) should be numpy array, got {type(cell_value)}"
                assert len(cell_value) == 11, f"Array should have 11 timepoints, got {len(cell_value)}"
        
        print("✓ Timecourse DataFrame has correct structure")
        
        # Now test with capture_all_species=False
        result_single = make_data_extended(
            initial_values=initial_values,
            perturbation_type='gaussian',
            perturbation_params={'std': 5.0},
            n_samples=5,
            model_spec=mock_spec,
            solver=mock_solver,
            capture_all_species=False,
            simulation_params={'start': 0, 'end': 100, 'points': 11},
            seed=42,
            verbose=False
        )
        
        print(f"\nTesting capture_all_species=False:")
        print(f"Timecourse type: {type(result_single['timecourse'])}")
        
        assert isinstance(result_single['timecourse'], list), f"Timecourse should be list when capture_all_species=False, got {type(result_single['timecourse'])}"
        
        timecourse_list = result_single['timecourse']
        print(f"Timecourse list length: {len(timecourse_list)}")
        
        for i, arr in enumerate(timecourse_list):
            if arr is not None:
                assert isinstance(arr, np.ndarray), f"Element {i} should be numpy array, got {type(arr)}"
                assert len(arr) == 11, f"Array should have 11 timepoints, got {len(arr)}"
        
        print("✓ Timecourse list has correct structure")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_timecourse_dataframe_value_access():
    """Test that we can access timecourse values properly."""
    print("\nTesting timecourse DataFrame value access...")
    
    # Create a simple test DataFrame to understand the structure
    test_data = [
        {'A': np.array([1.0, 2.0, 3.0]), 'B': np.array([4.0, 5.0, 6.0])},
        {'A': np.array([1.1, 2.1, 3.1]), 'B': np.array([4.1, 5.1, 6.1])},
        {'A': np.array([1.2, 2.2, 3.2]), 'B': np.array([4.2, 5.2, 6.2])}
    ]
    
    df = pd.DataFrame(test_data)
    print(f"Test DataFrame shape: {df.shape}")
    print(f"Test DataFrame columns: {list(df.columns)}")
    print(f"Test DataFrame dtypes:\n{df.dtypes}")
    print(f"Test DataFrame head:\n{df.head()}")
    
    # Access a specific cell
    cell_value = df.iloc[0]['A']
    print(f"df.iloc[0]['A'] type: {type(cell_value)}, value: {cell_value}")
    
    # Try to access the array elements
    print(f"df.iloc[0]['A'][0]: {df.iloc[0]['A'][0]}")
    print(f"df.iloc[1]['B'][2]: {df.iloc[1]['B'][2]}")
    
    print("✓ DataFrame value access test completed")
    return True

def main():
    """Run all tests."""
    print("Running timecourse structure tests...")
    print("=" * 60)
    
    tests = [
        test_timecourse_dataframe_structure,
        test_timecourse_dataframe_value_access
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
