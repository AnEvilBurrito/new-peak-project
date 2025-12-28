"""
Test to reproduce and fix the empty timecourse DataFrame issue.
"""
import sys
import os
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.utils.data_generation_helpers import make_data_extended

def test_empty_timecourse_reproduction():
    """Test to reproduce the empty timecourse DataFrame issue."""
    print("Testing empty timecourse DataFrame reproduction...")
    
    # Create mock solver that sometimes returns None to simulate failures
    class MockSolver:
        def __init__(self, fail_rate=0.0):
            self.call_count = 0
            self.fail_rate = fail_rate
        
        def set_state_values(self, values):
            pass
        
        def set_parameter_values(self, values):
            pass
        
        def simulate(self, start, end, points):
            self.call_count += 1
            # Simulate occasional failures
            if np.random.random() < self.fail_rate:
                raise RuntimeError("CV_TOO_MUCH_WORK")
            
            time_points = np.linspace(start, end, points)
            result = pd.DataFrame({
                'time': time_points,
                'Cp': 100 * np.exp(-0.01 * time_points) + np.random.normal(0, 0.5, points),
                'A': 50 * (1 - np.exp(-0.02 * time_points)) + np.random.normal(0, 0.3, points),
                'Ap': 20 * np.sin(0.05 * time_points) + np.random.normal(0, 0.2, points)
            })
            return result
    
    # Create mock model spec
    class MockModelSpec:
        def __init__(self):
            self.A_species = ['A']
            self.B_species = []
            self.C_species = []
    
    mock_spec = MockModelSpec()
    
    initial_values = {'A': 100.0}
    
    # Test 1: No failures
    print("\n1. Testing with no failures:")
    mock_solver = MockSolver(fail_rate=0.0)
    result1 = make_data_extended(
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
    
    print(f"  Timecourse type: {type(result1['timecourse'])}")
    print(f"  Timecourse shape: {result1['timecourse'].shape if isinstance(result1['timecourse'], pd.DataFrame) else 'list'}")
    print(f"  Timecourse head empty?: {result1['timecourse'].head().empty if isinstance(result1['timecourse'], pd.DataFrame) else 'N/A'}")
    print(f"  Timecourse columns: {list(result1['timecourse'].columns) if isinstance(result1['timecourse'], pd.DataFrame) else 'N/A'}")
    
    # Test 2: All failures (simulate with high fail rate)
    print("\n2. Testing with all failures (simulated by making solver fail):")
    
    class AlwaysFailSolver:
        def set_state_values(self, values):
            pass
        
        def set_parameter_values(self, values):
            pass
        
        def simulate(self, start, end, points):
            raise RuntimeError("CV_TOO_MUCH_WORK")
    
    mock_solver_fail = AlwaysFailSolver()
    
    try:
        result2 = make_data_extended(
            initial_values=initial_values,
            perturbation_type='gaussian',
            perturbation_params={'std': 5.0},
            n_samples=5,
            model_spec=mock_spec,
            solver=mock_solver_fail,
            simulation_params={'start': 0, 'end': 100, 'points': 11},
            seed=42,
            verbose=False,
            require_all_successful=False  # Allow failures
        )
        
        print(f"  Timecourse type: {type(result2['timecourse'])}")
        if isinstance(result2['timecourse'], pd.DataFrame):
            print(f"  Timecourse shape: {result2['timecourse'].shape}")
            print(f"  Timecourse head:\n{result2['timecourse'].head()}")
            print(f"  Timecourse columns: {list(result2['timecourse'].columns)}")
            print(f"  Timecourse is empty: {result2['timecourse'].empty}")
        elif isinstance(result2['timecourse'], list):
            print(f"  Timecourse list length: {len(result2['timecourse'])}")
            print(f"  Timecourse list elements: {result2['timecourse']}")
    
    except Exception as e:
        print(f"  Error during test: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Check the actual structure issue
    print("\n3. Testing DataFrame structure for display issues:")
    
    # Create a DataFrame with the structure we have
    test_df = pd.DataFrame([
        {'A': np.array([1.0, 2.0, 3.0]), 'Ap': np.array([0.1, 0.2, 0.3])},
        {'A': np.array([1.1, 2.1, 3.1]), 'Ap': np.array([0.11, 0.21, 0.31])},
        {'A': np.array([1.2, 2.2, 3.2]), 'Ap': np.array([0.12, 0.22, 0.32])}
    ])
    
    print(f"  Test DataFrame shape: {test_df.shape}")
    print(f"  Test DataFrame columns: {list(test_df.columns)}")
    print(f"  Test DataFrame head() output:\n{test_df.head()}")
    print(f"  Test DataFrame empty?: {test_df.empty}")
    print(f"  Test DataFrame has columns?: {len(test_df.columns) > 0}")
    
    # The issue might be that when all simulations fail, we get None values
    test_df_with_none = pd.DataFrame([
        None,
        None,
        None
    ])
    
    print(f"\n  DataFrame with None values shape: {test_df_with_none.shape}")
    print(f"  DataFrame with None values columns: {list(test_df_with_none.columns)}")
    print(f"  DataFrame with None values head():\n{test_df_with_none.head()}")
    print(f"  DataFrame with None values empty?: {test_df_with_none.empty}")
    
    return True

def test_timecourse_robust_structure():
    """Test that timecourse structure is robust to failures."""
    print("\nTesting robust timecourse structure...")
    
    # Create a helper to fix the structure issue
    def create_robust_timecourse_dataframe(timecourse_dicts, species_columns):
        """
        Create a robust timecourse DataFrame that handles None values properly.
        
        Args:
            timecourse_dicts: List of dictionaries (or None for failed simulations)
            species_columns: List of expected species column names
            
        Returns:
            DataFrame with consistent structure
        """
        if not timecourse_dicts:
            return pd.DataFrame()
        
        # Create a DataFrame with the right columns, even if empty
        df = pd.DataFrame(timecourse_dicts)
        
        # Ensure all expected columns exist (fill with None if missing)
        for col in species_columns:
            if col not in df.columns:
                df[col] = None
        
        return df
    
    # Test the helper
    test_data_good = [
        {'A': np.array([1, 2, 3]), 'Ap': np.array([0.1, 0.2, 0.3])},
        {'A': np.array([4, 5, 6]), 'Ap': np.array([0.4, 0.5, 0.6])}
    ]
    
    test_data_with_none = [
        None,
        {'A': np.array([1, 2, 3]), 'Ap': np.array([0.1, 0.2, 0.3])},
        None
    ]
    
    test_data_all_none = [None, None, None]
    
    print("Testing helper function:")
    
    df1 = create_robust_timecourse_dataframe(test_data_good, ['A', 'Ap'])
    print(f"  Good data DataFrame shape: {df1.shape}")
    print(f"  Good data DataFrame columns: {list(df1.columns)}")
    
    df2 = create_robust_timecourse_dataframe(test_data_with_none, ['A', 'Ap'])
    print(f"  Data with None DataFrame shape: {df2.shape}")
    print(f"  Data with None DataFrame columns: {list(df2.columns)}")
    
    df3 = create_robust_timecourse_dataframe(test_data_all_none, ['A', 'Ap'])
    print(f"  All None DataFrame shape: {df3.shape}")
    print(f"  All None DataFrame columns: {list(df3.columns)}")
    print(f"  All None DataFrame empty: {df3.empty}")
    
    return True

def main():
    """Run all tests."""
    print("Running timecourse empty DataFrame tests...")
    print("=" * 60)
    
    tests = [
        test_empty_timecourse_reproduction,
        test_timecourse_robust_structure
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
