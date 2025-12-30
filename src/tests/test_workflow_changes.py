#!/usr/bin/env python3
"""
Test script to verify the new batch_eval function and backward compatibility
of batch_eval_standard in Workflow.py
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..')
sys.path.insert(0, src_dir)

from ml.Workflow import batch_eval, batch_eval_standard, build_pipeline

def test_backward_compatibility():
    """Test that batch_eval_standard still works as before"""
    print("🧪 Testing backward compatibility of batch_eval_standard...")
    
    # Generate simple test data
    np.random.seed(42)
    n_samples = 50
    n_features = 5
    
    # Create synthetic feature data
    feature_data = pd.DataFrame({
        f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)
    })
    
    # Create target data with some relationship
    target_data = pd.DataFrame({
        'target': 0.3 * feature_data['feature_0'] + 
                 0.5 * feature_data['feature_1'] + 
                 np.random.randn(n_samples) * 0.2
    })
    
    # Test batch_eval_standard with reduced parameters for speed
    results_standard = batch_eval_standard(
        feature_data_list=[feature_data],
        feature_data_names=['test_features'],
        target_data=target_data,
        target_name='target',
        num_repeats=2,  # Reduced for testing
        o_random_seed=42,
        n_jobs=1  # Run serially for simplicity
    )
    
    # Verify results structure
    assert isinstance(results_standard, pd.DataFrame), "Results should be a DataFrame"
    assert len(results_standard) > 0, "Results should not be empty"
    expected_columns = ['Model', 'Feature Data', 'Mean Squared Error', 
                       'R2 Score', 'Pearson Correlation', 'Pearson P-Value']
    for col in expected_columns:
        assert col in results_standard.columns, f"Missing column: {col}"
    
    # Should have 5 models (standard set) × 1 feature set × 2 repeats = 10 rows
    assert len(results_standard) == 5 * 1 * 2, f"Expected 10 rows, got {len(results_standard)}"
    
    print("✅ batch_eval_standard backward compatibility test passed")
    print(f"   Results shape: {results_standard.shape}")
    print(f"   Unique models: {results_standard['Model'].unique()}")
    return True

def test_new_batch_eval_function():
    """Test the new batch_eval function with custom models"""
    print("\n🧪 Testing new batch_eval function with custom models...")
    
    # Generate simple test data
    np.random.seed(42)
    n_samples = 50
    n_features = 5
    
    # Create synthetic feature data
    feature_data = pd.DataFrame({
        f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)
    })
    
    # Create target data with some relationship
    target_data = pd.DataFrame({
        'target': 0.3 * feature_data['feature_0'] + 
                 0.5 * feature_data['feature_1'] + 
                 np.random.randn(n_samples) * 0.2
    })
    
    # Create custom models (only 2 instead of 5 for speed)
    custom_models = [
        build_pipeline(LinearRegression()),
        build_pipeline(RandomForestRegressor(n_estimators=50, random_state=42)),
    ]
    
    custom_models_desc = ['Custom Linear Regression', 'Custom Random Forest']
    
    # Test batch_eval with custom models
    results_custom = batch_eval(
        feature_data_list=[feature_data],
        feature_data_names=['test_features'],
        target_data=target_data,
        target_name='target',
        all_models=custom_models,
        all_models_desc=custom_models_desc,
        num_repeats=2,  # Reduced for testing
        o_random_seed=42,
        n_jobs=1  # Run serially for simplicity
    )
    
    # Verify results structure
    assert isinstance(results_custom, pd.DataFrame), "Results should be a DataFrame"
    assert len(results_custom) > 0, "Results should not be empty"
    expected_columns = ['Model', 'Feature Data', 'Mean Squared Error', 
                       'R2 Score', 'Pearson Correlation', 'Pearson P-Value']
    for col in expected_columns:
        assert col in results_custom.columns, f"Missing column: {col}"
    
    # Should have 2 custom models × 1 feature set × 2 repeats = 4 rows
    assert len(results_custom) == 2 * 1 * 2, f"Expected 4 rows, got {len(results_custom)}"
    
    # Verify custom model names are used
    unique_models = results_custom['Model'].unique()
    assert 'Custom Linear Regression' in unique_models
    assert 'Custom Random Forest' in unique_models
    
    print("✅ batch_eval custom models test passed")
    print(f"   Results shape: {results_custom.shape}")
    print(f"   Unique models: {unique_models}")
    return True

def test_error_handling():
    """Test error handling for invalid parameters"""
    print("\n🧪 Testing error handling...")
    
    # Generate simple test data
    np.random.seed(42)
    n_samples = 10
    n_features = 3
    
    feature_data = pd.DataFrame({
        f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)
    })
    
    target_data = pd.DataFrame({
        'target': np.random.randn(n_samples)
    })
    
    # Test 1: Mismatched model lists
    try:
        batch_eval(
            feature_data_list=[feature_data],
            feature_data_names=['test'],
            target_data=target_data,
            target_name='target',
            all_models=[build_pipeline(LinearRegression())],
            all_models_desc=['Model1', 'Model2'],  # Mismatch
            num_repeats=1,
            n_jobs=1
        )
        assert False, "Should have raised ValueError for mismatched lists"
    except ValueError as e:
        assert "must have the same length" in str(e)
        print("✅ Mismatched model lists error handled correctly")
    
    # Test 2: Empty models list
    try:
        batch_eval(
            feature_data_list=[feature_data],
            feature_data_names=['test'],
            target_data=target_data,
            target_name='target',
            all_models=[],  # Empty list
            all_models_desc=[],
            num_repeats=1,
            n_jobs=1
        )
        assert False, "Should have raised ValueError for empty models list"
    except ValueError as e:
        assert "cannot be empty" in str(e)
        print("✅ Empty models list error handled correctly")
    
    # Test 3: Target column not found
    try:
        batch_eval(
            feature_data_list=[feature_data],
            feature_data_names=['test'],
            target_data=target_data,
            target_name='nonexistent_column',  # Invalid column
            all_models=[build_pipeline(LinearRegression())],
            all_models_desc=['Model1'],
            num_repeats=1,
            n_jobs=1
        )
        assert False, "Should have raised ValueError for missing target column"
    except ValueError as e:
        assert "not found" in str(e)
        print("✅ Missing target column error handled correctly")
    
    return True

def main():
    """Run all tests"""
    print("="*60)
    print("🔬 Testing Workflow.py Changes")
    print("="*60)
    
    all_passed = True
    
    try:
        all_passed &= test_backward_compatibility()
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_new_batch_eval_function()
    except Exception as e:
        print(f"❌ New batch_eval function test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_error_handling()
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 All tests passed! The implementation is working correctly.")
        print("\nSummary of changes:")
        print("1. ✅ Created new batch_eval function with all_models and all_models_desc parameters")
        print("2. ✅ Updated batch_eval_standard to use batch_eval for backward compatibility")
        print("3. ✅ Added proper validation for new parameters")
        print("4. ✅ Maintained existing functionality for all scripts using batch_eval_standard")
    else:
        print("❌ Some tests failed. Please check the implementation.")
        sys.exit(1)
    
    return all_passed

if __name__ == "__main__":
    main()
