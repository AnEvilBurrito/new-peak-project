"""
Test to verify the hyphen/underscore fix in discover_task_lists() and filter_tasks_by_experiment() functions.

This test validates that the ML batch runner can handle:
- CSV with hyphens in experiment_type (e.g., "expression-noise-v1")
- Configuration with underscores (e.g., "expression_noise_v1")
- Folder names with underscores (e.g., "sy_simple_expression_noise_v1")
"""

import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import shutil

# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "..")
sys.path.insert(0, src_dir)

# Import the actual functions from run-ml-batch-v1.py
import importlib.util

# Path to the ML batch runner script
ml_batch_path = os.path.join(src_dir, "notebooks", "ch5-paper", "machine-learning", "run-ml-batch-v1.py")

# Load the module
spec = importlib.util.spec_from_file_location("ml_batch_module", ml_batch_path)
ml_batch_module = importlib.util.module_from_spec(spec)
sys.modules["ml_batch_module"] = ml_batch_module
spec.loader.exec_module(ml_batch_module)

# Get the functions we want to test
discover_task_lists = ml_batch_module.discover_task_lists
filter_tasks_by_experiment = ml_batch_module.filter_tasks_by_experiment
validate_csv_structure = ml_batch_module.validate_csv_structure


def test_hyphen_underscore_conversion_logic():
    """Test the hyphen/underscore conversion logic directly"""
    print("🧪 Testing hyphen/underscore conversion logic...")
    
    test_cases = [
        ("expression-noise-v1", "expression_noise_v1"),
        ("parameter-distortion-v2", "parameter_distortion_v2"),
        ("response-noise-v1", "response_noise_v1"),
        ("test-exp-v1", "test_exp_v1"),
    ]
    
    for input_str, expected_output in test_cases:
        # Test folder name conversion
        converted = input_str.replace('-', '_')
        assert converted == expected_output, f"Failed to convert '{input_str}' to '{expected_output}', got '{converted}'"
        print(f"  ✅ '{input_str}' -> '{converted}'")
    
    print("✅ Hyphen/underscore conversion logic test passed")
    return True


def test_discover_task_lists_hyphen_fix():
    """Test that discover_task_lists() handles hyphen/underscore mismatch"""
    print("🧪 Testing discover_task_lists() hyphen fix...")
    
    # Mock S3 manager
    s3_manager = Mock()
    s3_manager.save_result_path = "test/path"
    
    # Create mock CSV data with HYPHENS in experiment_type
    mock_csv_data = pd.DataFrame({
        "feature_data": [
            "sy_simple_expression_noise_v1/noise_0/noisy_features.pkl",
            "sy_simple_expression_noise_v1/noise_0.1/noisy_features.pkl"
        ],
        "feature_data_label": ["noisy_features_0", "noisy_features_0.1"],
        "target_data": [
            "sy_simple_expression_noise_v1/noise_0/original_targets.pkl",
            "sy_simple_expression_noise_v1/noise_0.1/original_targets.pkl"
        ],
        "target_data_label": ["original_targets", "original_targets"],
        "experiment_type": ["expression-noise-v1", "expression-noise-v1"],  # HYPHENS
        "level": [0, 0.1],
        "model_name": ["sy_simple", "sy_simple"]
    })
    
    # Mock load_data_from_path to return our CSV
    s3_manager.load_data_from_path = Mock(return_value=mock_csv_data)
    
    # Test with configuration using UNDERSCORES
    experiment_types = ["expression_noise_v1"]  # UNDERSCORES
    model_names = ["sy_simple"]
    
    result = discover_task_lists(experiment_types, model_names, s3_manager)
    
    # Verify S3 path construction - should use UNDERSCOPES for folder name
    expected_path = "test/path/data/sy_simple_expression_noise_v1/task_list.csv"
    s3_manager.load_data_from_path.assert_called_with(expected_path, data_format="csv")
    
    # Verify result - should find both rows despite hyphen/underscore mismatch
    assert len(result) == 2, f"Should find 2 tasks, got {len(result)}"
    assert all(result["model_name"] == "sy_simple")
    assert all(result["experiment_type"] == "expression-noise-v1")  # Original hyphens preserved
    
    print("✅ discover_task_lists() hyphen fix test passed")
    print(f"   Found {len(result)} tasks with experiment_type={result.iloc[0]['experiment_type']}")
    return True


def test_filter_tasks_by_experiment_hyphen_fix():
    """Test that filter_tasks_by_experiment() handles hyphen/underscore mismatch"""
    print("🧪 Testing filter_tasks_by_experiment() hyphen fix...")
    
    # Create DataFrame with hyphens in experiment_type
    df = pd.DataFrame({
        "feature_data": ["f1.pkl", "f2.pkl", "f3.pkl", "f4.pkl"],
        "feature_data_label": ["f1", "f2", "f3", "f4"],
        "target_data": ["t1.pkl", "t2.pkl", "t3.pkl", "t4.pkl"],
        "target_data_label": ["t1", "t2", "t3", "t4"],
        "experiment_type": ["expression-noise-v1", "parameter-distortion-v2", 
                          "expression-noise-v1", "response-noise-v1"],  # HYPHENS
        "level": [0, 1.1, 0.1, 0.05],
        "model_name": ["sy_simple", "sy_simple", "sy_simple", "sy_simple"]
    })
    
    # Test 1: Filter with underscores in configuration
    experiment_types = ["expression_noise_v1"]  # UNDERSCORES
    filtered_df = filter_tasks_by_experiment(df, experiment_types)
    
    assert len(filtered_df) == 2, f"Should find 2 expression-noise-v1 tasks, got {len(filtered_df)}"
    assert all(filtered_df["experiment_type"] == "expression-noise-v1")
    
    # Test 2: Filter with hyphens in configuration (also should work)
    experiment_types = ["expression-noise-v1"]  # HYPHENS
    filtered_df = filter_tasks_by_experiment(df, experiment_types)
    
    assert len(filtered_df) == 2, f"Should find 2 expression-noise-v1 tasks, got {len(filtered_df)}"
    
    # Test 3: Filter with multiple experiment types (mix of hyphens and underscores)
    experiment_types = ["expression_noise_v1", "parameter-distortion-v2"]  # Mixed
    filtered_df = filter_tasks_by_experiment(df, experiment_types)
    
    assert len(filtered_df) == 3, f"Should find 3 tasks, got {len(filtered_df)}"
    exp_types = set(filtered_df["experiment_type"])
    assert exp_types == {"expression-noise-v1", "parameter-distortion-v2"}
    
    print("✅ filter_tasks_by_experiment() hyphen fix test passed")
    return True


def test_end_to_end_workflow():
    """Test end-to-end workflow with hyphen/underscore fix"""
    print("🧪 Testing end-to-end workflow with hyphen fix...")
    
    # Create a more complex test scenario
    test_dir = tempfile.mkdtemp()
    
    try:
        # Simulate CSV generation by data-eng scripts (with hyphens)
        csv_data = pd.DataFrame({
            "feature_data": [
                "sy_simple_expression_noise_v1/noise_0/noisy_features.pkl",
                "sy_simple_expression_noise_v1/noise_0.1/noisy_features.pkl",
                "sy_simple_parameter_distortion_v2/distortion_1.1/features.pkl",
                "sy_simple_response_noise_v1/noise_0.05/noisy_features.pkl"
            ],
            "feature_data_label": [
                "noisy_features_0", "noisy_features_0.1", 
                "features_1.1", "noisy_features_0.05"
            ],
            "target_data": [
                "sy_simple_expression_noise_v1/noise_0/original_targets.pkl",
                "sy_simple_expression_noise_v1/noise_0.1/original_targets.pkl",
                "sy_simple_parameter_distortion_v2/distortion_1.1/targets.pkl",
                "sy_simple_response_noise_v1/noise_0.05/clean_targets.pkl"
            ],
            "target_data_label": [
                "original_targets", "original_targets",
                "original_targets", "original_targets"
            ],
            "experiment_type": [
                "expression-noise-v1", "expression-noise-v1",
                "parameter-distortion-v2", "response-noise-v1"
            ],  # All with hyphens
            "level": [0, 0.1, 1.1, 0.05],
            "model_name": ["sy_simple", "sy_simple", "sy_simple", "sy_simple"]
        })
        
        csv_path = os.path.join(test_dir, "task_list.csv")
        csv_data.to_csv(csv_path, index=False)
        
        # Load the CSV (simulating ML batch runner)
        loaded_df = pd.read_csv(csv_path)
        
        # Validate CSV structure
        assert validate_csv_structure(loaded_df), "CSV validation failed"
        
        # Test filtering with underscores in configuration
        experiment_types = ["expression_noise_v1", "parameter_distortion_v2"]  # UNDERSCORES
        model_names = ["sy_simple"]
        
        # Mock S3 manager for discover_task_lists
        s3_manager = Mock()
        s3_manager.save_result_path = "test/path"
        s3_manager.load_data_from_path = Mock(return_value=loaded_df)
        
        # Test discover_task_lists
        discovered_df = discover_task_lists(experiment_types, model_names, s3_manager)
        
        # Should find 3 tasks (2 expression-noise-v1 + 1 parameter-distortion-v2)
        assert len(discovered_df) == 3, f"Should find 3 tasks, got {len(discovered_df)}"
        
        # Test filter_tasks_by_experiment on discovered data
        filtered_df = filter_tasks_by_experiment(discovered_df, experiment_types)
        assert len(filtered_df) == 3, "Filtering should preserve all tasks"
        
        # Verify the workflow handles the mismatch correctly
        print(f"  CSV has experiment_types: {loaded_df['experiment_type'].unique()}")
        print(f"  Configuration uses: {experiment_types}")
        print(f"  Discovered {len(discovered_df)} tasks successfully")
        
        print("✅ End-to-end workflow test passed")
        return True
        
    finally:
        shutil.rmtree(test_dir)


def test_edge_cases():
    """Test edge cases for hyphen/underscore handling"""
    print("🧪 Testing edge cases...")
    
    # Edge case 1: Mixed hyphen/underscore in CSV
    df_mixed = pd.DataFrame({
        "feature_data": ["f1.pkl", "f2.pkl", "f3.pkl"],
        "feature_data_label": ["f1", "f2", "f3"],
        "target_data": ["t1.pkl", "t2.pkl", "t3.pkl"],
        "target_data_label": ["t1", "t2", "t3"],
        "experiment_type": ["expression-noise-v1", "expression_noise_v1", "expression-noise-v1"],  # Mixed!
        "level": [0, 0.1, 0.2],
        "model_name": ["sy_simple", "sy_simple", "sy_simple"]
    })
    
    # Should handle all variations
    experiment_types = ["expression_noise_v1"]
    filtered_df = filter_tasks_by_experiment(df_mixed, experiment_types)
    
    # Should find all 3 (since we normalize for comparison)
    assert len(filtered_df) == 3, f"Should find all 3 tasks regardless of hyphen/underscore, got {len(filtered_df)}"
    
    # Edge case 2: Empty DataFrame
    df_empty = pd.DataFrame(columns=[
        "feature_data", "feature_data_label", "target_data", 
        "target_data_label", "experiment_type", "level", "model_name"
    ])
    
    filtered_df = filter_tasks_by_experiment(df_empty, ["expression_noise_v1"])
    assert len(filtered_df) == 0, "Empty DataFrame should remain empty"
    
    # Edge case 3: None experiment_types
    df = pd.DataFrame({
        "experiment_type": ["expression-noise-v1", "parameter-distortion-v2"],
        "model_name": ["sy_simple", "sy_simple"]
    })
    
    filtered_df = filter_tasks_by_experiment(df, None)
    assert len(filtered_df) == 2, "None experiment_types should return all tasks"
    
    print("✅ Edge cases test passed")
    return True


def test_folder_name_generation():
    """Test that folder names are generated correctly with underscore conversion"""
    print("🧪 Testing folder name generation...")
    
    # Test cases: (model_name, exp_type, expected_folder_name)
    test_cases = [
        ("sy_simple", "expression_noise_v1", "sy_simple_expression_noise_v1"),
        ("sy_simple", "expression-noise-v1", "sy_simple_expression_noise_v1"),  # Hyphen converted
        ("fgfr4_model", "parameter_distortion_v2", "fgfr4_model_parameter_distortion_v2"),
        ("fgfr4_model", "parameter-distortion-v2", "fgfr4_model_parameter_distortion_v2"),
        ("test_model", "response-noise-v1", "test_model_response_noise_v1"),
    ]
    
    for model_name, exp_type, expected_folder in test_cases:
        # Simulate what discover_task_lists does
        folder_name = f"{model_name}_{exp_type.replace('-', '_')}"
        assert folder_name == expected_folder, f"Failed: {model_name}_{exp_type} -> {folder_name}, expected {expected_folder}"
        print(f"  ✅ {model_name}_{exp_type} -> {folder_name}")
    
    print("✅ Folder name generation test passed")
    return True


def main():
    """Run all tests"""
    print("="*70)
    print("🔬 Testing Hyphen/Underscore Fix for ML Batch Runner")
    print("="*70)
    
    tests = [
        test_hyphen_underscore_conversion_logic,
        test_discover_task_lists_hyphen_fix,
        test_filter_tasks_by_experiment_hyphen_fix,
        test_end_to_end_workflow,
        test_edge_cases,
        test_folder_name_generation,
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
            print(f"❌ {test_func.__name__} raised exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()
    
    print("="*70)
    print(f"Test results: {passed} passed, {failed} failed")
    print("="*70)
    
    if failed == 0:
        print("🎉 All hyphen/underscore fix tests passed!")
        print("\n📋 Summary of fixes validated:")
        print("1. ✅ discover_task_lists() converts hyphens to underscores for folder paths")
        print("2. ✅ discover_task_lists() handles hyphen/underscore mismatch when filtering CSV rows")
        print("3. ✅ filter_tasks_by_experiment() normalizes both CSV and config values for comparison")
        print("4. ✅ Folder names are generated with underscores (matching data-eng scripts)")
        print("5. ✅ End-to-end workflow handles the mismatch transparently")
        print("6. ✅ Edge cases are handled correctly")
        
        print("\n💡 The fix ensures compatibility between:")
        print("   - Data-eng scripts: Write 'expression-noise-v1' (hyphens) to CSV")
        print("   - Data-eng scripts: Create 'sy_simple_expression_noise_v1' (underscores) folders")
        print("   - ML runner: Can be configured with 'expression_noise_v1' (underscores)")
        print("   - ML runner: Automatically handles the conversion")
        
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
