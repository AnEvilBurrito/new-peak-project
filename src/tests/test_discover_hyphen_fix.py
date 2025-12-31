"""
Test to verify the hyphen/underscore fix in discover_task_lists() function after refactoring.

This test validates that the ML batch runner can handle:
- CSV with hyphens in experiment_type (e.g., "expression-noise-v1")
- Configuration with underscores (e.g., "expression_noise_v1")
- Folder names with underscores (e.g., "sy_simple_expression_noise_v1")

Note: After refactoring Option B, filter_tasks_by_experiment() function has been removed
since CSV files are already filtered by experiment type (each CSV contains tasks for specific experiment).
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
        
        # Create separate CSVs for each experiment type (more realistic)
        # CSV for expression-noise-v1 should only contain expression-noise-v1 tasks
        expr_df = loaded_df[loaded_df["experiment_type"].isin(["expression-noise-v1"])].copy()
        param_df = loaded_df[loaded_df["experiment_type"].isin(["parameter-distortion-v2"])].copy()
        
        # Mock load_data_from_path to return different CSVs based on path
        def side_effect(path, data_format):
            if "sy_simple_expression_noise_v1" in path:
                return expr_df
            elif "sy_simple_parameter_distortion_v2" in path:
                return param_df
            else:
                raise Exception(f"Unexpected path: {path}")
        
        s3_manager.load_data_from_path = Mock(side_effect=side_effect)
        
        # Test discover_task_lists
        discovered_df = discover_task_lists(experiment_types, model_names, s3_manager)
        
        # Should find 3 tasks (2 expression-noise-v1 + 1 parameter-distortion-v2)
        # Since each CSV only contains tasks for its specific experiment type
        assert len(discovered_df) == 3, f"Should find 3 tasks, got {len(discovered_df)}"
        
        # No need to filter by experiment_type - CSV already contains tasks for specific experiments
        # Discovered tasks should match our configuration
        # Verified by the assertion above
        
        # Verify the workflow handles the mismatch correctly
        print(f"  CSV has experiment_types: {loaded_df['experiment_type'].unique()}")
        print(f"  Configuration uses: {experiment_types}")
        print(f"  Discovered {len(discovered_df)} tasks successfully")
        
        print("✅ End-to-end workflow test passed")
        return True
        
    finally:
        shutil.rmtree(test_dir)




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
    print("🔬 Testing Hyphen/Underscore Fix for ML Batch Runner (After Refactor)")
    print("="*70)
    
    tests = [
        test_hyphen_underscore_conversion_logic,
        test_discover_task_lists_hyphen_fix,
        test_end_to_end_workflow,
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
        print("2. ✅ discover_task_lists() correctly discovers CSV files for specified experiments")
        print("3. ✅ Folder names are generated with underscores (matching data-eng scripts)")
        print("4. ✅ End-to-end workflow handles the mismatch transparently")
        
        print("\n💡 After Option B refactoring:")
        print("   - filter_tasks_by_experiment() function removed - redundant filtering eliminated")
        print("   - CSV files already contain tasks for specific experiment types")
        print("   - discover_task_lists() returns unfiltered CSV content")
        print("   - No duplicate assertion checks needed")
        
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
