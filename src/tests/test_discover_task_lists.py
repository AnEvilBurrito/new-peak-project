"""Test discover_task_lists functionality"""

import sys
import os
import pandas as pd
import pytest
from unittest.mock import Mock, patch

# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "..")
sys.path.insert(0, src_dir)

# Import from the correct path - the file is located at src/notebooks/ch5-paper/machine-learning/run-ml-batch-v1.py
# but Python doesn't handle hyphens in module names well, so we need to add the directory to path
notebooks_dir = os.path.join(src_dir, "notebooks")
sys.path.insert(0, notebooks_dir)

# Now we can import using the Python package structure
sys.path.insert(0, os.path.join(src_dir, "notebooks/ch5-paper/machine-learning"))


def test_discover_task_lists_basic():
    """Test basic discovery with mock S3 data"""
    # Mock S3 manager
    s3_manager = Mock()
    s3_manager.save_result_path = "test/path"
    
    # Mock CSV data
    mock_csv_data = pd.DataFrame({
        "feature_data": ["path/to/features.pkl"],
        "feature_data_label": ["test_features"],
        "target_data": ["path/to/targets.pkl"],
        "target_data_label": ["test_targets"],
        "experiment_type": ["expression-noise-v1"],
        "level": [0.5],
        "model_name": ["sy_simple"]
    })
    
    # Mock load_data_from_path to return our CSV
    s3_manager.load_data_from_path.return_value = mock_csv_data
    
    # Test discovery
    experiment_types = ["expression-noise-v1"]
    model_names = ["sy_simple"]
    
    result = discover_task_lists(experiment_types, model_names, s3_manager)
    
    # Verify S3 path construction
    expected_path = "test/path/data/sy_simple_expression-noise-v1/task_list.csv"
    s3_manager.load_data_from_path.assert_called_with(expected_path, data_format="csv")
    
    # Verify result
    assert len(result) == 1
    assert result.iloc[0]["model_name"] == "sy_simple"
    assert result.iloc[0]["experiment_type"] == "expression-noise-v1"


def test_discover_task_lists_multiple_experiments():
    """Test discovery with multiple experiment types"""
    s3_manager = Mock()
    s3_manager.save_result_path = "test/path"
    
    # Create mock data for two different experiments
    mock_data1 = pd.DataFrame({
        "experiment_type": ["expression-noise-v1"],
        "model_name": ["v1"],
        "level": [0.5]
    })
    
    mock_data2 = pd.DataFrame({
        "experiment_type": ["parameter-distortion-v2"],
        "model_name": ["v1"],
        "level": [0.3]
    })
    
    # Mock different returns for different paths
    def side_effect(path, data_format):
        if "v1_expression-noise-v1" in path:
            return mock_data1
        elif "v1_parameter-distortion-v2" in path:
            return mock_data2
        else:
            raise Exception("Path not found")
    
    s3_manager.load_data_from_path.side_effect = side_effect
    
    # Test with both experiment types
    experiment_types = ["expression-noise-v1", "parameter-distortion-v2"]
    model_names = ["v1"]
    
    result = discover_task_lists(experiment_types, model_names, s3_manager)
    
    # Should have 2 rows total
    assert len(result) == 2
    # Should have both experiment types
    assert set(result["experiment_type"].unique()) == {"expression-noise-v1", "parameter-distortion-v2"}


def test_discover_task_lists_missing_csv():
    """Test handling of missing CSV files"""
    s3_manager = Mock()
    s3_manager.save_result_path = "test/path"
    
    # Mock load to raise exception (file not found)
    s3_manager.load_data_from_path.side_effect = Exception("File not found")
    
    experiment_types = ["expression-noise-v1"]
    model_names = ["sy_simple"]
    
    result = discover_task_lists(experiment_types, model_names, s3_manager)
    
    # Should return empty DataFrame
    assert result.empty


def test_discover_task_lists_filtering():
    """Test that discovery filters by experiment_type and model_name"""
    s3_manager = Mock()
    s3_manager.save_result_path = "test/path"
    
    # Create CSV with multiple rows, some not matching our criteria
    mock_csv_data = pd.DataFrame({
        "feature_data": ["f1.pkl", "f2.pkl", "f3.pkl"],
        "feature_data_label": ["f1", "f2", "f3"],
        "target_data": ["t1.pkl", "t2.pkl", "t3.pkl"],
        "target_data_label": ["t1", "t2", "t3"],
        "experiment_type": ["expression-noise-v1", "other-experiment", "expression-noise-v1"],
        "level": [0.5, 0.3, 0.7],
        "model_name": ["sy_simple", "sy_simple", "other-model"]
    })
    
    s3_manager.load_data_from_path.return_value = mock_csv_data
    
    experiment_types = ["expression-noise-v1"]
    model_names = ["sy_simple"]
    
    result = discover_task_lists(experiment_types, model_names, s3_manager)
    
    # Should only return row 0 (matches both experiment_type and model_name)
    assert len(result) == 1
    assert result.iloc[0]["experiment_type"] == "expression-noise-v1"
    assert result.iloc[0]["model_name"] == "sy_simple"


def test_discover_task_lists_no_matches():
    """Test case where CSV has no matching rows"""
    s3_manager = Mock()
    s3_manager.save_result_path = "test/path"
    
    # CSV with no matching rows
    mock_csv_data = pd.DataFrame({
        "experiment_type": ["other-experiment"],
        "model_name": ["other-model"],
        "level": [0.5]
    })
    
    s3_manager.load_data_from_path.return_value = mock_csv_data
    
    experiment_types = ["expression-noise-v1"]
    model_names = ["sy_simple"]
    
    result = discover_task_lists(experiment_types, model_names, s3_manager)
    
    # Should log warning but return empty DataFrame
    assert result.empty


def test_discover_task_lists_folder_name_pattern():
    """Verify folder name construction pattern"""
    s3_manager = Mock()
    s3_manager.save_result_path = "base/path"
    
    mock_csv_data = pd.DataFrame({
        "experiment_type": ["test-exp-v1"],
        "model_name": ["test-model"],
        "level": [0.5]
    })
    
    s3_manager.load_data_from_path.return_value = mock_csv_data
    
    experiment_types = ["test-exp-v1"]
    model_names = ["test-model"]
    
    result = discover_task_lists(experiment_types, model_names, s3_manager)
    
    # Check that path was constructed correctly
    expected_path = "base/path/data/test-model_test-exp-v1/task_list.csv"
    s3_manager.load_data_from_path.assert_called_with(expected_path, data_format="csv")


if __name__ == "__main__":
    # Import here to avoid path issues
    # Add the parent directory to sys.path
    import sys
    import os
    
    # Get the directory of this test file
    test_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(test_dir, "../..")
    sys.path.insert(0, project_root)
    
    # Add the specific module directory
    ml_batch_dir = os.path.join(project_root, "src/notebooks/ch5-paper/machine-learning")
    sys.path.insert(0, ml_batch_dir)
    
    # Now import
    from run_ml_batch_v1 import discover_task_lists
    
    # Run tests
    test_discover_task_lists_basic()
    print("✓ Basic test passed")
    
    test_discover_task_lists_filtering()
    print("✓ Filtering test passed")
    
    test_discover_task_lists_folder_name_pattern()
    print("✓ Folder name pattern test passed")
    
    print("\nAll tests passed!")
