"""Simple test for discover_task_lists functionality without complex imports"""

import sys
import os
import pandas as pd
from unittest.mock import Mock

# Add the run-ml-batch-v1.py directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
ml_batch_path = os.path.join(current_dir, "../notebooks/ch5-paper/machine-learning")
sys.path.insert(0, ml_batch_path)

# Now we can import the function directly
sys.path.insert(0, os.path.dirname(ml_batch_path))

# Try to import
import run_ml_batch_v1


def test_discover_task_lists_basic():
    """Test basic discovery with mock S3 data"""
    # Get the function
    discover_task_lists = run_ml_batch_v1.discover_task_lists
    
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
    
    print("✓ Basic test passed")


def test_discover_task_lists_folder_name_pattern():
    """Verify folder name construction pattern"""
    # Get the function
    discover_task_lists = run_ml_batch_v1.discover_task_lists
    
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
    
    print("✓ Folder name pattern test passed")


if __name__ == "__main__":
    print("Testing discover_task_lists functionality...")
    
    try:
        test_discover_task_lists_basic()
        test_discover_task_lists_folder_name_pattern()
        
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
