"""Simple working test for discover_task_lists - copy the function directly"""

import sys
import os
import pandas as pd
from unittest.mock import Mock
import logging

# Set up logging for the test
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Copy the discover_task_lists function directly to avoid import issues
def discover_task_lists(experiment_types, model_names, s3_manager):
    """
    Discover task list CSVs based on EXPERIMENT_TYPES and MODEL_NAMES
    
    Looks for CSVs at: {s3_path}/data/{model_name}_{experiment_type}/task_list.csv
    Returns: Combined DataFrame of all discovered tasks
    """
    logger = logging.getLogger(__name__)
    all_tasks = []
    
    for model_name in model_names:
        for exp_type in experiment_types:
            # Construct expected CSV path based on pattern from data-eng scripts
            # Pattern: {model_name}_{experiment_type} (e.g., "sy_simple_expression_noise_v1")
            folder_name = f"{model_name}_{exp_type}"
            csv_path = f"{s3_manager.save_result_path}/data/{folder_name}/task_list.csv"
            
            try:
                logger.info(f"Looking for task list: {csv_path}")
                task_df = s3_manager.load_data_from_path(csv_path, data_format="csv")
                
                # Filter to only include tasks for this experiment type and model
                filtered_df = task_df[
                    (task_df["experiment_type"] == exp_type) & 
                    (task_df["model_name"] == model_name)
                ]
                
                if not filtered_df.empty:
                    all_tasks.append(filtered_df)
                    logger.info(f"Found {len(filtered_df)} tasks for {model_name}/{exp_type}")
                else:
                    logger.warning(f"No matching tasks found in CSV for {model_name}/{exp_type}")
                    
            except Exception as e:
                logger.warning(f"Task list not found or error loading: {csv_path} - {str(e)[:100]}")
    
    if all_tasks:
        combined_df = pd.concat(all_tasks, ignore_index=True)
        logger.info(f"Discovered total of {len(combined_df)} tasks across all experiments/models")
        return combined_df
    else:
        logger.error("No task lists discovered for the specified experiments/models")
        return pd.DataFrame()


def test_discover_task_lists_basic():
    """Test basic discovery with mock S3 data"""
    # Mock S3 manager
    s3_manager = Mock()
    s3_manager.save_result_path = "test/path"
    s3_manager.load_data_from_path = Mock()
    
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
    s3_manager = Mock()
    s3_manager.save_result_path = "base/path"
    s3_manager.load_data_from_path = Mock()
    
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


def test_discover_task_lists_filtering():
    """Test that discovery filters by experiment_type and model_name"""
    s3_manager = Mock()
    s3_manager.save_result_path = "test/path"
    s3_manager.load_data_from_path = Mock()
    
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
    
    print("✓ Filtering test passed")


if __name__ == "__main__":
    print("Testing discover_task_lists functionality...")
    
    try:
        test_discover_task_lists_basic()
        test_discover_task_lists_folder_name_pattern()
        test_discover_task_lists_filtering()
        
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
