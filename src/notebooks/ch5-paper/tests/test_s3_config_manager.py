#!/usr/bin/env python3
"""
Test script for S3ConfigManager functionality.
This script tests the basic operations of the S3ConfigManager class.
"""

# %%
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# Add the models path to import S3ConfigManager
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../models'))
from utils.s3_config_manager import S3ConfigManager, test_s3_connection

# %%
print("🧪 Testing S3ConfigManager...")

# Test S3 connection first
print("\n1. Testing S3 Connection...")
if test_s3_connection():
    print("✅ Connection test passed")
else:
    print("❌ Connection test failed - check environment variables")
    sys.exit(1)

# %%
# Initialize S3ConfigManager
print("\n2. Initializing S3ConfigManager...")
try:
    manager = S3ConfigManager()
    print("✅ S3ConfigManager initialized successfully")
except Exception as e:
    print(f"❌ S3ConfigManager initialization failed: {e}")
    sys.exit(1)

# %%
# Create test notebook configuration
print("\n3. Creating test notebook configuration...")
test_config = {
    'notebook_name': 'test-experiment',
    'exp_number': '99',
    'version_number': 'v1',
    'section_number': '00',
    'description': 'Test experiment for S3ConfigManager validation'
}

# Test data for various operations
print("✅ Test configuration created")
print(f"Test config: {test_config}")

# %%
# Test 1: Save and load configuration
print("\n4. Testing configuration save/load...")
try:
    # Create test configuration data
    config_data = {
        'experiment_params': {
            'model_type': 'test_model',
            'sample_size': 1000,
            'drug_concentration': [0.1, 1.0, 10.0],
            'time_points': [0, 1, 2, 3, 4]
        },
        'ml_settings': {
            'algorithm': 'random_forest',
            'n_estimators': 100,
            'test_size': 0.2
        }
    }
    
    # Save configuration
    manager.save_config(test_config, config_data)
    print("✅ Configuration saved to S3")
    
    # Load configuration
    loaded_config = manager.load_config(test_config)
    print("✅ Configuration loaded from S3")
    
    # Verify configuration matches
    assert loaded_config == config_data, "Configuration does not match after save/load"
    print("✅ Configuration validation passed")
    
except Exception as e:
    print(f"❌ Configuration test failed: {e}")

# %%
# Test 2: Save and load data (pickle format)
print("\n5. Testing pickle data save/load...")
try:
    # Create test data (numpy array)
    test_data = {
        'features': np.random.rand(100, 5),
        'targets': np.random.rand(100),
        'metadata': {'created': '2025-12-15', 'test': True}
    }
    
    # Save data
    manager.save_data(test_config, test_data, 'test_pickle_data', 'pkl')
    print("✅ Pickle data saved to S3")
    
    # Load data
    loaded_data = manager.load_data(test_config, 'test_pickle_data', 'pkl')
    print("✅ Pickle data loaded from S3")
    
    # Verify data matches
    assert np.array_equal(loaded_data['features'], test_data['features'])
    assert np.array_equal(loaded_data['targets'], test_data['targets'])
    assert loaded_data['metadata'] == test_data['metadata']
    print("✅ Pickle data validation passed")
    
except Exception as e:
    print(f"❌ Pickle data test failed: {e}")

# %%
# Test 3: Save and load data (CSV format)
print("\n6. Testing CSV data save/load...")
try:
    # Create test DataFrame
    test_df = pd.DataFrame({
        'feature_1': np.random.rand(50),
        'feature_2': np.random.rand(50),
        'target': np.random.randint(0, 2, 50)
    })
    
    # Save data
    manager.save_data(test_config, test_df, 'test_csv_data', 'csv')
    print("✅ CSV data saved to S3")
    
    # Load data
    loaded_df = manager.load_data(test_config, 'test_csv_data', 'csv')
    print("✅ CSV data loaded from S3")
    
    # Verify data matches with relaxed comparison for CSV data
    # CSV round-trip may cause dtype differences (e.g., int64 vs int32) which are acceptable
    pd.testing.assert_frame_equal(loaded_df, test_df, check_dtype=False)
    print("✅ CSV data validation passed")
    
except Exception as e:
    print(f"❌ CSV data test failed: {e}")

# %%
# Test 4: Load data from direct path
print("\n7. Testing load_data_from_path...")
try:
    # First, save some test data using the standard method to get a known S3 key
    test_direct_data = {
        'direct_test_key': 'direct_test_value',
        'numbers': [1, 2, 3, 4, 5],
        'nested': {'a': 1, 'b': 2}
    }
    
    # Save using save_data to generate the S3 key
    manager.save_data(test_config, test_direct_data, 'test_direct_data', 'pkl')
    print("✅ Test data saved for direct path test")
    
    # Construct the expected S3 key using the same logic as save_data/load_data
    version_number = test_config.get('version_number', 'v1')
    filename = f"{version_number}_test_direct_data.pkl"
    # We need to replicate the _get_s3_key logic
    section_number = test_config.get('section_number', '00')
    exp_number = test_config.get('exp_number')
    notebook_name = test_config.get('notebook_name')
    base_path = f"{manager.save_result_path}/{section_number}_{exp_number}_{version_number}_{notebook_name}"
    s3_key = f"{base_path}/data/{filename}"
    
    print(f"Expected S3 key: {s3_key}")
    
    # Now load using the new direct path method
    loaded_direct_data = manager.load_data_from_path(s3_key, 'pkl')
    print("✅ Data loaded using direct S3 key")
    
    # Verify data matches
    assert loaded_direct_data['direct_test_key'] == test_direct_data['direct_test_key']
    assert loaded_direct_data['numbers'] == test_direct_data['numbers']
    assert loaded_direct_data['nested'] == test_direct_data['nested']
    print("✅ Direct path load validation passed")
    
    # Also test CSV format
    test_direct_df = pd.DataFrame({
        'col1': [10, 20, 30],
        'col2': ['a', 'b', 'c']
    })
    
    manager.save_data(test_config, test_direct_df, 'test_direct_csv', 'csv')
    csv_filename = f"{version_number}_test_direct_csv.csv"
    csv_key = f"{base_path}/data/{csv_filename}"
    
    loaded_direct_df = manager.load_data_from_path(csv_key, 'csv')
    pd.testing.assert_frame_equal(loaded_direct_df, test_direct_df, check_dtype=False)
    print("✅ CSV direct path load validation passed")
    
    # Test txt format if supported
    test_text = "This is a test text file content.\nWith multiple lines.\nEnd of file."
    manager.save_data(test_config, test_text, 'test_direct_txt', 'txt')
    txt_filename = f"{version_number}_test_direct_txt.txt"
    txt_key = f"{base_path}/data/{txt_filename}"
    
    loaded_text = manager.load_data_from_path(txt_key, 'txt')
    assert loaded_text == test_text
    print("✅ TXT direct path load validation passed")
    
    print("✅ All direct path load tests passed")
    
except Exception as e:
    print(f"❌ Direct path load test failed: {e}")

# %%
# Test 6: Save figure
print("\n8. Testing figure save...")
try:
    # Create test figure
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.plot(x, y, label='sin(x)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Test Figure')
    ax.legend()
    ax.grid(True)
    
    # Save figure
    manager.save_figure(test_config, fig, 'test_figure', 'png', dpi=150)
    print("✅ Figure saved to S3")
    
    # Close figure to free memory
    plt.close(fig)
    print("✅ Figure test passed")
    
except Exception as e:
    print(f"❌ Figure test failed: {e}")

# %%
# Test 7: List experiment files
print("\n9. Testing file listing...")
try:
    files = manager.list_experiment_files(test_config)
    print("✅ File listing completed")
    
    print(f"Config files: {files['config']}")
    print(f"Data files: {files['data']}")
    print(f"Figure files: {files['figures']}")
    
    # Verify expected files exist (including those created by direct path tests)
    expected_files = [
        'v1_config.yml',
        'v1_test_pickle_data.pkl',
        'v1_test_csv_data.csv',
        'v1_test_figure.png',
        'v1_test_direct_data.pkl',
        'v1_test_direct_csv.csv',
        'v1_test_direct_txt.txt'
    ]
    
    found_files = []
    for category in files.values():
        for file_path in category:
            filename = file_path.split('/')[-1]
            found_files.append(filename)
    
    for expected_file in expected_files:
        assert expected_file in found_files, f"Expected file {expected_file} not found"
    
    print("✅ File listing validation passed")
    
except Exception as e:
    print(f"❌ File listing test failed: {e}")

# %%
# Test 8: Cleanup (optional - comment out to keep test files)
print("\n10. Testing cleanup (optional)...")
try:
    # Uncomment the line below to actually delete test files
    # manager.delete_experiment_files(test_config)
    print("⚠️  Cleanup skipped (files preserved for verification)")
    # print("✅ Test files deleted from S3")
    
except Exception as e:
    print(f"❌ Cleanup test failed: {e}")

# %%
print("\n" + "="*50)
print("🎉 All tests completed!")
print("="*50)

print("\nTest files created in S3:")
print(f"Bucket: {manager.bucket_name}")
print(f"Base path: {manager.save_result_path}")
print(f"Experiment: {test_config['section_number']}_{test_config['exp_number']}_{test_config['version_number']}_{test_config['notebook_name']}")

print("\nTo verify manually, check your S3 bucket for the test files.")
print("To clean up test files, uncomment the delete_experiment_files call in the test.")
