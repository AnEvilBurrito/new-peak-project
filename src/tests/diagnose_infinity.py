"""Diagnostic script to identify infinity/large values in parameter-distortion-v2 datasets"""

import pandas as pd
import numpy as np
import sys
import os

# Add src to path for imports - adjusted for src/tests location
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "..")
sys.path.insert(0, src_dir)

try:
    from models.utils.s3_config_manager import S3ConfigManager
    S3_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import S3ConfigManager: {e}")
    S3_AVAILABLE = False

def inspect_dataset(s3_path, s3_manager):
    """Load and inspect a dataset from S3"""
    try:
        print(f"\n{'='*60}")
        print(f"Inspecting: {s3_path}")
        print(f"{'='*60}")
        
        data = s3_manager.load_data_from_path(s3_path, data_format="pkl")
        
        if isinstance(data, pd.DataFrame):
            print(f"Type: DataFrame, Shape: {data.shape}")
            print(f"Columns: {list(data.columns)}")
            values = data.values
        elif isinstance(data, np.ndarray):
            print(f"Type: numpy array, Shape: {data.shape}")
            values = data
        elif isinstance(data, (list, dict)):
            print(f"Type: {type(data).__name__}, Length: {len(data)}")
            # Convert to array for analysis
            if isinstance(data, list):
                values = np.array(data)
            else:
                values = np.array(list(data.values()))
        else:
            print(f"Type: {type(data).__name__}")
            return
        
        # Check for problematic values
        total_elements = values.size if hasattr(values, 'size') else len(values)
        
        # Check NaN
        nan_count = np.isnan(values).sum() if hasattr(values, 'size') else 0
        print(f"NaN values: {nan_count}/{total_elements} ({nan_count/total_elements*100:.2f}%)")
        
        # Check Infinity
        inf_count = np.isinf(values).sum() if hasattr(values, 'size') else 0
        print(f"Infinite values: {inf_count}/{total_elements} ({inf_count/total_elements*100:.2f}%)")
        
        # Check float32 limits
        if hasattr(values, 'dtype'):
            print(f"Data dtype: {values.dtype}")
        
        if hasattr(values, 'max'):
            max_val = values.max()
            min_val = values.min()
            print(f"Max value: {max_val}")
            print(f"Min value: {min_val}")
            
            # Check if values exceed float32 positive range
            float32_max = np.finfo(np.float32).max
            float32_min = np.finfo(np.float32).min
            
            if max_val > float32_max:
                print(f"⚠️  WARNING: Max value exceeds float32 max ({float32_max})")
            if min_val < float32_min:
                print(f"⚠️  WARNING: Min value exceeds float32 min ({float32_min})")
            
            # Check for extremely large values
            large_threshold = 1e30
            large_count = np.sum(np.abs(values) > large_threshold) if hasattr(values, 'size') else 0
            print(f"Values > {large_threshold}: {large_count}")
            
            # Calculate statistics
            if isinstance(data, pd.DataFrame):
                print("\nColumn-wise statistics:")
                for col in data.columns:
                    col_data = data[col]
                    if pd.api.types.is_numeric_dtype(col_data):
                        col_max = col_data.max()
                        col_min = col_data.min()
                        col_mean = col_data.mean()
                        print(f"  {col}: max={col_max:.6g}, min={col_min:.6g}, mean={col_mean:.6g}")
        
        # Sample data
        if isinstance(data, pd.DataFrame):
            print("\nFirst 3 rows:")
            print(data.head(3))
        elif hasattr(data, '__len__') and len(data) > 0:
            print(f"\nFirst 3 elements: {data[:3] if len(data) > 3 else data}")
            
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main diagnostic function"""
    if not S3_AVAILABLE:
        print("S3ConfigManager not available. Trying to load from local files...")
        # Try local path fallback
        # For now, just exit
        print("Cannot proceed without S3 access.")
        return
    
    # Initialize S3 manager
    s3_manager = S3ConfigManager()
    
    # Base path from configuration
    base_path = s3_manager.save_result_path
    print(f"S3 base path: {base_path}")
    
    # Test datasets from failed tasks list
    test_datasets = [
        # Distortion 0 datasets
        f"{base_path}/data/sy_simple_parameter_distortion_v2/distortion_0/features.pkl",
        f"{base_path}/data/sy_simple_parameter_distortion_v2/distortion_0/dynamic_features.pkl",
        f"{base_path}/data/sy_simple_parameter_distortion_v2/distortion_0/last_time_points.pkl",
        f"{base_path}/data/sy_simple_parameter_distortion_v2/distortion_0/targets.pkl",
        
        # Distortion 1.1 datasets (for comparison)
        f"{base_path}/data/sy_simple_parameter_distortion_v2/distortion_1.1/features.pkl",
    ]
    
    print("🔍 Starting diagnostic analysis of parameter-distortion-v2 datasets")
    print("Focusing on distortion_0 first (no parameter distortion)")
    
    for dataset_path in test_datasets:
        inspect_dataset(dataset_path, s3_manager)
    
    # Additional analysis: check dynamic feature calculation
    print(f"\n{'='*60}")
    print("DYNAMIC FEATURE CALCULATION ANALYSIS")
    print(f"{'='*60}")
    
    # Try to load timecourse data to understand source
    timecourse_path = f"{base_path}/data/sy_simple_parameter_distortion_v2/distortion_0/timecourses.pkl"
    try:
        timecourse_data = s3_manager.load_data_from_path(timecourse_path, data_format="pkl")
        print(f"Timecourse data loaded: {type(timecourse_data)}")
        if isinstance(timecourse_data, pd.DataFrame):
            print(f"Timecourse shape: {timecourse_data.shape}")
            print(f"Timecourse columns: {list(timecourse_data.columns)[:10]}...")
            
            # Check if timecourse has extreme values
            if hasattr(timecourse_data, 'values'):
                tc_values = timecourse_data.values
                tc_inf = np.isinf(tc_values).sum()
                tc_nan = np.isnan(tc_values).sum()
                print(f"Timecourse - Infinite: {tc_inf}, NaN: {tc_nan}")
    except Exception as e:
        print(f"Could not load timecourse data: {e}")
    
    print(f"\n{'='*60}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
