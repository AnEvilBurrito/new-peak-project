"""Check dynamic features for extreme values across distortion factors"""

import pandas as pd
import numpy as np
import sys
import os

# Add src to path for imports - updated for src/tests location
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "..")
sys.path.insert(0, src_dir)

from models.utils.s3_config_manager import S3ConfigManager

def check_dynamic_features(distortion_factor, s3_manager, base_path):
    """Check dynamic features for a given distortion factor"""
    dyn_path = f"{base_path}/data/sy_simple_parameter_distortion_v2/distortion_{distortion_factor}/dynamic_features.pkl"
    
    try:
        dyn_features = s3_manager.load_data_from_path(dyn_path, data_format="pkl")
        if isinstance(dyn_features, pd.DataFrame):
            print(f"\nDistortion {distortion_factor} dynamic features:")
            print(f"  Shape: {dyn_features.shape}")
            
            # Check for extreme values
            X_values = dyn_features.values
            print(f"  Max: {X_values.max():.6g}")
            print(f"  Min: {X_values.min():.6g}")
            print(f"  Mean: {X_values.mean():.6g}")
            
            # Check for values that might cause float32 issues
            float32_max = np.finfo(np.float32).max
            large_count = np.sum(np.abs(X_values) > 1e6)
            if large_count > 0:
                print(f"  ⚠️  Large values (>1e6): {large_count}")
            
            if X_values.max() > float32_max:
                print(f"  ❌ Max exceeds float32 max ({float32_max:.6g})")
            elif X_values.max() > 1e10:
                print(f"  ⚠️  Very large max value (>1e10)")
            
            # Check column-wise extremes
            extreme_cols = []
            for col in dyn_features.columns:
                col_data = dyn_features[col]
                if pd.api.types.is_numeric_dtype(col_data):
                    col_max = col_data.max()
                    if abs(col_max) > 1e6:
                        extreme_cols.append((col, col_max))
            
            if extreme_cols:
                print(f"  Extreme columns (max > 1e6):")
                for col, val in extreme_cols[:5]:  # Show first 5
                    print(f"    {col}: {val:.6g}")
                if len(extreme_cols) > 5:
                    print(f"    ... and {len(extreme_cols)-5} more")
                    
    except Exception as e:
        print(f"❌ Error loading dynamic_features for distortion_{distortion_factor}: {e}")

def main():
    s3 = S3ConfigManager()
    base_path = s3.save_result_path
    
    distortion_factors = [0, 1.1, 1.3, 1.5, 2.0, 3.0]
    
    print("=" * 70)
    print("CHECKING DYNAMIC FEATURES FOR EXTREME VALUES")
    print("=" * 70)
    
    for df in distortion_factors:
        check_dynamic_features(df, s3, base_path)

if __name__ == "__main__":
    main()
