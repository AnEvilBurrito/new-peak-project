"""Analyze target values across distortion factors to identify extreme values"""

import pandas as pd
import numpy as np
import sys
import os

# Add src to path for imports - updated for src/tests location
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "..")
sys.path.insert(0, src_dir)

from models.utils.s3_config_manager import S3ConfigManager

def analyze_distortion_level(distortion_factor, s3_manager, base_path):
    """Analyze target values for a given distortion factor"""
    target_path = f"{base_path}/data/sy_simple_parameter_distortion_v2/distortion_{distortion_factor}/targets.pkl"
    
    try:
        targets = s3_manager.load_data_from_path(target_path, data_format="pkl")
        if isinstance(targets, pd.DataFrame):
            y = targets.iloc[:, 0]
        else:
            y = targets
            
        print(f"\nDistortion factor: {distortion_factor}")
        print(f"  Samples: {len(y)}")
        print(f"  Mean: {y.mean():.6g}")
        print(f"  Std: {y.std():.6g}")
        print(f"  Min: {y.min():.6g}")
        print(f"  Max: {y.max():.6g}")
        print(f"  Median: {y.median():.6g}")
        
        # Check for extreme values
        float32_max = np.finfo(np.float32).max
        large_count = np.sum(np.abs(y) > 1e6)  # Count values > 1 million
        if large_count > 0:
            print(f"  ⚠️  Large values (>1e6): {large_count}")
        
        if y.max() > float32_max:
            print(f"  ❌ Max exceeds float32 max ({float32_max:.6g})")
        elif y.max() > 1e10:
            print(f"  ⚠️  Very large max value (>1e10)")
        
        # Check distribution percentiles
        if len(y) > 0:
            percentiles = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]
            perc_values = np.percentile(y, percentiles)
            print(f"  Percentiles:")
            for p, v in zip(percentiles, perc_values):
                print(f"    {p}%: {v:.6g}")
        
    except Exception as e:
        print(f"❌ Error loading distortion_{distortion_factor}: {e}")

def main():
    s3 = S3ConfigManager()
    base_path = s3.save_result_path
    
    distortion_factors = [0, 1.1, 1.3, 1.5, 2.0, 3.0]
    
    print("=" * 70)
    print("ANALYSIS OF TARGET VALUES ACROSS DISTORTION FACTORS")
    print("=" * 70)
    
    for df in distortion_factors:
        analyze_distortion_level(df, s3, base_path)
    
    # Also check if feature datasets have extreme values for higher distortions
    print(f"\n{'='*70}")
    print("CHECKING FEATURE DATASETS FOR EXTREME VALUES")
    print(f"{'='*70}")
    
    for df in distortion_factors:
        if df == 0:
            continue  # Already checked
            
        feature_path = f"{base_path}/data/sy_simple_parameter_distortion_v2/distortion_{df}/features.pkl"
        try:
            features = s3.load_data_from_path(feature_path, data_format="pkl")
            if isinstance(features, pd.DataFrame):
                X_values = features.values
                print(f"\nDistortion {df} features:")
                print(f"  Shape: {features.shape}")
                print(f"  Max: {X_values.max():.6g}")
                print(f"  Min: {X_values.min():.6g}")
                print(f"  Mean: {X_values.mean():.6g}")
        except Exception as e:
            print(f"❌ Error loading features for distortion_{df}: {e}")

if __name__ == "__main__":
    main()
