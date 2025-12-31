"""Test script to reproduce the ML error with parameter-distortion-v2 data"""

import pandas as pd
import numpy as np
import sys
import os

# Add src to path for imports - adjusted for src/tests location
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "..")
sys.path.insert(0, src_dir)

from models.utils.s3_config_manager import S3ConfigManager
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def test_single_dataset(feature_path, target_path):
    """Test ML on a single feature-target pair"""
    print(f"\n{'='*60}")
    print(f"Testing: {feature_path}")
    print(f"Target: {target_path}")
    print(f"{'='*60}")
    
    s3 = S3ConfigManager()
    
    # Load data
    X = s3.load_data_from_path(feature_path, data_format="pkl")
    y_df = s3.load_data_from_path(target_path, data_format="pkl")
    
    # Ensure y is a Series
    if isinstance(y_df, pd.DataFrame):
        y = y_df.iloc[:, 0]
    else:
        y = y_df
    
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"X dtype: {X.dtypes}")
    print(f"y dtype: {y.dtype}")
    
    # Check for problematic values
    X_values = X.values
    print(f"X - NaN: {np.isnan(X_values).sum()}, Inf: {np.isinf(X_values).sum()}")
    print(f"X - Max: {X_values.max()}, Min: {X_values.min()}")
    
    y_values = y.values
    print(f"y - NaN: {np.isnan(y_values).sum()}, Inf: {np.isinf(y_values).sum()}")
    print(f"y - Max: {y_values.max()}, Min: {y_values.min()}")
    
    # Align indices
    common_idx = X.index.intersection(y.index)
    X_aligned = X.loc[common_idx]
    y_aligned = y.loc[common_idx]
    print(f"After alignment: X shape {X_aligned.shape}, y shape {y_aligned.shape}")
    
    # Try to convert to float32 (what sklearn does internally)
    try:
        X_float32 = X_aligned.astype(np.float32)
        print("✅ Successfully converted X to float32")
        print(f"float32 X - Max: {X_float32.values.max()}, Min: {X_float32.values.min()}")
    except Exception as e:
        print(f"❌ Error converting X to float32: {e}")
    
    # Build the same pipeline as in Workflow.py
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Try train/test split and fit
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_aligned, y_aligned, test_size=0.2, random_state=42
        )
        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        # Fit model
        print("Fitting RandomForest...")
        pipeline.fit(X_train, y_train)
        print("✅ Model fitted successfully!")
        
        # Score
        score = pipeline.score(X_test, y_test)
        print(f"Model R2 score: {score:.4f}")
        
    except Exception as e:
        print(f"❌ Error during model fitting: {e}")
        import traceback
        traceback.print_exc()
        
        # Additional diagnostic: check if any columns have extreme variance
        print("\nColumn statistics:")
        for col in X_aligned.columns:
            col_data = X_aligned[col]
            if pd.api.types.is_numeric_dtype(col_data):
                col_mean = col_data.mean()
                col_std = col_data.std()
                col_max = col_data.max()
                col_min = col_data.min()
                print(f"  {col}: mean={col_mean:.6g}, std={col_std:.6g}, max={col_max:.6g}, min={col_min:.6g}")

def main():
    s3 = S3ConfigManager()
    base_path = s3.save_result_path
    
    # Test with distortion_0 datasets (no parameter distortion)
    test_cases = [
        ("features.pkl", "targets.pkl"),
        ("dynamic_features.pkl", "targets.pkl"),
        ("last_time_points.pkl", "targets.pkl"),
    ]
    
    for feature_file, target_file in test_cases:
        feature_path = f"{base_path}/data/sy_simple_parameter_distortion_v2/distortion_0/{feature_file}"
        target_path = f"{base_path}/data/sy_simple_parameter_distortion_v2/distortion_0/{target_file}"
        test_single_dataset(feature_path, target_path)
    
    print(f"\n{'='*60}")
    print("Testing with distortion_1.1 (with parameter distortion)")
    print(f"{'='*60}")
    
    # Also test with distortion_1.1
    feature_path = f"{base_path}/data/sy_simple_parameter_distortion_v2/distortion_1.1/features.pkl"
    target_path = f"{base_path}/data/sy_simple_parameter_distortion_v2/distortion_1.1/targets.pkl"
    test_single_dataset(feature_path, target_path)

if __name__ == "__main__":
    main()
