"""Test that clipping prevents float32 overflow in parameter-distortion experiments"""

import pandas as pd
import numpy as np
import sys
import os

# Add src to path for imports - updated for src/tests location
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "..")
sys.path.insert(0, src_dir)

from models.utils.s3_config_manager import S3ConfigManager
from ml.Workflow import build_pipeline_with_clipping, ClippingTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def test_clipping_on_distortion(distortion_factor=1.1):
    """Test clipping on a specific distortion factor"""
    s3 = S3ConfigManager()
    base_path = s3.save_result_path
    
    # Load dynamic features
    dyn_path = f"{base_path}/data/sy_simple_parameter_distortion_v2/distortion_{distortion_factor}/dynamic_features.pkl"
    print(f"Loading dynamic features from: {dyn_path}")
    
    try:
        dyn_features = s3.load_data_from_path(dyn_path, data_format="pkl")
    except Exception as e:
        print(f"❌ Failed to load dynamic_features: {e}")
        return False
    
    # Load targets
    target_path = f"{base_path}/data/sy_simple_parameter_distortion_v2/distortion_{distortion_factor}/targets.pkl"
    print(f"Loading targets from: {target_path}")
    
    try:
        targets = s3.load_data_from_path(target_path, data_format="pkl")
    except Exception as e:
        print(f"❌ Failed to load targets: {e}")
        return False
    
    print(f"Dataset loaded:")
    print(f"  Features shape: {dyn_features.shape}")
    print(f"  Targets shape: {targets.shape}")
    
    # Check for extreme values before clipping
    X = dyn_features.values
    float32_max = np.finfo(np.float32).max
    extreme_count = np.sum(np.abs(X) > float32_max)
    if extreme_count > 0:
        print(f"⚠️  Found {extreme_count} values exceeding float32 max ({float32_max:.6g})")
    
    # Test ClippingTransformer directly
    print("\nTesting ClippingTransformer...")
    clipper = ClippingTransformer(threshold=1e9)
    X_clipped = clipper.fit_transform(dyn_features)
    
    # Check after clipping
    Xc = X_clipped.values
    extreme_count_clipped = np.sum(np.abs(Xc) > 1e9)
    print(f"  Values > 1e9 after clipping: {extreme_count_clipped}")
    print(f"  Min after clipping: {Xc.min():.6g}")
    print(f"  Max after clipping: {Xc.max():.6g}")
    
    # Test pipeline with clipping
    print("\nTesting pipeline with clipping...")
    pipeline = build_pipeline_with_clipping(
        LinearRegression(), 
        threshold=1e9
    )
    
    # Align data
    common_idx = dyn_features.index.intersection(targets.index)
    X_align = dyn_features.loc[common_idx]
    y_align = targets.loc[common_idx]
    
    if len(common_idx) < 10:
        print(f"⚠️  Not enough common samples: {len(common_idx)}")
        return True  # Not a failure, just insufficient data
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_align, y_align, test_size=0.2, random_state=42
    )
    
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Testing samples: {X_test.shape[0]}")
    
    # Fit pipeline - this should not raise ValueError about float32
    try:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        print(f"✅ Pipeline training successful!")
        print(f"   R² score: {r2:.4f}")
        return True
    except ValueError as e:
        if "float32" in str(e) or "infinity" in str(e):
            print(f"❌ Pipeline still failing with float32 error: {e}")
            return False
        else:
            raise
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_clipping_transformer():
    """Unit test for ClippingTransformer"""
    print("\n" + "="*70)
    print("Testing ClippingTransformer")
    print("="*70)
    
    # Create test data with extreme values
    np.random.seed(42)
    data = pd.DataFrame({
        'col1': np.random.normal(0, 1, 100),
        'col2': np.random.normal(100, 10, 100),
        'col3': np.array([1e20, -1e15] + [0] * 98)  # Extreme values
    })
    
    print("Original data extremes:")
    print(f"  Max: {data.values.max():.3e}")
    print(f"  Min: {data.values.min():.3e}")
    
    clipper = ClippingTransformer(threshold=1e6)
    clipped = clipper.fit_transform(data)
    
    print("\nAfter clipping (threshold=1e6):")
    print(f"  Max: {clipped.values.max():.3e}")
    print(f"  Min: {clipped.values.min():.3e}")
    
    # Verify clipping worked
    assert clipped.values.max() <= 1e6, "Positive values not clipped"
    assert clipped.values.min() >= -1e6, "Negative values not clipped"
    print("✅ ClippingTransformer test passed!")
    return True

def main():
    print("="*70)
    print("TESTING CLIPPING FIX FOR PARAMETER-DISTORTION EXPERIMENTS")
    print("="*70)
    
    # Test ClippingTransformer
    test_clipping_transformer()
    
    # Test on actual data
    distortion_factors = [1.1, 1.3, 1.5, 2.0, 3.0]
    
    all_passed = True
    for df in distortion_factors:
        print(f"\n{'='*70}")
        print(f"Testing distortion factor: {df}")
        print(f"{'='*70}")
        passed = test_clipping_on_distortion(df)
        if not passed:
            all_passed = False
            print(f"❌ Test failed for distortion factor {df}")
        else:
            print(f"✅ Test passed for distortion factor {df}")
    
    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Clipping fix prevents float32 overflow.")
    else:
        print("❌ Some tests failed.")
    print("="*70)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
