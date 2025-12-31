"""Fix for extreme dynamic feature values by clipping to reasonable percentiles"""

import pandas as pd
import numpy as np

def clip_extreme_values(df, lower_percentile=0.1, upper_percentile=99.9):
    """
    Clip extreme values in a DataFrame based on percentiles.
    
    Args:
        df: DataFrame containing numeric values
        lower_percentile: Lower percentile for clipping (default: 0.1)
        upper_percentile: Upper percentile for clipping (default: 99.9)
        
    Returns:
        Clipped DataFrame
    """
    df_clipped = df.copy()
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Calculate percentiles
            lower = np.percentile(df[col], lower_percentile)
            upper = np.percentile(df[col], upper_percentile)
            
            # Clip values
            df_clipped[col] = df_clipped[col].clip(lower=lower, upper=upper)
            
            # Count clipped values
            clipped_lower = (df[col] < lower).sum()
            clipped_upper = (df[col] > upper).sum()
            
            if clipped_lower > 0 or clipped_upper > 0:
                print(f"  {col}: clipped {clipped_lower} lower, {clipped_upper} upper values")
    
    return df_clipped

def clip_extreme_values_robust(df, threshold=1e6):
    """
    Clip extreme values to a fixed threshold.
    
    Args:
        df: DataFrame containing numeric values
        threshold: Maximum absolute value allowed (default: 1e6)
        
    Returns:
        Clipped DataFrame
    """
    df_clipped = df.copy()
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Clip extreme positive and negative values
            df_clipped[col] = df_clipped[col].clip(lower=-threshold, upper=threshold)
            
            # Count clipped values
            clipped_lower = (df[col] < -threshold).sum()
            clipped_upper = (df[col] > threshold).sum()
            
            if clipped_lower > 0 or clipped_upper > 0:
                print(f"  {col}: clipped {clipped_lower} below -{threshold:.1e}, {clipped_upper} above {threshold:.1e}")
    
    return df_clipped

def test_clipping():
    """Test the clipping functions"""
    # Create test data with extreme values
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    # Normal data
    normal_data = np.random.normal(0, 1, (n_samples, n_features))
    
    # Add some extreme values
    extreme_data = normal_data.copy()
    extreme_data[0, 0] = 1e20  # Extremely large positive
    extreme_data[1, 1] = -1e15  # Extremely large negative
    extreme_data[2, 2] = 1e8    # Large positive
    extreme_data[3, 3] = -1e7   # Large negative
    
    df = pd.DataFrame(extreme_data, columns=[f'col{i}' for i in range(n_features)])
    
    print("Original DataFrame stats:")
    print(f"  Min: {df.values.min():.3e}")
    print(f"  Max: {df.values.max():.3e}")
    print(f"  Mean: {df.values.mean():.3e}")
    
    # Test percentile clipping
    df_percentile = clip_extreme_values(df, lower_percentile=1, upper_percentile=99)
    print("\nAfter percentile clipping (1-99%):")
    print(f"  Min: {df_percentile.values.min():.3e}")
    print(f"  Max: {df_percentile.values.max():.3e}")
    print(f"  Mean: {df_percentile.values.mean():.3e}")
    
    # Test threshold clipping
    df_threshold = clip_extreme_values_robust(df, threshold=1e6)
    print("\nAfter threshold clipping (|value| <= 1e6):")
    print(f"  Min: {df_threshold.values.min():.3e}")
    print(f"  Max: {df_threshold.values.max():.3e}")
    print(f"  Mean: {df_threshold.values.mean():.3e}")
    
    return df, df_percentile, df_threshold

if __name__ == "__main__":
    test_clipping()
