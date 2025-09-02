import numpy as np
import pandas as pd

def calculate_spread_metrics(target_df, target_column='Oa'):
    """
    Calculate various spread/variability metrics for target data.
    
    Args:
        target_df (pd.DataFrame): DataFrame containing target values.
        target_column (str): Column name to analyze (default 'Oa').
    
    Returns:
        dict: Dictionary containing spread metrics:
            - range: max - min
            - variance: sample variance
            - std_dev: standard deviation
            - iqr: interquartile range
            - mad: mean absolute deviation
    """
    if target_column not in target_df.columns:
        raise ValueError(f"Column '{target_column}' not found in DataFrame.")
    
    data = target_df[target_column].dropna()  # Drop missing values
    
    metrics = {
        'range': data.max() - data.min(),
        'variance': data.var(),
        'std_dev': data.std(),
        'iqr': np.percentile(data, 75) - np.percentile(data, 25),
        'mad': data.mad()
    }
    
    return metrics
