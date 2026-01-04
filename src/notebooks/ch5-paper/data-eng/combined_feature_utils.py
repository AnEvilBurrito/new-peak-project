"""
Shared utilities for creating combined feature sets with intelligent suffixing.

This module provides functions to combine individual feature types (original features,
dynamic features, last time points) into combined feature sets with appropriate
suffixing to avoid column name conflicts.

Key function: create_combined_feature_sets()
"""

import pandas as pd
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def concat_with_smart_suffix(
    df1: pd.DataFrame, 
    df2: pd.DataFrame, 
    df1_type: str = "original", 
    df2_type: str = "dynamic_no_outcome"
) -> pd.DataFrame:
    """
    Concatenate DataFrames with intelligent suffixing based on feature type.
    
    Dynamic features already have descriptive suffixes (_auc, _median, etc.) 
    so they don't get additional suffixes. Original features get '_original' 
    suffix, and last time points get '_last' suffix.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame  
        df1_type: Type of features in df1 ('original', 'dynamic_with_outcome', 
                 'dynamic_no_outcome', 'last_with_outcome', 'last_no_outcome')
        df2_type: Type of features in df2 ('original', 'dynamic_with_outcome',
                 'dynamic_no_outcome', 'last_with_outcome', 'last_no_outcome')
    
    Returns:
        Combined DataFrame with appropriate suffixes
    """
    # Map feature types to suffix strategies
    suffix_strategies = {
        'original': '_original',  # Original features need suffix
        'dynamic_with_outcome': '',  # Dynamic features already have descriptive suffixes
        'dynamic_no_outcome': '',    # Dynamic features already have descriptive suffixes
        'last_with_outcome': '_last',  # Last time points need suffix
        'last_no_outcome': '_last'      # Last time points need suffix
    }
    
    suffix1 = suffix_strategies.get(df1_type, '')
    suffix2 = suffix_strategies.get(df2_type, '')
    
    # Only add suffix if not empty
    if suffix1:
        df1_suffixed = df1.add_suffix(suffix1)
    else:
        df1_suffixed = df1
        
    if suffix2:
        df2_suffixed = df2.add_suffix(suffix2)
    else:
        df2_suffixed = df2
    
    # Concatenate
    combined = pd.concat([df1_suffixed, df2_suffixed], axis=1)
    return combined


def validate_feature_dataframes(
    original_features: pd.DataFrame,
    dynamic_features_with_outcome: pd.DataFrame,
    last_time_points_with_outcome: pd.DataFrame,
    dynamic_features_no_outcome: pd.DataFrame,
    last_time_points_no_outcome: pd.DataFrame
) -> Tuple[bool, Optional[str]]:
    """
    Validate that all input DataFrames have consistent shapes and indices.
    
    Args:
        original_features: DataFrame of original features
        dynamic_features_with_outcome: DataFrame of dynamic features with outcome
        last_time_points_with_outcome: DataFrame of last time points with outcome
        dynamic_features_no_outcome: DataFrame of dynamic features without outcome
        last_time_points_no_outcome: DataFrame of last time points without outcome
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    feature_sets = {
        'original': original_features,
        'dynamic_with_outcome': dynamic_features_with_outcome,
        'last_with_outcome': last_time_points_with_outcome,
        'dynamic_no_outcome': dynamic_features_no_outcome,
        'last_no_outcome': last_time_points_no_outcome
    }
    
    # Check for empty DataFrames
    for name, df in feature_sets.items():
        if df is None or len(df) == 0:
            return False, f"Empty DataFrame for {name}"
    
    # Get reference shape and index
    ref_shape = original_features.shape
    ref_index = original_features.index
    
    # Check consistency
    for name, df in feature_sets.items():
        if df.shape[0] != ref_shape[0]:
            return False, f"Row count mismatch: {name} has {df.shape[0]} rows, original has {ref_shape[0]}"
        if not df.index.equals(ref_index):
            return False, f"Index mismatch for {name}"
    
    return True, None


def create_combined_feature_sets(
    original_features: pd.DataFrame,
    dynamic_features_with_outcome: pd.DataFrame,
    last_time_points_with_outcome: pd.DataFrame,
    dynamic_features_no_outcome: pd.DataFrame,
    last_time_points_no_outcome: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """
    Create combined feature sets from individual feature sets with intelligent suffixing.
    
    Creates four combined feature sets:
    1. Original + dynamic features without outcome
    2. Original + last time points without outcome
    3. Original + dynamic features with outcome
    4. Original + last time points with outcome
    
    Args:
        original_features: DataFrame of original features
        dynamic_features_with_outcome: DataFrame of dynamic features with outcome
        last_time_points_with_outcome: DataFrame of last time points with outcome
        dynamic_features_no_outcome: DataFrame of dynamic features without outcome
        last_time_points_no_outcome: DataFrame of last time points without outcome
    
    Returns:
        Dictionary mapping combined feature set names to DataFrames
    
    Raises:
        ValueError: If input DataFrames are inconsistent
    """
    logger.info("Creating combined feature sets...")
    
    # Validate inputs
    is_valid, error_msg = validate_feature_dataframes(
        original_features, dynamic_features_with_outcome,
        last_time_points_with_outcome, dynamic_features_no_outcome,
        last_time_points_no_outcome
    )
    
    if not is_valid:
        if error_msg:
            logger.warning(f"Validation failed: {error_msg}. Skipping combinations.")
            return {}
        else:
            raise ValueError("Feature DataFrame validation failed")
    
    combined_sets = {}
    
    # 1. Original + dynamic features without outcome
    combined_sets['original_plus_dynamic_no_outcome'] = concat_with_smart_suffix(
        original_features, dynamic_features_no_outcome,
        df1_type="original", df2_type="dynamic_no_outcome"
    )
    
    # 2. Original + last time points without outcome
    combined_sets['original_plus_last_no_outcome'] = concat_with_smart_suffix(
        original_features, last_time_points_no_outcome,
        df1_type="original", df2_type="last_no_outcome"
    )
    
    # 3. Original + dynamic features with outcome
    combined_sets['original_plus_dynamic_with_outcome'] = concat_with_smart_suffix(
        original_features, dynamic_features_with_outcome,
        df1_type="original", df2_type="dynamic_with_outcome"
    )
    
    # 4. Original + last time points with outcome
    combined_sets['original_plus_last_with_outcome'] = concat_with_smart_suffix(
        original_features, last_time_points_with_outcome,
        df1_type="original", df2_type="last_with_outcome"
    )
    
    logger.info(f"Created {len(combined_sets)} combined feature sets")
    for name, df in combined_sets.items():
        logger.info(f"  {name}: shape {df.shape}")
    
    return combined_sets


def get_combined_feature_names() -> list:
    """
    Get the standard names of combined feature sets.
    
    Returns:
        List of combined feature set names
    """
    return [
        'original_plus_dynamic_no_outcome',
        'original_plus_last_no_outcome',
        'original_plus_dynamic_with_outcome',
        'original_plus_last_with_outcome'
    ]


def save_combined_features_to_dict(
    combined_features: Dict[str, pd.DataFrame],
    feature_dict: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """
    Add combined features to a feature dictionary.
    
    Args:
        combined_features: Dictionary of combined feature DataFrames
        feature_dict: Existing dictionary of feature DataFrames
    
    Returns:
        Updated dictionary with combined features added
    """
    if not combined_features:
        return feature_dict
    
    updated_dict = feature_dict.copy()
    for name, df in combined_features.items():
        updated_dict[name] = df
    
    return updated_dict