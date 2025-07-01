# load_target_data.py

import pandas as pd
import numpy as np

def load_target_data(df: pd.DataFrame, species: str, use_combined_index: bool = True) -> pd.DataFrame:
    """
    Extracts the last time point for a specified species from simulation timecourse data.

    Parameters:
    - df (pd.DataFrame): Long-format timecourse DataFrame.
    - species (str): The species column to extract (e.g., 'Cp', 'pERK').
    - use_combined_index (bool): Whether to use 'RunID-IC_ID-ParamSet_ID' as index.

    Returns:
    - pd.DataFrame: DataFrame with a single column (species), one row per simulation group.
    """
    group_cols = ['RunID', 'IC_ID', 'ParamSet_ID']
    time_col = 'Time'

    if species not in df.columns:
        raise ValueError(f"Species '{species}' not found in the dataframe.")

    # Sort to ensure last timepoint is correctly chosen
    df_sorted = df.sort_values(by=group_cols + [time_col])
    grouped = df_sorted.groupby(group_cols)

    # Extract the last timepoint's value of the species
    last_values = grouped[species].last().reset_index()

    if use_combined_index:
        last_values['ID'] = last_values.apply(lambda row: f"{row['RunID']}-{row['IC_ID']}-{row['ParamSet_ID']}", axis=1)
        last_values.set_index('ID', inplace=True)
        last_values = last_values[[species]]  # Drop grouping columns
    else:
        last_values.set_index(group_cols, inplace=True)

    return last_values