# load_target_data.py

import pandas as pd
import numpy as np

def load_target_data(df: pd.DataFrame, species: str, index_option: str = "combined_index") -> pd.DataFrame:
    """
    Extracts the last time point for a specified species from simulation timecourse data.

    Parameters:
    - df (pd.DataFrame): Long-format timecourse DataFrame.
    - species (str): The species column to extract (e.g., 'Cp', 'pERK').
    - index_option (str): One of ['combined_index', 'cell_id', 'param_index', 'ranged_index'].

    Returns:
    - pd.DataFrame: DataFrame with a single column (species), one row per simulation group.
    """
    valid_options = {"combined_index", "cell_id", "param_index", "ranged_index"}
    if index_option not in valid_options:
        raise ValueError(f"Invalid index_option: '{index_option}'. Must be one of {valid_options}.")

    group_cols = ['RunID', 'IC_ID', 'ParamSet_ID']
    time_col = 'Time'

    if species not in df.columns:
        raise ValueError(f"Species '{species}' not found in the dataframe.")

    # Ensure last timepoint is chosen correctly
    df_sorted = df.sort_values(by=group_cols + [time_col])
    grouped = df_sorted.groupby(group_cols)
    last_values = grouped[species].last().reset_index()

    # Set index according to index_option
    if index_option == "combined_index":
        last_values['ID'] = last_values.apply(lambda row: f"{row['RunID']}-{row['IC_ID']}-{row['ParamSet_ID']}", axis=1)
        last_values.set_index('ID', inplace=True)
        last_values = last_values[[species]]
    elif index_option == "cell_id":
        last_values.set_index('IC_ID', inplace=True)
        last_values = last_values[[species]]
    elif index_option == "param_index":
        last_values.set_index('ParamSet_ID', inplace=True)
        last_values = last_values[[species]]
    elif index_option == "ranged_index":
        last_values = last_values[[species]]
        last_values.reset_index(drop=True, inplace=True)

    return last_values
