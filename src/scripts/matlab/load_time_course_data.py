import pandas as pd
import numpy as np

def load_timecourse_data(df: pd.DataFrame, use_combined_index: bool = True) -> pd.DataFrame:
    """
    Transform a time-course DataFrame with one row per timepoint
    into a wide format with arrays for each variable, grouped by (RunID, IC_ID, ParamSet_ID).

    Parameters:
    - df (pd.DataFrame): Input DataFrame in long format.
    - use_combined_index (bool): If True, set the index to 'RunID-IC_ID-ParamSet_ID'.

    Returns:
    - pd.DataFrame: Transformed DataFrame with one row per simulation and each column a numpy array over time.
    """
    # Identify grouping columns and value columns
    group_cols = ['RunID', 'IC_ID', 'ParamSet_ID']
    time_col = 'Time'
    value_cols = [col for col in df.columns if col not in group_cols + [time_col]]

    # Sort and group the data
    df_sorted = df.sort_values(by=group_cols + [time_col])
    grouped = df_sorted.groupby(group_cols)

    # Build new rows with each value column turned into an array
    records = []
    index = []

    for keys, group in grouped:
        record = {}
        for col in value_cols:
            record[col] = group[col].to_numpy()
        records.append(record)
        if use_combined_index:
            index.append(f"{keys[0]}-{keys[1]}-{keys[2]}")
        else:
            index.append(keys)

    # Create the output DataFrame
    result_df = pd.DataFrame(records, index=index)

    # Rename columns if needed (optional)
    # result_df.columns = [your_custom_names]

    return result_df