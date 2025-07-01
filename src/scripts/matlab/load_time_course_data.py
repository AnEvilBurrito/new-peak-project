# load_timecourse_data.py

import pandas as pd
import numpy as np

def load_timecourse_data(df: pd.DataFrame, index_option: str = "combined_index") -> pd.DataFrame:
    """
    Transform a time-course DataFrame with one row per timepoint into a wide format where each variable is a time array,
    grouped by (RunID, IC_ID, ParamSet_ID).

    Parameters:
    - df (pd.DataFrame): Input DataFrame in long format.
    - index_option (str): One of ['combined_index', 'cell_id', 'param_index', 'ranged_index'].

    Returns:
    - pd.DataFrame: Transformed DataFrame with one row per simulation group and time-series arrays per species.
    """
    valid_options = {"combined_index", "cell_id", "param_index", "ranged_index"}
    if index_option not in valid_options:
        raise ValueError(f"Invalid index_option: '{index_option}'. Must be one of {valid_options}.")

    group_cols = ['RunID', 'IC_ID', 'ParamSet_ID']
    time_col = 'Time'
    value_cols = [col for col in df.columns if col not in group_cols + [time_col]]

    df_sorted = df.sort_values(by=group_cols + [time_col])
    grouped = df_sorted.groupby(group_cols)

    records = []
    index = []

    for keys, group in grouped:
        record = {col: group[col].to_numpy() for col in value_cols}
        records.append(record)

        run_id, ic_id, param_id = keys
        if index_option == "combined_index":
            index.append(f"{run_id}-{ic_id}-{param_id}")
        elif index_option == "cell_id":
            index.append(ic_id)
        elif index_option == "param_index":
            index.append(param_id)
        elif index_option == "ranged_index":
            index.append(None)  # will be replaced by default RangeIndex

    result_df = pd.DataFrame(records)

    # Apply index
    if index_option != "ranged_index":
        result_df.index = index

    return result_df
