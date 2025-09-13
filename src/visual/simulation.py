import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import copy 

def visualise_simulation(df, columns=None, activated_only=False, outcome_activation=False, 
                         show_legend=True, 
                         time_period=None,
                         normalise=False,
                         fig_size=(12, 6)):
    """
    Visualize time series data from a pandas DataFrame with deterministic coloring.
    
    Parameters:
    df (pd.DataFrame): Input dataframe containing 'time' column
    columns (list): Selected columns to plot (None plots all non-time columns)
    activated_only (bool): Show only columns ending with 'a' (activated species)
    outcome_activation (bool): Special mode to plot only 'O' and 'Oa'
    show_legend (bool): Toggle legend display
    time_period (tuple): If set, filter data to show only within time range (start, end).
                        Can be (start, None) for start time only, (None, end) for end time only,
                        or (start, end) for both start and end times.
    normalise (bool): If True, normalise each column to range [0, 1] using min-max normalization
                     to better compare response ranges across species
    fig_size (tuple): Figure size as (width, height)
    """
    # Set visual properties
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'Arial'
    rcParams['font.size'] = 12
    rcParams['axes.titlesize'] = 16
    rcParams['axes.labelsize'] = 14
    
    # make a copy of the dataframe to avoid modifying the original
    copy_df = copy.deepcopy(df)
    
    # Handle special cases
    if outcome_activation:
        columns = ['O', 'Oa']
    elif activated_only:
        columns = [col for col in copy_df.columns if col.endswith('a') and col != 'time']
    elif columns is None:
        columns = [col for col in copy_df.columns if col != 'time']
    
    # Prepare custom colors
    np.random.seed(42)  # Fixed random seed for reproducibility
    
    # Create custom color handling
    palette = sns.color_palette('colorblind') + sns.color_palette('bright')
    
    # Remove blues from base palette
    palette = [color for color in palette 
               if not (color[0] < 0.3 and color[1] < 0.6 and color[2] > 0.7)]
    
    # Generate deterministic color mapping
    unique_species = sorted(set(columns))  # Sort for deterministic ordering
    color_map = {}
    
    # Assign fixed colors for O and Oa
    color_map['O'] = (0.6, 0.8, 1.0)   # Light blue
    color_map['Oa'] = (0.0, 0.3, 0.7)  # Dark blue
    
    # Assign colors to other species
    for i, specie in enumerate(unique_species):
        if specie not in ['O', 'Oa']:
            # Use modulo to cycle through available colors
            color_map[specie] = palette[i % len(palette)]
    
    # Apply time period filtering if specified
    if time_period is not None:
        if not isinstance(time_period, tuple) or len(time_period) != 2:
            raise ValueError("time_period must be a tuple of (start, end) where either can be None")
        
        start_time, end_time = time_period
        
        # Apply start time filter
        if start_time is not None:
            copy_df = copy_df[copy_df['time'] >= start_time].copy()
        
        # Apply end time filter
        if end_time is not None:
            copy_df = copy_df[copy_df['time'] <= end_time].copy()
        
        # Check if filtering resulted in empty DataFrame
        if copy_df.empty:
            period_str = f"between {start_time} and {end_time}" if start_time is not None and end_time is not None else \
                        f"after {start_time}" if start_time is not None else \
                        f"before {end_time}"
            raise ValueError(f"No data found {period_str}")
    
    # Apply normalization if specified
    if normalise:
        df_normalized = copy_df.copy()
        for col in columns:
            if col in df_normalized.columns:
                col_min = df_normalized[col].min()
                col_max = df_normalized[col].max()
                
                # Handle constant values (min == max)
                if col_max == col_min:
                    df_normalized[col] = 0.5  # Set to middle value for constant columns
                else:
                    df_normalized[col] = (df_normalized[col] - col_min) / (col_max - col_min)
        copy_df = df_normalized
    
    # Prepare data for plotting
    melt_df = copy_df.melt(id_vars='time', value_vars=columns, 
                      var_name='Species', value_name='Value')
    
    # Create plot
    plt.figure(figsize=fig_size)
    ax = sns.lineplot(
        data=melt_df, 
        x='time', 
        y='Value', 
        hue='Species',
        palette=color_map,
        linewidth=2.5
    )
    
    # Customize plot
    title_suffix = ''
    if outcome_activation:
        title_suffix = ' (Outcome Activation)'
    elif activated_only:
        title_suffix = ' (Activated Only)'
    
    if normalise:
        title_suffix += ' (Normalized)'
    
    ax.set_title(f'Time Series of Selected Species{title_suffix}')
    ax.set_xlabel('Time', weight='bold')
    
    # Update y-axis label based on normalization
    y_label = 'Normalized Value (0-1)' if normalise else 'Concentration/Value'
    ax.set_ylabel(y_label, weight='bold')
    
    # Configure legend
    if show_legend:
        plt.legend(title='Species', bbox_to_anchor=(1.05, 1), loc='upper left')
    else: 
        plt.legend().remove()
    
    plt.tight_layout()
    # return final figure
    return plt.gcf()
