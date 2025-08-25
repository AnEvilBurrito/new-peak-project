import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
import hashlib
import numpy as np

def visualise_simulation(df, columns=None, activated_only=False, outcome_activation=False, show_legend=True):
    """
    Visualize time series data from a pandas DataFrame with deterministic coloring.
    
    Parameters:
    df (pd.DataFrame): Input dataframe containing 'time' column
    columns (list): Selected columns to plot (None plots all non-time columns)
    activated_only (bool): Show only columns ending with 'a' (activated species)
    outcome_activation (bool): Special mode to plot only 'O' and 'Oa'
    show_legend (bool): Toggle legend display
    """
    # Set visual properties
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'Arial'
    rcParams['font.size'] = 12
    rcParams['axes.titlesize'] = 16
    rcParams['axes.labelsize'] = 14
    
    # Handle special cases
    if outcome_activation:
        columns = ['O', 'Oa']
    elif activated_only:
        columns = [col for col in df.columns if col.endswith('a') and col != 'time']
    elif columns is None:
        columns = [col for col in df.columns if col != 'time']
    
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
    
    # Prepare data for plotting
    melt_df = df.melt(id_vars='time', value_vars=columns, 
                      var_name='Species', value_name='Value')
    
    # Create plot
    plt.figure(figsize=(12, 6))
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
    
    ax.set_title(f'Time Series of Selected Species{title_suffix}')
    ax.set_xlabel('Time', weight='bold')
    ax.set_ylabel('Concentration/Value', weight='bold')
    
    # Configure legend
    if show_legend:
        plt.legend(title='Species', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    # return final figure
    return plt.gcf()