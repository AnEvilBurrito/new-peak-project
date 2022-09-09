import matplotlib.pyplot as plt
import pandas as pd

def plot_state_with_exp_data(exp_state: str, sim_result, mock_time, mock_data, mock_data_std, stim_time, color='blue'):
    plt.plot(sim_result['time'][stim_time:],
             sim_result[f'{exp_state}'][stim_time:], label=exp_state, color=color)
    plt.scatter(mock_time, mock_data, label=f'exp {exp_state}', color=color)
    plt.errorbar(mock_time, mock_data, yerr=mock_data_std,
                 fmt='none', color=color)
    plt.grid()
    plt.legend()
    plt.show()


def normalize(df: pd.DataFrame, pdf_min, pdf_max):
    result = df.copy()
    for feature_name in df.columns:
        if feature_name != 'error':
            max_value = pdf_max[feature_name]
            min_value = pdf_min[feature_name]
            result[feature_name] = (
                df[feature_name] - min_value) / (max_value - min_value)
    return result


def generate_min_and_max_bound(bounds, params_name):
    min_bound = {}
    max_bound = {}
    for i, p in enumerate(params_name):
        min_bound[p] = bounds[i][0]
        max_bound[p] = bounds[i][1]

    return min_bound, max_bound


def create_bounds(params: list, div_size: int = 10, min_val: float = -1, max_val: float = -1):

    bounds = []
    if min_val > 0 and max_val > 0:
        for p in params:
            bounds.append((min_val, max_val))
    else: 
        for param in params:
            bounds.append((param/div_size, param*div_size))
    
    return bounds
