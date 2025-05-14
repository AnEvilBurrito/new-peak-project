import pandas as pd
import numpy as np
from models.Solver.RoadrunnerSolver import RoadrunnerSolver

# measure responsiveness of the model to the drug 
def measure_model_responsiveness(result: pd.DataFrame, drug_time_point, verbose=0, absolute=False):
    nearest_time_index = (result['time'] - drug_time_point).abs().idxmin()
    if verbose:
        print(f'Nearest time index to {drug_time_point}: {nearest_time_index}')
    
    # final time index = len(result) - 1
    final_time_index = len(result) - 1
    # get the value of Cp at the nearest time index and the final time index
    Cp_initial_time = result['Cp'][0]
    Cp_before_drug_time = result['Cp'][nearest_time_index - 1]
    Cp_nearest_time = result['Cp'][nearest_time_index]
    Cp_final_time = result['Cp'][final_time_index]
    
    if Cp_initial_time == 0:
        Cp_initial_time = 1
        if verbose:
            print(f'Cp_initial_time was 0, set to 1')
    rate_of_change_untreated = (Cp_before_drug_time - Cp_initial_time) / Cp_initial_time
    total_rate_of_change_untreated = (Cp_before_drug_time - Cp_initial_time) / 100
    if verbose:
        print(f'Initial Cp: {Cp_initial_time:.2f}, Nearest Cp: {Cp_nearest_time:.2f}, Final Cp: {Cp_final_time:.2f}')
        print(f'Rate of change of Cp (IRoC): {rate_of_change_untreated:.4f}')
        print(f'Total rate of change of Cp (ITRoC): {total_rate_of_change_untreated:.4f}')
        
    rate_of_change_drug_induced = (Cp_final_time - Cp_nearest_time) / Cp_nearest_time
    total_rate_of_change_drug_induced = (Cp_final_time - Cp_nearest_time) / 100
    if verbose:
        print(f'Drug-induced Rate of change of Cp (DRoC): {rate_of_change_drug_induced:.4f}')
        print(f'Drug-induced Total rate of change of Cp (DTRoC): {total_rate_of_change_drug_induced:.4f}')
    if absolute:
        rate_of_change_untreated = abs(rate_of_change_untreated)
        total_rate_of_change_untreated = abs(total_rate_of_change_untreated)
        rate_of_change_drug_induced = abs(rate_of_change_drug_induced)
        total_rate_of_change_drug_induced = abs(total_rate_of_change_drug_induced)
    return rate_of_change_untreated, total_rate_of_change_untreated, rate_of_change_drug_induced, total_rate_of_change_drug_induced


def sample_randomized_values(base_dict, value_range, multiplier_range=None):
    sampled = {}
    is_global_range = isinstance(value_range, tuple) and len(value_range) == 2

    for key, base_val in base_dict.items():
        if is_global_range:
            low, high = value_range
        else:
            low, high = value_range.get(key, (0.5 * base_val, 1.5 * base_val))

        if multiplier_range:
            m_low, m_high = multiplier_range
            multiplier = np.random.uniform(m_low, m_high)
            val = base_val * multiplier
        else:
            val = np.random.uniform(low, high)
        
        sampled[key] = val
    return sampled

def test_average_model_responsiveness(G, param_range, specie_range, multiplier_range,
                                      drug_time_point=500, n_runs=10, sim_time=1000,
                                      sim_step=100, seed=None, verbose=False):
    if seed is not None:
        np.random.seed(seed)

    solver = RoadrunnerSolver()
    solver.compile(G.get_sbml_model())

    metrics_list = []
    labels = ["IRoC", "ITRoC", "DRoC", "DTRoC"]

    for i in range(n_runs):
        rand_params = sample_randomized_values(G.get_parameters(), param_range, multiplier_range)
        rand_states = sample_randomized_values(G.get_state_variables(), specie_range)

        solver.set_parameter_values(rand_params)
        solver.set_state_values(rand_states)

        result = solver.simulate(0, sim_time, sim_step)
        metrics = measure_model_responsiveness(result, drug_time_point, absolute=True)
        metrics_list.append(metrics)

        if verbose:
            print(f"Run {i+1}: {[f'{label}={m:.4f}' for label, m in zip(labels, metrics)]}")

    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics_list, columns=labels)

    # Summary statistics
    summary_df = metrics_df.agg(['mean', 'std']).T.reset_index()
    summary_df.columns = ['Metric', 'Mean', 'Std']

    return metrics_df, summary_df


