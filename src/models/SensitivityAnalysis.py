from models.ModelBuilder import ModelBuilder
from models.Solver.Solver import Solver 

## Helper functions
import warnings
import numpy as np

def sensitivity_analysis(builder: ModelBuilder, solver: Solver, specie_name: str, specie_range: tuple, simulation_time: float, simulation_step: int):
    all_results = []
    for specie in specie_range:
        solver.set_state_values({specie_name: np.float64(specie)})
        try: 
            res = solver.simulate(0, simulation_time, simulation_step)
            all_results.append(res)
        except Exception as e:
            warnings.warn(f"Simulation failed for specie {specie}: {e}")
            continue
    return all_results

def extract_states_from_results(results, state_name, index):
    all_states = []
    for result in results:
        state = result[state_name]
        # get the index of the state in the result, keep in mind that state is pd.Series
        specific_state = state.iloc[index]
        all_states.append(specific_state)
    return all_states

def get_sensitivity_score(states):
    state_sensitivity = []  
    for i, specific_states in enumerate(states):
        # get the max and min of the Cp final state list 
        max_state = max(specific_states)
        min_state = min(specific_states)
        # get the range of the Cp final state list 
        range_state = max_state - min_state
        # append to the list 
        state_sensitivity.append(range_state)
    # print the mean of the state sensitivity
    return sum(state_sensitivity) / len(state_sensitivity)

def get_sensitivity_score(states):
    state_sensitivity = []  
    for i, specific_states in enumerate(states):
        # get the max and min of the Cp final state list 
        max_state = max(specific_states)
        min_state = min(specific_states)
        # get the range of the Cp final state list 
        range_state = max_state - min_state
        # append to the list 
        state_sensitivity.append(range_state)
    # print the mean of the state sensitivity
    return state_sensitivity