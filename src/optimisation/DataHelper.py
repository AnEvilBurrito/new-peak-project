from typing import Dict, Iterable, Tuple, List
import numpy as np 

class DataHelper:

    '''
    A wrapper for a dict that stores experimental data.
    '''

    def __init__(self):

        self.exp_data = {}
        self.sim_data = {}
        self.total_size = 0 

    def add_exp_data(self, time: Iterable, value: Iterable, std_error: Iterable, name: str):

        '''
        Add experimental data to the dict.
        '''

        assert len(time) == len(value) == len(std_error), 'time, value, and std_error must have the same length'

        if name in self.exp_data:
            raise ValueError(f'Experimental data with name {name} already exists')

        self.total_size += len(time)

        self.exp_data[name] = {'time': time, 'value': value, 'std_error': std_error}

    def get_exp_state_names(self) -> List[str]:

        '''
        Get the names of the experimental data.
        '''

        return list(self.exp_data.keys())

    def get_exp_data(self, name: str) -> Dict:

        '''
        Get experimental data from the dict.
        '''

        if name not in self.exp_data:
            raise ValueError(f'Experimental data with name {name} does not exist')

        return self.exp_data[name]

    def get_exp_data_time(self, name: str) -> Iterable:
            
        '''
        Get the time data from the dict.
        '''

        if name not in self.exp_data:
            raise ValueError(f'Experimental data with name {name} does not exist')

        return self.exp_data[name]['time']

    def match_sim_data(self, result):

        # find closest index in sim time array compared to experimental time
        for name in self.exp_data:
            exp_time = self.exp_data[name]['time']
            model_data = []
            for t in exp_time:
                idx = np.argmin(np.abs(result['time'] - t))
                model_value = result[f'{name}'][idx]
                model_data.append(model_value)

            self.sim_data[name] = model_data

    

    def get_data_for_optimisation(self):

        '''
        Returns experimental standard deviations and values, as well 
        as model values in a single array.
        '''

        exp_std = []
        exp_value = []
        sim_value = []

        for name in self.exp_data:
            exp_std += list(self.exp_data[name]['std_error'])
            exp_value += list(self.exp_data[name]['value'])
            sim_value += self.sim_data[name]

        return np.array(exp_std), np.array(exp_value), np.array(sim_value)
        

    def get_data_tuple(self, name: str) -> Tuple[Iterable, Iterable, Iterable]:

        '''
        Get experimental data from the dict as a tuple.
        '''

        if name not in self.exp_data:
            raise ValueError(f'Experimental data with name {name} does not exist')

        return self.exp_data[name]['time'], self.exp_data[name]['value'], self.exp_data[name]['std_error']
