from typing import Dict, Iterable, Tuple, List

class ExperimentalData:

    '''
    A wrapper for a dict that stores experimental data.
    '''

    def __init__(self):

        self.data = {}

    def add_data(self, time: Iterable, value: Iterable, std_error: Iterable, name: str):

        '''
        Add experimental data to the dict.
        '''

        assert len(time) == len(value) == len(std_error), 'time, value, and std_error must have the same length'

        if name in self.data:
            raise ValueError(f'Experimental data with name {name} already exists')

        self.data[name] = {'time': time, 'value': value, 'std_error': std_error}

    def get_names(self) -> List[str]:

        '''
        Get the names of the experimental data.
        '''

        return list(self.data.keys())

    def get_data(self, name: str) -> Dict:

        '''
        Get experimental data from the dict.
        '''

        if name not in self.data:
            raise ValueError(f'Experimental data with name {name} does not exist')

        return self.data[name]

    def get_data_tuple(self, name: str) -> Tuple[Iterable, Iterable, Iterable]:

        '''
        Get experimental data from the dict as a tuple.
        '''

        if name not in self.data:
            raise ValueError(f'Experimental data with name {name} does not exist')

        return self.data[name]['time'], self.data[name]['value'], self.data[name]['std_error']