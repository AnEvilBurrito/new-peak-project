import pandas as pd

class ExpDataLoader:

    '''
    Assists in the loading and processing of data formats for ExperimentalData class
    from raw data files

    Data should be loaded in a .csv format from pandas. The data should be in the following format:
    experimental_state, replicate_number, time_point, value, 

    exp_state and replicate_number are used to infer list construction of the data. 

    extend the mock time by {stim_time} in the model
    '''

    def __init__(self):
        self.exp_data = None

    def load_data(self, filename, delimiter=','):
        '''
        Loads the data from a file and stores it in the class

        filename: the name of the file to load
        '''
        self.exp_data = pd.read_csv(filename, delimiter=delimiter)

    def get_data(self):
        '''
        Returns the data that has been loaded into the class
        '''
        return self.exp_data

    def get_experimental_states(self):
        '''
        Returns the list of experimental states that are in the data
        '''
        return self.exp_data['experimental_state'].unique()

    def get_state_average_for_each_time_point(self, state):
        '''
        Returns the average of the experimental data for a given state
        '''
        return self.exp_data[self.exp_data['experimental_state'] == state].groupby('time_point').mean()['value']

    def get_state_std_for_each_time_point(self, state):
        '''
        Returns the standard deviation of the experimental data for a given state
        '''
        return self.exp_data[self.exp_data['experimental_state'] == state].groupby('time_point').std()['value']

    def get_state_modified_time(self, state, stim_time):
        '''
        Returns the time points for a given state, modified by the stim_time
        '''
        return self.exp_data[self.exp_data['experimental_state'] == state]['time_point'].unique() + stim_time

    def get_state_raw_data(self, state):
        '''
        Returns the raw data for a given state
        '''
        return self.exp_data[self.exp_data['experimental_state'] == state]['value']
