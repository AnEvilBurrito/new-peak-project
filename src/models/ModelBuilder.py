# The following function generates antimony strings from high level syntax

from typing import List, Union, Tuple
import antimony
import roadrunner

from .Reaction import Reaction
from .ReactionArchtype import ReactionArchtype

class ModelBuilder:

    '''
    Docstring
    '''

    def __init__(self, name):
        self.name = name
        self.reactions: List[Reaction] = []
        # co-pilot generated, not sure if this is necessary

        self.states = {}
        self.parameters = {}
        self.variables = {}

        self.custom_strings = {}

        self.roadrunner_model: roadrunner.RoadRunner = None # type: roadrunner.RoadRunner

    def get_parameters(self):
        '''
        Extracts parameters and their values from all reactions 
        in the class and returns a dict while subjecting to a naming rule 
        
        '''
        parameters = {}
        i = 0
        while i < len(self.reactions):
            r = self.reactions[i]
            # first, get the parameters names from the archtype
            # and perform naming rule

            
            #Here, a simple naming rule is implemented. It simply appends the reaction index 
            #to the parameter name
            #TODO: implement more complex naming rules in the future 
            
            r_index = f'J{i}'

            parameters.update(r.get_reaction_parameters(r_index))

            i += 1

        return parameters

    def get_state_variables(self):
        '''
        Extracts state variables and their values from all reactions 
        in the class and returns a dict

        non-unique state variables will only be repeated once, their 
        default value will only follow the first repeated state variable
        '''

        states = {}
        for r in self.reactions:
            states.update(r.get_reaction_states())

        return states

    def get_other_variables(self):
        '''
        Doc
        '''
        return self.variables

    def get_all_variables_keys(self):
        '''
        Doc
        '''
        return list(self.get_state_variables().keys()) + list(self.variables.keys())

    def add_reaction(self, reaction: Reaction):
        '''
        Doc
        '''
        self.reactions.append(reaction)
        # NOTE: This is an ugly hack to make sure the reaction is added to the model
        # TODO: find a better way to do this
        self.states.update(self.get_state_variables())
        self.parameters.update(self.get_parameters())

    def inject_antimony_string_at(self, ant_string: str, position: str = 'reaction'):

        '''
        position can only be in str: top, reaction, state, parameters, end 
        '''
        all_positions = ['top', 'reaction', 'state', 'parameters', 'end']
        for p in all_positions:
            if p == position:
                if p in self.custom_strings:
                    self.custom_strings[p] += ant_string + '\n'
                else: 
                    self.custom_strings[p] = ant_string + '\n'
                break


    def add_simple_piecewise(self, before_value: float, activation_time: float, after_value: float, state_name: str):
        '''
        Adds a simple piecewise function to the state variable state_name
        '''
        self.variables[state_name] = f'{state_name} := piecewise({before_value}, time < {activation_time}, {after_value})'
        # self.inject_antimony_string_at(f"{state_name} := piecewise({after_value}, time > {activation_time}, {before_value})", 'parameters')

    def get_antimony_model(self):
        '''
        Doc
        '''
        antimony_string = ''

        antimony_string += f'model {self.name}\n\n'

        # add top custom str 
        if 'top' in self.custom_strings:
            antimony_string += self.custom_strings['top']

        # add reactions
        
        # first, add reaction custom str 
        if 'reaction' in self.custom_strings:
            antimony_string += self.custom_strings['reaction']

        i = 0
        while i < len(self.reactions):
            r = self.reactions[i]
            r_index = f'J{i}'
            antimony_string += r.get_antimony_reaction_str(r_index)
            antimony_string += '\n'
            if r.reversible: 
                antimony_string += r.get_antimony_reactions_reverse_str(r_index)
                antimony_string += '\n'
            i += 1

        # add state vars

        # first, add state custom str
        antimony_string += '\n'
        antimony_string += '# State variables in the system\n'
        if 'state' in self.custom_strings:
            antimony_string += self.custom_strings['state']

        all_states = self.get_state_variables()
        for key, val in all_states.items():
            antimony_string += f'{key}={val}\n'
        antimony_string += '\n'

        # add parameters

        # first, add parameter custom str
        antimony_string += '# Parameters in the system\n'
        if 'parameters' in self.custom_strings:
            antimony_string += self.custom_strings['parameters']

        all_params = self.get_parameters()
        for key, val in all_params.items():
            antimony_string += f'{key}={val}\n'

        # add other variables
        antimony_string += '\n'
        antimony_string += '# Other variables in the system\n'
        for key, val in self.variables.items():
            antimony_string += f'{val}\n'
        antimony_string += '\n' 

        # add end custom str
        if 'end' in self.custom_strings:
            antimony_string += self.custom_strings['end']

        antimony_string += '\nend'

        return antimony_string
    
    def get_sbml_model(self) -> str:

        '''
        Doc
        '''
        
        ant_model = self.get_antimony_model()
        antimony.clearPreviousLoads()
        antimony.freeAll()
        code = antimony.loadAntimonyString(ant_model)
        if code >= 0:
            mid = antimony.getMainModuleName()
            sbml_model = antimony.getSBMLString(mid)
            return sbml_model

        raise Exception('Error in loading antimony model')


    def compile_to_roadrunner(self, sbml_model_str: str):

        roadrunner_model = roadrunner.RoadRunner(sbml_model_str)
        self.roadrunner_model = roadrunner_model
        print('Roadrunner model compiled, run self.roadrunner_model.simulate() to simulate')


    def get_roadrunner_model(self, sbml_str: str):

        roadrunner_model = roadrunner.RoadRunner(sbml_str)
        print('Roadrunner model compiled and returned')
        return roadrunner_model

    
    # These two functions are testing helpers only, not to use in practice
    

    def simulate(self, start: float, end: float, step: float):
        '''
        Simulates the model using roadrunner and returns the results
        '''

        roadrunner_model = roadrunner.RoadRunner(self.get_sbml_model())
        self.r_model = roadrunner_model
        r_solved = roadrunner_model.simulate(start, end, step)
        self.r_solved = r_solved
        return r_solved

    def plot(self):
        '''
        Plots the results of the simulation
        '''
        self.r_model.plot()

        

        
