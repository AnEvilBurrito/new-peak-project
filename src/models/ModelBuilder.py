# The following function generates antimony strings from high level syntax
from typing import List
from .Reaction import Reaction


class ModelBuilder:

    def __init__(self, name):
        self.name = name
        self.reactions: List[Reaction] = []
        # co-pilot generated, not sure if this is necessary
        self.ant_model = ''
        self.sbml_model = ''
        self.r_model = ''
        self.r_solved = ''
        self.r_plot = ''

    def get_parameters(self):
        '''
        Extracts parameters and their values from all reactions 
        in the class and returns a dict while subjecting to a naming rule 
        
        '''

        parameters_names = []
        parameters = {}
        i = 0
        while i < len(self.reactions):
            r = self.reactions[i]
            # first, get the parameters names from the archtype
            # and perform naming rule

            '''
            Here, a simple naming rule is implemented. It simply appends the reaction index 
            to the parameter name
            TODO: implement more complex naming rules in the future 
            '''
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

        states_list = {}
        for r in self.reactions:
            states_list.update(r.get_reaction_states())

        return states_list

    def add_reaction(self, reaction: Reaction):
        self.reactions.append(reaction)

    def compile_antimony(self):

        antimony_string = ''

        antimony_string += f'model {self.name}\n\n'

        # add reactions
        i = 0
        while i < len(self.reactions):
            r = self.reactions[i]
            r_index = f'J{i}'
            antimony_string += r.get_antimony_reaction_str(r_index)
            antimony_string += '\n'
            i += 1

        # add state vars
        antimony_string += '\n'
        antimony_string += '# State variables in the system\n'
        all_states = self.get_state_variables()
        for key, val in all_states.items():
            antimony_string += f'{key}={val}\n'
        antimony_string += '\n'

        # add parameters
        antimony_string += '# Parameters in the system\n'
        all_params = self.get_parameters()
        for key, val in all_params.items():
            antimony_string += f'{key}={val}\n'

        antimony_string += '\nend'

        return antimony_string
