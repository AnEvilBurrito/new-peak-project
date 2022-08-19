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

        self.r_model = None 
        self.r_solved = None 

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

        states_list = {}
        for r in self.reactions:
            states_list.update(r.get_reaction_states())

        return states_list

    def add_reaction(self, reaction: Reaction):
        '''
        Doc
        '''
        self.reactions.append(reaction)
        # NOTE: This is an ugly hack to make sure the reaction is added to the model
        # TODO: find a better way to do this
        self.states.update(self.get_state_variables())
        self.parameters.update(self.get_parameters())

    def add_reaction_test(self, reaction_archtype: ReactionArchtype,
                     reactants: Tuple[str],
                     products: Tuple[str],
                     extra_states: Tuple[str] = (),
                     parameters_values: Union[dict, tuple, int, float] = (),
                     reactant_values: Union[dict, tuple, int, float] = (),
                     product_values: Union[dict, tuple, int, float] = ()):
        
        '''
        Docstring, consider specialised parameter assignment syntax 
        symbol @{idx}_{param_name}: inject parameter name from reaction idx's parameter param_name 
        symbol %-{idx}_{param_name}: given parameter i, inject parameter name from reaction i-idx parameter param_name (relative)
        symbol ${param_name}: inject parameter name from global parameter param_name, global parameters have fixed names 
        '''
        reaction = Reaction(reaction_archtype, reactants, products, extra_states, parameters_values, reactant_values, product_values)
        pass 

    def inject_antimony_string_at(self, ant_string: str, position: str = 'reaction'):

        '''
        position can only be in str: top, reaction, state, parameters, end 
        '''

    def add_simple_piecewise(self, before_value: float, activation_time: float, after_value: float, state_name: str):
        '''
        Adds a simple piecewise function to the state variable state_name
        '''
        pass 

    def get_antimony_model(self):
        '''
        Doc
        '''
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

        

        
