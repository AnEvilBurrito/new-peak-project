from typing import Union, Tuple, List 
from .ReactionArchtype import ReactionArchtype

class Reaction: 
    '''
    This class represents a reaction in the system.
        ENSURE: use (a,) instead of (a) for tuples typing
        reaction_archtype: ReactionArchtype, 
        reactants: Tuple[str], 
        products: Tuple[str], 
        extra_states: Tuple[str] = (),
        parameters_values: Union[dict, tuple] = (),
        reactant_values: Union[dict, tuple] = (),
        product_values: Union[dict, tuple] = ()
    '''
    def __init__(self, 
        reaction_archtype: ReactionArchtype, 
        reactants: Tuple[str], 
        products: Tuple[str], 
        extra_states: Tuple[str] = (),
        parameters_values: Union[dict, tuple] = (),
        reactant_values: Union[dict, tuple] = (),
        product_values: Union[dict, tuple] = ()):

        # TODO: perform some error checking on the input
        

        assert len(reactants) == reaction_archtype.reactants_count, f'length of reactants must be equal to the number of reactants in the reaction archtype, {len(reactants)} != {reaction_archtype.reactants_count}'
        assert len(products) == reaction_archtype.products_count, f'length of products must be equal to the number of products in the reaction archtype, {len(products)} != {reaction_archtype.products_count}'
        assert reaction_archtype.validate_parameters(parameters_values, reaction_archtype.parameters), 'parameters_values must be valid, see implementation of validate_parameters in ReactionArchtype'

        self.archtype = reaction_archtype
        # must specify reactant, product and extra state names if given in rate law
        self.reactants_names = reactants    
        self.products_names = products
        self.extra_states = extra_states

        # override values if provided
        self.parameters_values = parameters_values
        self.reactant_values = reactant_values
        self.product_values = product_values


    def get_antimony_reaction_str(self, r_index: str) -> str:
        '''
        generates an antimony string for the reaction, given the index of the reaction 
            r_index: str, represents reaction name in the system, usually an simple index 
        
        '''
        reactant_str = ' + '.join(self.reactants_names)
        product_str = ' + '.join(self.products_names)
        rate_law_str = self.archtype.rate_law 
        # rate law substitution needs to occur for reactants, products, extra states and parameters
        i = 0 
        while i < len(self.reactants_names):
            archtype_name = self.archtype.reactants[i]
            replacement_name = self.reactants_names[i]
            rate_law_str = rate_law_str.replace(archtype_name, replacement_name)
            i += 1 
            
        i = 0
        while i < len(self.products_names):
            archtype_name = self.archtype.products[i]
            replacement_name = self.products_names[i]
            rate_law_str = rate_law_str.replace(archtype_name, replacement_name)
            i += 1

        i = 0
        while i < len(self.archtype.extra_states):
            archtype_name = self.archtype.extra_states[i]
            replacement_name = self.extra_states[i]
            rate_law_str = rate_law_str.replace(archtype_name, replacement_name)
            i += 1

        i = 0
        while i < len(self.archtype.parameters):
            archtype_name = self.archtype.parameters[i]
            replacement_name = r_index + '_' + archtype_name
            rate_law_str = rate_law_str.replace(archtype_name, str(replacement_name))
            i += 1

        return f'{r_index}: {reactant_str} -> {product_str}; {rate_law_str}'

    def __str__(self) -> str:
        
        return self.get_antimony_reaction_str(r_index='react')