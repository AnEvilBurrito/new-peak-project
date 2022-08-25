# The following function generates antimony strings from high level syntax

from typing import Dict
from .ModelBuilder import ModelBuilder
from .Reaction import Reaction
from .LinkedParameters import LinkedParameters as LP
from .ArchtypeCollections import *

class SimpleScaffoldBuilder(ModelBuilder):

    '''
    NOTE: HUGE assumptions are made here:
    Specie A binding to B does not affect binding of A to C, or B to C, this is of course 
    not true in reality, but it is a simplification that is made here. Thus, we assumed
    independent binding kinetics of A to B to C to approximate A binding to C.
    '''

    def __init__(self, name):
        
        super().__init__(name)
        self.scaffold_connections: Dict[str, list] = {}


    def add_reaction(self, reaction: Reaction):

        all_scaffolds = self.get_all_scaffolds()
        # check if the reactants and products are potentially scaffold proteins 
        # if so, we need to map the reaction to each scaffold sub-specie 

        # first, check if the reactants are scaffold proteins
        for reactant in reaction.reactants_names:
            if reactant in all_scaffolds:
                # then, find direct interactors of the scaffold protein 
                interactors = self._directly_connected_species(reactant)
                # create a new reaction for each interactor using linked
                # parameters
                A = reactant
                for M in interactors:
                    new_reaction = Reaction(reaction.archtype, (f'{A}_{M}',), reaction.products_names, linked_parameters=reaction.linked_parameters)
                    super().add_reaction(new_reaction)
                
        # TODO: finish implementing this function

        return super().add_reaction(reaction)

    def get_all_scaffolds(self):
        return list(self.scaffold_connections.keys())

    def rule_add_scaffold_reactions(self, scaffold: str, interactors: list):
        
        '''
        NOTE: at this stage, reactions have not been created yet 
        This function is essentially constructing a non-directed adjacency list
        scaffold: name of the scaffold 
        interactors: a list of str representing interactors of the scaffold 
        '''
        
        # first, check if the scaffold is already in the scaffold_connections
        if scaffold in self.scaffold_connections:
            # if so, check if the interactors are already in the scaffold_connections
            for interactor in interactors:
                if interactor not in self.scaffold_connections[scaffold]:
                    self.scaffold_connections[scaffold].append(interactor)
        else:
            self.scaffold_connections[scaffold] = interactors

        # then, check if the interactors are already in the scaffold_connections
        for interactor in interactors:
            if interactor in self.scaffold_connections:
                if scaffold not in self.scaffold_connections[interactor]:
                    self.scaffold_connections[interactor].append(scaffold)
            else:
                self.scaffold_connections[interactor] = [scaffold]

    def _directly_connected_species(self, s: str) -> list:

        '''
        Directly connected species to specie s must be in self.scaffold_connections[s]
        '''

        if s in self.scaffold_connections:
            return self.scaffold_connections[s]
        else:
            return []
    

    def _indirectly_connected_species(self, s: str) -> list: 

        '''
        Implement breadth first search (BFS) to traverse self.scaffold_connections and 
        find the non-directly connected species to specie s
        '''

        # first, check if the scaffold is already in the scaffold_connections
        if s not in self.scaffold_connections:
            return []

        # if so, check if the interactors are already in the scaffold_connections
        visited = set()
        queue = [s]
        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                queue.extend(self._directly_connected_species(node))
        return list(visited - set(self._directly_connected_species(s)) - set([s]))

    def rule_build_scaffold_connections(self):
        '''
        Building reactions, as well as additional variables using existing
        connections listed in self.scaffold_connections
        '''

        completed_pairs = []
    
        for s in self.scaffold_connections:
            self.variables[s] = f'{s} = 0'
            direct_interactors = self._directly_connected_species(s)
            indirect_interactors = self._indirectly_connected_species(s)
            for di in direct_interactors:
                A, B = f'{s}_{di}', f'{di}_{s}'
                P = f'bound_{A}'
                if A not in completed_pairs and B not in completed_pairs:
                    self.add_reaction(Reaction(mass_action_21, (A, B), (P,)))
                    self.variables[f'TFB_{A}'] = f'TFB_{A} = {P}/100'
                    self.variables[f'TFB_{B}'] = f'TFB_{B} = {P}/100'
                    completed_pairs.append(A)
                    completed_pairs.append(B)

            for ii in indirect_interactors:
                # crux of the algorithm, A total frac bound to B (vice versa) is 
                TFB_A_B = f'TFB_{s}_{ii}'
                TFB_B_A = f'TFB_{ii}_{s}'

                TFB_A_B_expr = f'{TFB_A_B} = '
                # We must filter out nodes that are not connected to s, then build sum of 
                # frac bound of ii to s 
                direct_interactors_of_ii = self._directly_connected_species(ii)
                interactors_of_s = self._indirectly_connected_species(s) + self._directly_connected_species(s)
                set_of_interactors = self._common_elements_of_two_lists(direct_interactors_of_ii, interactors_of_s)
                for M in set_of_interactors:
                    TFB_A_B_expr += f'TFB_{M}_{s}*TFB_{ii}_{M} + '
                TFB_A_B_expr = TFB_A_B_expr[:-3]

                self.variables[TFB_A_B] = TFB_A_B_expr

                TFB_B_A_expr = f'{TFB_B_A} = '
                # We must filter out nodes that are not connected to ii, then build sum of
                # frac bound of s to ii
                direct_interactors_of_s = self._directly_connected_species(s)
                interactors_of_ii = self._indirectly_connected_species(ii) + self._directly_connected_species(ii)
                set_of_interactors_reverse = self._common_elements_of_two_lists(direct_interactors_of_s, interactors_of_ii)
                for M in set_of_interactors_reverse:
                    TFB_B_A_expr += f'TFB_{M}_{ii}*TFB_{s}_{M} + '
                TFB_B_A_expr = TFB_B_A_expr[:-3]
                
                self.variables[TFB_B_A] = TFB_B_A_expr

    def _common_elements_of_two_lists(self, l1, l2):
        return list(set(l1) & set(l2))

    def get_antimony_model(self):
        '''
        run build before creating antimony model 
        '''
        self.rule_build_scaffold_connections()
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



        

        
