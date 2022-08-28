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
            direct_interactors = self._directly_connected_species(s)
            indirect_interactors = self._indirectly_connected_species(s)
            n_di = len(direct_interactors)
            for di in direct_interactors: 
                A = f'{s}_{di}'
                self.add_reaction(Reaction(michaelis_menten_fixed, (s,), (A,), reactant_values=n_di*100))

            for di in direct_interactors:
                # create michalean mapping between s and subspecies of s with 
                # fixed parameters for each subspecies 
                A = f'{s}_{di}'

                # create mass action binding reaction between s and di

                B = f'{di}_{s}'
                P = f'bound_{A}'
                if A not in completed_pairs and B not in completed_pairs:
                    self.add_reaction(Reaction(mass_action_21, (A, B), (P,)))
                    self.variables[f'TFB_{A}'] = f'TFB_{A} := {P}/100'
                    self.variables[f'TFB_{B}'] = f'TFB_{B} := {P}/100'
                    completed_pairs.append(A)
                    completed_pairs.append(B)

            for ii in indirect_interactors:
                # crux of the algorithm, A total frac bound to B (vice versa) is 
                TFB_A_B = f'TFB_{s}_{ii}'
                TFB_B_A = f'TFB_{ii}_{s}'

                TFB_A_B_expr = f'{TFB_A_B} := '
                # We must filter out nodes that are not connected to s, then build sum of 
                # frac bound of ii to s 
                direct_interactors_of_ii = self._directly_connected_species(ii)
                interactors_of_s = self._indirectly_connected_species(s) + self._directly_connected_species(s)
                set_of_interactors = self._common_elements_of_two_lists(direct_interactors_of_ii, interactors_of_s)
                for M in set_of_interactors:
                    TFB_A_B_expr += f'TFB_{M}_{s}*TFB_{ii}_{M} + '
                TFB_A_B_expr = TFB_A_B_expr[:-3]

                self.variables[TFB_A_B] = TFB_A_B_expr

                TFB_B_A_expr = f'{TFB_B_A} := '
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
        return super().get_antimony_model()



        

        
