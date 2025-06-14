from models.ModelBuilder import ModelBuilder
from models.Reaction import Reaction
from models.ReactionArchtype import ReactionArchtype
from models.ArchtypeCollections import *
from dataclasses import dataclass


import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from joblib import Parallel, delayed

# create a python dataclass called regulation which stores from and to species and the type of regulation
@dataclass 
class Regulation:
    """Class to store the regulation information"""
    from_specie: str
    to_specie: str
    reg_type: str

class ModelSpecification:

    def __init__(self):
        self.A_species = []
        self.B_species = []
        self.C_species = []
        self.regulations = []
        self.randomise_parameters = True
        self.regulation_types = []
        self.C_weak_stimulators = 0
        self.C_strong_stimulators = 0
        self.C_allosteric_inhibitors = 0
        self.C_competitive_inhibitors = 0
        
    def __str__(self):
        # return a string representation of the object, which is its current states 
        return f'A Species: {self.A_species}\n' + \
                f'B Species: {self.B_species}\n' + \
                f'C Species: {self.C_species}\n' + \
                f'Regulations: {self.regulations}\n' + \
                f'Regulation Types: {self.regulation_types}\n'

    def generate_specifications_old(self, random_seed, NA, NR, verbose=1):
        # WARNING: This method is deprecated and should not be used
        random_seed_number = random_seed  # None if do not want to fix the seed
        if random_seed_number is not None:
            np.random.seed(random_seed_number)
            
        if verbose == 1:
            print('--- Generating a random network ---')
            print(f'Random Seed: {random_seed_number}')
            print(f'Number of A Species: {NA}')
            print(f'Number of B Species: {NA}')
            print(f'Number of C Species: 1')
            print(f'Number of Regulations: {NR}')
            print('\n')

        # based on the `NA` parameter, create a number of species for A

        A_species = [f'A{i}' for i in range(NA)]
        B_species = [f'B{i}' for i in range(NA)]
        C_species = ['C']

        self.A_species = A_species
        self.B_species = B_species
        self.C_species = C_species
        
        # based on the number of NR, create random connections between any two species in the network
        regulation_types_choice = ['up', 'down']
        regulations = []
        reg_types = []

        all_species = A_species + B_species + C_species
        B_and_C = B_species + C_species

        # max attempts is the permutation of specie C and B * 100
        max_attempt = len(B_and_C)**2 * 100
        current_attempt = 0
        while len(regulations) < NR and current_attempt < max_attempt:
            from_specie = np.random.choice(B_and_C)
            to_specie = np.random.choice(all_species)
            reg = (from_specie, to_specie)
            reverse_reg = (to_specie, from_specie)

            # also exclude self-regulations and B -> C regulations
            if from_specie == to_specie or (from_specie in B_species and to_specie in C_species):
                continue

            if reg not in regulations and reverse_reg not in regulations:
                reg_type = np.random.choice(regulation_types_choice)
                regulations.append(reg)
                reg_types.append(reg_type)


        if current_attempt == max_attempt:
            print('Failed to generate the network in the given number of attempts, max attempt:', max_attempt)
            exit(1)

        for i, reg in enumerate(regulations):
            if verbose == 1:
                print(f'Feedback Regulation {i}: {reg} - {reg_types[i]}')

        # each Ap index affects every B -> Bp reaction index
        for i in range(NA):
            regulations.append((f'A{i}', f'B{i}'))
            reg_types.append('up')
            if verbose == 1:
                print(f'A to B Stimulation {i+NR}: {f"A{i}"} - {f"B{i}"} - up')
                
        self.regulations = regulations
        self.regulation_types = reg_types
        
        # randomise the number of stimulators and inhibitors for len(B_species) 

        stimulator_number = np.random.randint(0, len(B_species)+1)
        if stimulator_number == 0:
            strong_stimulators = 0
            weak_stimulators = 0
        else:
            strong_stimulators = np.random.randint(0, stimulator_number)
            weak_stimulators = stimulator_number - strong_stimulators
            
        strong_stimulators = 0
        weak_stimulators = stimulator_number - strong_stimulators
            
        inhibitor_number = len(B_species) - stimulator_number 
        if inhibitor_number == 0:
            allosteric_inhibitors = 0
            competitive_inhibitors = 0 
        else:
            allosteric_inhibitors = np.random.randint(0, inhibitor_number)
            competitive_inhibitors = inhibitor_number - allosteric_inhibitors

        if verbose == 1:
            print(f'Stimulators: {stimulator_number}, Inhibitors: {inhibitor_number}')  
            print(f'Strong Stimulators: {strong_stimulators}, Weak Stimulators: {weak_stimulators}')
            print(f'Allosteric Inhibitors: {allosteric_inhibitors}, Competitive Inhibitors: {competitive_inhibitors}')
            
        self.C_weak_stimulators = weak_stimulators
        self.C_strong_stimulators = strong_stimulators
        self.C_allosteric_inhibitors = allosteric_inhibitors
        self.C_competitive_inhibitors = competitive_inhibitors

    def generate_specifications(self, random_seed, NA, NR, verbose=1):
        # systematically generate C specie regulations insead of defining varialbles for each type of regulation
        random_seed_number = random_seed  # None if do not want to fix the seed
        if random_seed_number is not None:
            rng = np.random.default_rng(random_seed_number)
            
        if verbose == 1:
            print('--- Generating a random network ---')
            print(f'Random Seed: {random_seed_number}')
            print(f'Number of A Species: {NA}')
            print(f'Number of B Species: {NA}')
            print(f'Number of C Species: 1')
            print(f'Number of Regulations: {NR}')
            print('\n')

        # based on the `NA` parameter, create a number of species for A

        A_species = [f'A{i}' for i in range(NA)]
        B_species = [f'B{i}' for i in range(NA)]
        C_species = ['C']

        self.A_species = A_species
        self.B_species = B_species
        self.C_species = C_species
        
        # based on the number of NR, create random connections between any two species in the network
        regulation_types_choice = ['up', 'down']
        regulations = []
        reg_types = []

        all_species = A_species + B_species + C_species
        B_and_C = B_species + C_species

        # max attempts is the permutation of specie C and B * 100
        max_attempt = len(B_and_C)**2 * 100
        current_attempt = 0
        while len(regulations) < NR and current_attempt < max_attempt:
            from_specie = str(rng.choice(B_and_C))
            to_specie = str(rng.choice(all_species))
            reg = (from_specie, to_specie)
            reverse_reg = (to_specie, from_specie)

            # also exclude self-regulations and B -> C regulations
            if from_specie == to_specie or (from_specie in B_species and to_specie in C_species):
                continue

            if reg not in regulations and reverse_reg not in regulations:
                reg_type = str(rng.choice(regulation_types_choice))
                regulations.append(reg)
                reg_types.append(reg_type)


        if current_attempt == max_attempt:
            print('Failed to generate the network in the given number of attempts, max attempt:', max_attempt)
            exit(1)

        for i, reg in enumerate(regulations):
            if verbose == 1:
                print(f'Feedback Regulation {i}: {reg} - {reg_types[i]}')

        # each Ap index affects every B -> Bp reaction index
        for i in range(NA):
            regulations.append((f'A{i}', f'B{i}'))
            reg_types.append('up')
            if verbose == 1:
                print(f'A to B Stimulation {i+NR}: {f"A{i}"} - {f"B{i}"} - up')
                
        stimulator_number = rng.integers(0, len(B_species)+1)
        inhibitor_number = len(B_species) - stimulator_number
        # generate b to c regulations
        for i in range(stimulator_number):
            regulations.append((f'B{i}', 'C'))
            reg_types.append('up')
            if verbose == 1:
                print(f'B to C Stimulation {i+NR+NA}: {f"B{i}"} - C - up')
        for i in range(inhibitor_number):
            regulations.append((f'B{stimulator_number+i}', 'C'))
            reg_types.append('down')
            if verbose == 1:
                print(f'B to C Inhibition {i+NR+NA+stimulator_number}: {f"B{i}"} - C - down')

        self.regulations = regulations
        self.regulation_types = reg_types

    def generate_archtype_and_regulators(self, specie):

        all_regulations = self.regulations
        all_regulation_types = self.regulation_types
        
        regulators_for_specie = []
        for i, reg in enumerate(all_regulations):
            if reg[1] == specie:
                reg_type = all_regulation_types[i]
                regulators_for_specie.append((reg[0], reg_type))

        if len(regulators_for_specie) == 0:
            return michaelis_menten, ()

        total_up_regulations = len([r for r in regulators_for_specie if r[1] == 'up'])
        total_down_regulations = len([r for r in regulators_for_specie if r[1] == 'down'])

        rate_law = create_archtype_michaelis_menten(stimulators=0,
                                                    stimulator_weak=total_up_regulations,
                                                    allosteric_inhibitors=0,
                                                    competitive_inhibitors=total_down_regulations)

        # sort the regulators by type, up first and down second
        regulators_for_specie = sorted(regulators_for_specie, key=lambda x: x[1], reverse=True)
        regulators_sorted = [r[0] for r in regulators_for_specie]
        # regulators_sorted_phos = [r[0]+'p' for r in regulators_for_specie]
        regulators_sorted_modified = []
        for r in regulators_sorted:
            if 'D' in r:
                regulators_sorted_modified.append(r)
            else:
                regulators_sorted_modified.append(r+'p')
        # print(f'Sorted regulators information: {regulators_for_specie}')
        # print(f'Final regulators for {specie}: {regulators_sorted_phos}')
        # print(f'Rate law for {specie}: {rate_law}')
        return rate_law, regulators_sorted_modified


    # generate random parameters informed by a scale
    def generate_random_parameters(self, reaction_archtype: ReactionArchtype, scale_range, multiplier_range, random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)

        assumed_values = reaction_archtype.assume_parameters_values
        # print(f'Assumed values: {assumed_values}')
        r_params = []
        for key, value in assumed_values.items():
            rand = rng.uniform(value*scale_range[0], value*scale_range[1])
            rand *= rng.uniform(multiplier_range[0], multiplier_range[1])
            r_params.append(rand)

        return tuple(r_params)

    def generate_network_old(self, network_name, mean_range_species, rangeScale_params, rangeMultiplier_params, verbose=1, random_seed=None) -> ModelBuilder:
        
        '''
        Returns a pre-compiled ModelBuilder object with the given specifications, 
        ready to be simulated. Pre-compiled model allows the user to manually set the initial values of the species
        before compiling to Antimony or SBML. 
        Parameters:
            network_name: str, the name of the network
            mean_range_species: tuple, the range of the mean values for the species
            rangeScale_params: tuple, the range of the scale values for the parameters
            rangeMultiplier_params: tuple, the range of the multiplier values for the parameters
            verbose: int, the verbosity level of the function
            random_seed: int, the random seed to use for reproducibility
        '''
        
        model = ModelBuilder(network_name)

        # fix np random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # convert a list of species to a tuple of species
        B_species_tuple_phos = []
        for b in self.B_species:
            b_specie_phos = b + 'p'
            B_species_tuple_phos.append(b_specie_phos)

        B_species_tuple_phos = tuple(B_species_tuple_phos)


        '''A Specie reactions'''
        for specie in self.A_species:

            # create the rate law for the specie
            rate_law, regulators = self.generate_archtype_and_regulators(specie)

            # generate a random set of parameters for reaction A -> Ap
            r_params = michaelis_menten.assume_parameters_values.values()
            if self.randomise_parameters:
                r_params = self.generate_random_parameters(michaelis_menten, rangeScale_params, rangeMultiplier_params)

            # add the reaction Ap -> A to the model
            model.add_reaction(Reaction(michaelis_menten, (specie+'p',), (specie,), parameters_values=tuple(r_params), zero_init=False))

            # generate a random initial value for A
            random_mean = np.random.randint(mean_range_species[0], mean_range_species[1])

            # generate a random set of parameters for reaction Ap -> A
            r_params_reverse = rate_law.assume_parameters_values.values()
            if self.randomise_parameters:
                r_params_reverse = self.generate_random_parameters(rate_law, rangeScale_params, rangeMultiplier_params)

            # add the reaction Ap -> A to the model
            model.add_reaction(Reaction(rate_law, (specie,), (specie+'p',),
                                        reactant_values=random_mean,
                                        extra_states=regulators,
                                        parameters_values=tuple(r_params_reverse), zero_init=False))

        '''B Specie reactions'''

        for specie in self.B_species:
            # create the rate law for the specie
            rate_law, regulators = self.generate_archtype_and_regulators(specie)

            # generate a random set of parameters for reaction B -> Bp
            r_params = michaelis_menten.assume_parameters_values.values()
            if self.randomise_parameters:
                r_params = self.generate_random_parameters(michaelis_menten, rangeScale_params, rangeMultiplier_params)

            # add the reaction Bp -> B to the model
            model.add_reaction(Reaction(michaelis_menten, (specie+'p',), (specie,),
                                        parameters_values=tuple(r_params), zero_init=False))

            # generate a random initial value for B
            random_mean = np.random.randint(mean_range_species[0], mean_range_species[1])

            # generate a random set of parameters for reaction B -> Bp
            r_params_reverse = rate_law.assume_parameters_values.values()
            if self.randomise_parameters:
                r_params_reverse = self.generate_random_parameters(rate_law, rangeScale_params, rangeMultiplier_params)

            # add the reaction B -> Bp to the model
            model.add_reaction(Reaction(rate_law, (specie,), (specie+'p',),
                                        reactant_values=random_mean,
                                        extra_states=regulators,
                                        parameters_values=tuple(r_params_reverse), zero_init=False))


        '''C Specie reactions'''
        # randomise the number of stimulators and inhibitors for len(B_species)
        rate_law_C = create_archtype_michaelis_menten(stimulators=self.C_strong_stimulators,
                                                    stimulator_weak=self.C_weak_stimulators,
                                                    allosteric_inhibitors=self.C_allosteric_inhibitors,
                                                    competitive_inhibitors=self.C_competitive_inhibitors)

        c_params = rate_law_C.assume_parameters_values.values()
        if self.randomise_parameters:
            c_params = self.generate_random_parameters(rate_law_C, rangeScale_params, rangeMultiplier_params)

        for specie in self.C_species:
            model.add_reaction(Reaction(rate_law_C, (specie,), (specie+'p',),
                            extra_states=B_species_tuple_phos, parameters_values=tuple(c_params), zero_init=False))
            model.add_reaction(Reaction(michaelis_menten, (specie+'p',),
                            (specie,), reactant_values=0, product_values=100, zero_init=False))

        model.precompile()
        # add stimulation reactions
        if verbose == 1:
            print('Model States: ', len(model.states))
            print('Model Parameters: ', len(model.parameters))
            print('Model Reactions: ', len(model.reactions))
            print('\n')
            print('--- Antimony Model ---')
            print('\n')
            print(model.get_antimony_model())
            print('\n')
            
        return model 
    
    def generate_network(self, network_name, mean_range_species, rangeScale_params, rangeMultiplier_params, verbose=1, random_seed=None) -> ModelBuilder:
        '''
        Returns a pre-compiled ModelBuilder object with the given specifications, 
        ready to be simulated. Pre-compiled model allows the user to manually set the initial values of the species
        before compiling to Antimony or SBML. 
        Parameters:
            network_name: str, the name of the network
            mean_range_species: tuple, the range of the mean values for the species
            rangeScale_params: tuple, the range of the scale values for the parameters
            rangeMultiplier_params: tuple, the range of the multiplier values for the parameters
            verbose: int, the verbosity level of the function
            random_seed: int, the random seed to use for reproducibility
        '''
        model = ModelBuilder(network_name)

        # fix np random seed for reproducibility
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        
        # convert a list of species to a tuple of species
        B_species_tuple_phos = []
        for b in self.B_species:
            b_specie_phos = b + 'p'
            B_species_tuple_phos.append(b_specie_phos)

        B_species_tuple_phos = tuple(B_species_tuple_phos)


        '''A Specie reactions'''
        for specie in self.A_species:

            # create the rate law for the specie
            rate_law, regulators = self.generate_archtype_and_regulators(specie)

            # generate a random set of parameters for reaction A -> Ap
            r_params = michaelis_menten.assume_parameters_values.values()
            if self.randomise_parameters:
                r_params = self.generate_random_parameters(michaelis_menten, rangeScale_params, rangeMultiplier_params, random_seed=random_seed)

            # add the reaction Ap -> A to the model
            model.add_reaction(Reaction(michaelis_menten, (specie+'p',), (specie,), parameters_values=tuple(r_params), zero_init=False))

            # generate a random initial value for A
            random_mean = rng.integers(mean_range_species[0], mean_range_species[1])

            # generate a random set of parameters for reaction Ap -> A
            r_params_reverse = rate_law.assume_parameters_values.values()
            if self.randomise_parameters:
                r_params_reverse = self.generate_random_parameters(rate_law, rangeScale_params, rangeMultiplier_params, random_seed=random_seed)

            # add the reaction Ap -> A to the model
            model.add_reaction(Reaction(rate_law, (specie,), (specie+'p',),
                                        reactant_values=random_mean,
                                        extra_states=regulators,
                                        parameters_values=tuple(r_params_reverse), zero_init=False))

        '''B Specie reactions'''

        for specie in self.B_species:
            # create the rate law for the specie
            rate_law, regulators = self.generate_archtype_and_regulators(specie)

            # generate a random set of parameters for reaction Bp -> B
            r_params = michaelis_menten.assume_parameters_values.values()
            if self.randomise_parameters:
                r_params = self.generate_random_parameters(michaelis_menten, rangeScale_params, rangeMultiplier_params, random_seed=random_seed)

            # add the reaction Bp -> B to the model
            model.add_reaction(Reaction(michaelis_menten, (specie+'p',), (specie,),
                                        parameters_values=tuple(r_params), zero_init=False))

            # generate a random initial value for B
            random_mean = rng.integers(mean_range_species[0], mean_range_species[1])

            # generate a random set of parameters for reaction B -> Bp
            r_params_reverse = rate_law.assume_parameters_values.values()
            if self.randomise_parameters:
                r_params_reverse = self.generate_random_parameters(rate_law, rangeScale_params, rangeMultiplier_params, random_seed=random_seed)

            # add the reaction B -> Bp to the model
            model.add_reaction(Reaction(rate_law, (specie,), (specie+'p',),
                                        reactant_values=random_mean,
                                        extra_states=regulators,
                                        parameters_values=tuple(r_params_reverse), zero_init=False))
            
        '''C Specie reactions'''
        C_specie = 'C'    
        rate_law, regulators = self.generate_archtype_and_regulators(C_specie)
        
        # generate a random set of parameters for reaction C -> Cp, using the rate law
        c_params = rate_law.assume_parameters_values.values()
        if self.randomise_parameters:
            c_params = self.generate_random_parameters(rate_law, rangeScale_params, rangeMultiplier_params, random_seed=random_seed)
            
        # add the reaction C -> Cp to the model
        model.add_reaction(Reaction(rate_law, (C_specie,), (C_specie+'p',),
                                    extra_states=regulators, parameters_values=tuple(c_params), zero_init=False))
        
        # generate a random set of parameters for reaction Cp -> C, using the michaelis menten rate law
        r_params_reverse = michaelis_menten.assume_parameters_values.values()
        if self.randomise_parameters:
            r_params_reverse = self.generate_random_parameters(michaelis_menten, rangeScale_params, rangeMultiplier_params, random_seed=random_seed)
            
        # add the reaction Cp -> C to the model
        model.add_reaction(Reaction(michaelis_menten, (C_specie+'p',), (C_specie,),
                                    reactant_values=0, product_values=100,
                                    parameters_values=tuple(r_params_reverse), zero_init=False))
        
        model.precompile()
        # add stimulation reactions
        if verbose == 1:
            print('Model States: ', len(model.states))
            print('Model Parameters: ', len(model.parameters))
            print('Model Reactions: ', len(model.reactions))
            print('\n')
            print('--- Antimony Model ---')
            print('\n')
            print(model.get_antimony_model())
            print('\n')

        return model
    
    def get_regulations(self):
        '''
        extracts the regulations and their types from the model, as lists of tuples
        '''
        regulations = []
        for i, reg in enumerate(self.regulations):
            regulation = (reg[0], reg[1])
            regulation_type = self.regulation_types[i]
            regulations.append((regulation, regulation_type))
        return regulations
        
    def get_feedback_regulations(self):
        
        '''
        extracts the feedback regulations from the model regulations
        '''
        
        feedback_regs = []
        for i, reg in enumerate(self.regulations):
            feedback = True
            specie_1 = reg[0]
            specie_2 = reg[1]
            if 'A' in specie_1:
                # exact A -> B regulations are not feedback regulations
                number_1 = int(specie_1[1:])
                number_2 = int(specie_2[1:])
                if number_1 == number_2 and 'B' in specie_2:
                    feedback = False
            if 'B' in specie_1:
                # B -> C forward regulations are not feedback regulations
                if 'C' in specie_2:
                    feedback = False
            if 'D' in specie_1:
                # drug regulations are not feedback regulations
                feedback = False
            if feedback:
                feedback_regs.append((reg, self.regulation_types[i]))
        return feedback_regs
        
    def remove_regulation(self, reg, reg_type):
        
        '''
        removes a regulation from the model
        '''
        
        if reg in self.regulations:
            index = self.regulations.index(reg)
            # check if the regulation type matches
            if self.regulation_types[index] == reg_type:
                self.regulations.pop(index)
                self.regulation_types.pop(index)
        else: 
            raise ValueError(f'Regulation {reg} {reg_type} not found in the model spec')
            
    def add_regulation(self, reg, reg_type):
        
        '''
        adds a regulation to the model
        '''
        
        if reg not in self.regulations:
            self.regulations.append(reg)
            self.regulation_types.append(reg_type)
        else: 
            raise ValueError(f'Regulation {reg} already exists in the model spec')

### Helper functions for generating data
def manual_reset(runner_model, initial_values):
    for s, v in initial_values.items():
        runner_model[f'init({s})'] = v
        sp = s + 'p'
        runner_model[f'init({sp})'] = 0
    return runner_model

def get_model_initial_values(model_spec: ModelSpecification, runner_model):
    '''
    Get the initial values of the model species
        This function should be called after the model has been compiled 
        to preserve the initial values before any perturbations
    '''
    initial_values = {}
    for s in model_spec.A_species:
        initial_values[s] = runner_model.getValue(f'init({s})')

    for s in model_spec.B_species:
        initial_values[s] = runner_model.getValue(f'init({s})')
        
    for s in model_spec.C_species:
        initial_values[s] = runner_model.getValue(f'init({s})')
    return initial_values

def print_model_initial_values(model_spec: ModelSpecification, runner):
    # print the model states and parameter values
    for s in model_spec.A_species:
        print(f'{s}: {runner[s]} {runner[f"init({s})"]}')
        sp = s + 'p'
        print(f'{sp}: {runner[sp]} {runner[f"init({sp})"]}')
    for s in model_spec.B_species:
        print(f'{s}: {runner[s]} {runner[f"init({s})"]}')
        sp = s + 'p'
        print(f'{sp}: {runner[sp]} {runner[f"init({sp})"]}')
    for s in model_spec.C_species:
        print(f'{s}: {runner[s]} {runner[f"init({s})"]}')
        sp = s + 'p'
        print(f'{sp}: {runner[sp]} {runner[f"init({sp})"]}')

    # print the model parameters
    for r in runner.getModel().getGlobalParameterIds():
        print(f'{r}: {runner[r]}')


### Generate feature and target data
def generate_feature_data(model_spec: ModelSpecification, runner_model, perturbation_type: str, perturbation_params, n, seed=None):
    '''
    Generate a dataframe of perturbed values for the model
        model_spec: ModelSpecification object   
        model: roadrunner model object
        perturbation_type: str, the type of perturbation to apply, either 'uniform' or 'gaussian'
        perturbation_params: dict of parameters for the perturbation, for
            'uniform' perturbation, the params are {'min': float, 'max': float}
            'gaussian' perturbation, the param is either {'std': float} or {'rsd': float}
                'rsd' is the relative standard deviation of the perturbation, i.e. std/mean
        n: int, the number of perturbations to generate
    ''' 
    # validate parameters
    if perturbation_type not in ['uniform', 'gaussian']:
        raise ValueError('Perturbation type must be either "uniform" or "gaussian"')
    
    if perturbation_type == 'uniform':
        if 'min' not in perturbation_params or 'max' not in perturbation_params:
            raise ValueError('For uniform perturbation, the parameters must contain "min" and "max" keys')
    elif perturbation_type == 'gaussian':
        # either 'std' or 'rsd' must be in the parameters
        if 'std' not in perturbation_params and 'rsd' not in perturbation_params:
            raise ValueError('For gaussian perturbation, the parameters must contain "std" or "rsd" keys')
    
    # set the random seed if provided, for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    # grab the initial values of all A species
    initial_values = {}
    for s in model_spec.A_species:
        initial_values[s] = runner_model.getValue(f'init({s})')
        # print(s, runner[s])

    for s in model_spec.B_species:
        initial_values[s] = runner_model.getValue(f'init({s})')

    all_perturbed_values = []
    for _ in range(n):
        perturbed_values = {}
        if perturbation_type == 'uniform':
            min_ = perturbation_params['min']
            max_ = perturbation_params['max']  
            for s in model_spec.A_species:
                perturbed_values[s] = initial_values[s] * np.random.uniform(min_, max_)
            for s in model_spec.B_species:
                perturbed_values[s] = initial_values[s] * np.random.uniform(min_, max_)
            all_perturbed_values.append(perturbed_values)
        elif perturbation_type == 'gaussian':
            if 'std' in perturbation_params:
                sigma = perturbation_params['std']
            elif 'rsd' in perturbation_params:
                rsd = perturbation_params['rsd']
                sigma = rsd * initial_values[s]
            for s in model_spec.A_species:
                mu = initial_values[s]
                perturbed_values[s] = np.random.normal(mu, sigma)
            for s in model_spec.B_species:
                mu = initial_values[s]
                perturbed_values[s] = np.random.normal(mu, sigma)
            all_perturbed_values.append(perturbed_values)
        
    # put the perturbed values into a dataframe
    feature_df = pd.DataFrame(all_perturbed_values)
    return feature_df


def generate_target_data(model_spec, runner_model, feature_df, initial_values, simulation_params={'start': 0, 'end': 500, 'points': 100}):
    '''
    Generate the target data for the model
        model_spec: ModelSpecification object   
        model: roadrunner model object
        feature_df: dataframe of perturbed values
        simulation_params: dict of parameters for the simulation, for
            'start': float, the start time of the simulation
            'end': float, the end time of the simulation
            'points': int, the number of points to simulate
    Returns:
        target_df: dataframe of the target values
        time_course_data: list of the time course data for each perturbation
    '''
    # validate the simulation parameters
    if 'start' not in simulation_params or 'end' not in simulation_params or 'points' not in simulation_params:
        raise ValueError('Simulation parameters must contain "start", "end" and "points" keys')
    
    # iterate the dataframe and simulate each perturbation
    all_perturbed_results = []
    time_course_data = []

    for i in range(feature_df.shape[0]):
        # Reset rr model and simulate with each perturbation
        runner_model.reset()
        runner_model = manual_reset(runner_model, initial_values)
        perturbed_values = feature_df.iloc[i]

        # set the perturbed values
        for s in model_spec.A_species:
            runner_model[f'init({s})'] = perturbed_values[s]
            
        for s in model_spec.B_species:
            runner_model[f'init({s})'] = perturbed_values[s]

        # simulate the model and grab only the C and Cp values at the end
        start, end, points = simulation_params['start'], simulation_params['end'], simulation_params['points']
        res = runner_model.simulate(start, end, points)
        perturbed_results = {}
        for c in model_spec.C_species:
            perturbed_results[f'{c}p'] = res[f'[{c}p]'][-1]
        all_perturbed_results.append(perturbed_results)
        
        # store the run of Cp into time_course_data
        time_course_data.append(res['[Cp]'])

    runner_model = manual_reset(runner_model, initial_values)
    target_df = pd.DataFrame(all_perturbed_results)
    return target_df, time_course_data

def generate_model_timecourse_data(model_spec, runner_model, feature_df, initial_values, simulation_params={'start': 0, 'end': 500, 'points': 100}, capture_species='all'):
    # validate the simulation parameters
    if 'start' not in simulation_params or 'end' not in simulation_params or 'points' not in simulation_params:
        raise ValueError(
            'Simulation parameters must contain "start", "end" and "points" keys')

    # iterate the dataframe and simulate each perturbation
    all_outputs = []

    for i in range(feature_df.shape[0]):
        output = {}
        # Reset rr model and simulate with each perturbation
        runner_model.reset()
        runner_model = manual_reset(runner_model, initial_values)
        perturbed_values = feature_df.iloc[i]

        # set the perturbed values
        for s in model_spec.A_species:
            runner_model[f'init({s})'] = perturbed_values[s]

        for s in model_spec.B_species:
            runner_model[f'init({s})'] = perturbed_values[s]

        # simulate the model and grab only the C and Cp values at the end
        start, end, points = simulation_params['start'], simulation_params['end'], simulation_params['points']
        res = runner_model.simulate(start, end, points)

        if capture_species == 'all':
            all_species = model_spec.A_species + model_spec.B_species + model_spec.C_species
            for s in all_species:
                output[s] = res[f'[{s}]']
                sp = s + 'p'
                output[sp] = res[f'[{sp}]']
        else:
            for s in capture_species:
                output[s] = res[f'[{s}]']
                sp = s + 'p'
                output[sp] = res[f'[{sp}]']
        all_outputs.append(output)

    runner_model = manual_reset(runner_model, initial_values)
    output_df = pd.DataFrame(all_outputs)
    return output_df

# Engineering Data Processing Methods 

def last_time_point_method(time_course_data, selected_species = None):
    if selected_species is None:
        selected_species = time_course_data.columns
    else:
        selected_species = selected_species
    selected_time_course_data = time_course_data[selected_species]
    last_time_points = selected_time_course_data.map(lambda x: x[-1])
    return last_time_points

def get_dynamic_features(col_data: pd.Series, 
                         normalise: bool = True,
                         abs_change_tolerance: float = 0.01) -> list:
    
    # dynamic features
    auc = np.trapz(col_data)
    max_val = np.max(col_data)
    max_time = np.argmax(col_data)
    min_val = np.min(col_data)
    min_time = np.argmin(col_data)

    median_val = np.median(col_data)

    # calculation of total fold change (tfc)
    start_val = col_data.iloc[0]
    end_val = col_data.iloc[-1]

    tfc = 0 
    if start_val == 0:
        tfc = 1000
    else:
        if end_val - start_val >= 0:
            tfc = (end_val - start_val) / start_val
        elif end_val - start_val < 0:
            if end_val == 0:
                tfc = -1000
            else:
                tfc = -((start_val - end_val) / end_val)

    # calculation of time to stability (tsv)
    tsv = len(col_data)
    while tsv > 1:
        if abs(col_data.iloc[tsv-1] - col_data.iloc[tsv-2]) < abs_change_tolerance:
            tsv -= 1
        else:
            tsv_value = col_data.iloc[tsv-1]
            break
    if tsv == 1:
        tsv_value = col_data.iloc[0]

    max_sim_time = len(col_data)
    n_auc = auc / max_sim_time
    n_max_time = max_time / max_sim_time
    n_min_time = min_time / max_sim_time
    n_tsv = tsv / max_sim_time
    
    if not normalise:
        # reset the values to the original values
        n_auc = auc
        n_max_time = max_time
        n_min_time = min_time
        n_tsv = tsv 

    dynamic_features = [n_auc, median_val, tfc, n_max_time,
                        max_val, n_min_time, min_val, n_tsv, tsv_value, start_val]

    return dynamic_features


def dynamic_features_method(time_course_data, selected_features=None, n_cores=1, verbose=0):
    if selected_features is None:
        selected_features = time_course_data.columns
    else:
        selected_features = selected_features
    
    # use parallel processing to speed up the calculation with tqdm
    def process_data(row_data):
        row_dynamic_features = []
        for feature in selected_features:
            col_data = row_data[feature]
            # convert to pd Series for easier manipulation
            col_data = pd.Series(col_data)
            dyn_feats = get_dynamic_features(col_data)
            row_dynamic_features.extend(dyn_feats)
        return row_dynamic_features
    
    if n_cores > 1 or n_cores == -1:
        all_dynamic_features = Parallel(n_jobs=n_cores)(delayed(process_data)(time_course_data.iloc[i]) for i in tqdm(range(time_course_data.shape[0]), desc="Calculating dynamic features", disable=verbose==0))
    else:
        all_dynamic_features = []
        # iterate each row in the time course data
        for i in tqdm(range(time_course_data.shape[0]), desc="Calculating dynamic features", disable=verbose==0):
            row_dynamic_features = []
            row_data = time_course_data.iloc[i]
            for feature in selected_features:
                col_data = row_data[feature]
                # convert to pd Series for easier manipulation
                col_data = pd.Series(col_data)
                dyn_feats = get_dynamic_features(col_data)
                row_dynamic_features.extend(dyn_feats)
            all_dynamic_features.append(row_dynamic_features)

    dynamic_feature_label = ['auc', 'median', 'tfc', 'tmax', 'max', 'tmin', 'min', 'ttsv', 'tsv', 'init']    
    new_df = pd.DataFrame(all_dynamic_features, columns=[s + '_' + dynamic_feature for s in selected_features for dynamic_feature in dynamic_feature_label], index=time_course_data.index)

    return new_df


def qualitative_features_method(time_course_data, selected_features=None):
    '''
    This method identifys the qualitative features of the time course data into specific categories
        - [feature]_inc: overall increasing trend of a time course, 0 if false and 1 if true
        - [feature]_dec: overall decreasing trend of a time course, 0 if false and 1 if true
        - [feature]_rebound: time course decreases below initla value but increases later, 0 if false and 1 if true
        - [feature]_overshoot: time course increases first above the initial value but decreases  
    '''
    if selected_features is None:
        selected_features = time_course_data.columns
    else:
        selected_features = selected_features


### Suboptimal Model Generation Methods 

def specific_pruning(model_spec: ModelSpecification, pruned_regulations):
    '''
    Prune specific regulations from the model based on the input list of regulations
    '''
    pass 


def systematic_edge_pruning(old_model_spec: ModelSpecification, old_model: ModelBuilder, edge_number: int, random_seed=42): 
    '''
    Prune the model by removing `edge_number` number of regulations from the model_spec randomly 
    '''
    # fix the random seed for reproducibility
    np.random.seed(random_seed)
    
    regulations = deepcopy(old_model_spec.regulations)
    regulation_types = deepcopy(old_model_spec.regulation_types)
    new_model_spec = deepcopy(old_model_spec)
    
    regulations_without_D = [r for r in regulations if 'D' not in r[0]]
    
    assert len(regulations_without_D) > edge_number + 1, "Error in systematic edge pruning: The `number of regulations that is not drug regulations` + 1 to prune must be less than the total number of regulations in the original model specification"
    assert edge_number > 0, "Error in systematic edge pruning: The number of regulations to prune must be greater than 0"
    
    reg_deleted = []

    i = 0 
    while i < edge_number:
        # randomly select an index to remove
        index = np.random.randint(0, len(regulations))
        # do not remove drug regulations
        if 'D' in regulations[index][0]: 
            # TODO: this is inefficient, a better method would be to filter D species out of selection 
            continue
        reg_deleted.append((regulations[index], regulation_types[index]))   
        del regulations[index]
        del regulation_types[index]
        i += 1
    
    new_model_spec.regulations = regulations
    new_model_spec.regulation_types = regulation_types
    
    # generate a new model with the pruned regulations
    new_model = new_model_spec.generate_network('sub_'+old_model.name, (0, 100), (0.1, 1), (0.1, 1), verbose=0, random_seed=random_seed)
    
    # copy the initial values from the old model to the new model    
    new_model = copy_over_params_states(old_model, new_model)
            
    return new_model_spec, new_model, reg_deleted
    
def copy_over_params_states(old_model: ModelBuilder, new_model: ModelBuilder):
    '''
    for each parameter and state in the old model, copy over the values to the new model if the parameter/state exists in the new model
    '''
    params, states = old_model.parameters.copy(), old_model.states.copy()
    for p in params:
        if p in new_model.parameters:
            new_model.parameters[p] = old_model.parameters[p]
    for s in states:
        if s in new_model.states:
            new_model.states[s] = old_model.states[s]

    return new_model

def systematic_specie_pruning(old_model_spec: ModelSpecification, old_model: ModelBuilder, specie_number: int): 
    '''
    Prune the model by removing `specie_number` number of species from the model_spec randomly. The remaining model will have at least one specie in A and B which are connected by a regulation, and one specie in C.
    '''
    pass
        
    