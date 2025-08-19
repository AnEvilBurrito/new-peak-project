import numpy as np
from ..Utils import *

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
