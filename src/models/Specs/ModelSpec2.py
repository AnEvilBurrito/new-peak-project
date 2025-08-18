import numpy as np
import logging
from ..ArchtypeCollections import create_archtype_michaelis_menten, michaelis_menten
from ..ReactionArchtype import ReactionArchtype
from ..ModelBuilder import ModelBuilder
from ..Reaction import Reaction
from .Regulation import Regulation
from .Drug import Drug

class ModelSpec2:
    def __init__(self, num_intermediate_layers=3):
        '''
        Initialize the ModelSpecification class with attributes for Drugs, Receptors, Intermediate Layers, and Outcomes.
        ModelSpec2 will automatically handle drug interactions 
        '''
        # Initialize drugs, receptors, and outcomes
        self.drugs = []
        self.receptors = []
        self.outcomes = []

        # Dynamically generate intermediate layers
        self.intermediate_layers = [[] for _ in range(num_intermediate_layers)]
        
        self.regulations = []
        self.randomise_parameters = True

        self.drugs = []
        self.drug_values = {}
    
    def get_all_species(self, include_drugs=True, include_receptors=True, include_outcomes=True):
        '''
        Returns a list of all species in the model, including drugs, receptors, intermediate layers, and outcomes.
        '''
        all_species = []
        if include_drugs:
            all_species.extend(self.drugs)
        if include_receptors:
            all_species.extend(self.receptors)
        for layer in self.intermediate_layers:
            all_species.extend(layer)
        if include_outcomes:
            all_species.extend(self.outcomes)
        return all_species
    
    def __str__(self):
        pass 
    
    def add_drug(self, drug: Drug, value=None):
        ''' 
        Adds a drug to the model. 
        Input: 
            drug: Drug | The drug to add to the model
            value: float | if not None, the value of the drug to set in the model
        '''
        
        # drug name is added to species list 
        self.drugs.append(drug)
        if value is not None: 
            self.drug_values[drug.name] = value
        else: 
            self.drug_values[drug.name] = drug.default_value
        
        # update regulations based on species
        for i in range(len(drug.regulation)): 
            specie = drug.regulation[i]
            type = drug.regulation_type[i]
            all_species = self.get_all_species(include_drugs=False, include_outcomes=False)
            if specie not in all_species: 
                raise ValueError(f"Drug model not compatible: Specie {specie} not found in the model")
            if type != 'up' and type != 'down': 
                raise ValueError(f"Drug model not compatible: Regulation type must be either 'up' or 'down'")
            
            reg = Regulation(from_specie=drug.name, to_specie=specie, reg_type=type)
            self.regulations.append(reg)
            

    def generate_specifications(self, num_cascades, num_regulations, random_seed=None, verbose=1):
        '''
        Extend the generate_specifications method to include the D -> R -> Intermediate -> O architecture.
        '''
        
        logger = logging.getLogger(__name__)
        logger.debug('--- Generating D -> R -> Intermediate -> O architecture ---')
        logger.debug(f'Drugs: {self.drugs}')
        logger.debug(f'Receptors: {self.receptors}')
        logger.debug(f'Intermediate Layers: {self.intermediate_layers}')
        logger.debug(f'Outcomes: {self.outcomes}')
            
        # systematically generate C specie regulations insead of defining varialbles for each type of regulation
        random_seed_number = random_seed  # None if do not want to fix the seed
        if random_seed_number is not None:
            rng = np.random.default_rng(random_seed_number)
            
        logger.debug('--- Generating a random network ---')
        logger.debug(f'Random Seed: {random_seed_number}')
        logger.debug(f'Number of cascades: {num_cascades}')
        logger.debug(f'Number of Regulations: {num_regulations}')
        logger.debug('\n')
  
        # based on the number of cascades, generate the species, index from 1
        self.receptors = [f'R{i+1}' for i in range(num_cascades)]
        self.intermediate_layers = [[f'I{i+1}_{j+1}' for j in range(num_cascades)] for i in range(len(self.receptors))]
        # there should only be one outcome in this spec 
        self.outcomes = ['O']
        

        # first generate default regulations between receptors, intermediate layers, and outcomes using
        # the Regulation class
        for i in range(len(self.receptors)):
            # Recaptor i will regulate I1_i 
            reg = Regulation(from_specie=self.receptors[i], to_specie=self.intermediate_layers[0][i], reg_type='up')
            self.regulations.append(reg)
        # based on the number of intermediate layers, generate the regulations between them
        for i in range(len(self.intermediate_layers) - 1):
            for j in range(len(self.intermediate_layers[i])):
                # Intermediate layer i will regulate Intermediate layer i+1
                reg = Regulation(from_specie=self.intermediate_layers[i][j], to_specie=self.intermediate_layers[i+1][j], reg_type='up')
                self.regulations.append(reg)
        # finally, the last intermediate layer will regulate the outcome
        for i in range(len(self.intermediate_layers[-1])):
            reg = Regulation(from_specie=self.intermediate_layers[-1][i], to_specie=self.outcomes[0], reg_type='up')
            self.regulations.append(reg)
            
        # finally, based on num_regulations, generate regulations between different species, but outcome 
        # should not be regulating other species
        all_species = self.get_all_species(include_drugs=False, include_outcomes=False)
        for _ in range(num_regulations):
            # randomly select two species to regulate each other
            species1 = rng.choice(all_species)
            species2 = rng.choice(all_species)
            reg_type = rng.choice(['up', 'down'])
            reg = Regulation(from_specie=species1, to_specie=species2, reg_type=reg_type)
            total_species = len(all_species)
            max_iterations = total_species * 10  # arbitrary large number to avoid infinite loop
            num_iterations = 0
            # while loop to ensure that the two species are not the same and the regulation doesn't already exist
            while (species1 == species2 or reg in self.regulations) and num_iterations < max_iterations:
                if species1 == species2:
                    # randomly select a new species if they are the same
                    species2 = rng.choice(all_species)
                else:
                    # if they are different, check if the regulation already exists
                    if reg in self.regulations:
                        # if it exists, randomly select a new species
                        species1 = rng.choice(all_species)
                        species2 = rng.choice(all_species)
                        reg = Regulation(from_specie=species1, to_specie=species2, reg_type=reg_type)
                num_iterations += 1
            # if we reach here, we have found two different species to regulate each other
            if num_iterations >= max_iterations:
                # throw an error if we cannot find two different species
                raise ValueError("Could not find two different species to regulate each other. Max iterations reached.", max_iterations)
            
            self.regulations.append(reg)
        logger.debug(f'Generated Regulations: {self.regulations}')
        
    ### Helpers 
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
        
        # generate reaction classes layer by layer, starting from receptors
        # reactions are in the form of R1->R1a (which is an activation of R1)
        for receptor in self.receptors:
            # forward reaction
            rate_law, regulators = self.generate_archtype_and_regulators(receptor)
            
        
        
        model.precompile()
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
