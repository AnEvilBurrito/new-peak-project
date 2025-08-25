import numpy as np
import logging
from ..ArchtypeCollections import create_archtype_michaelis_menten, michaelis_menten
from ..ReactionArchtype import ReactionArchtype
from ..ModelBuilder import ModelBuilder
from ..Reaction import Reaction
from .Regulation import Regulation
from .Drug import Drug

class ModelSpec2:
    def __init__(self, num_intermediate_layers=2):
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
        self.ordinary_regulations = []
        self.feedback_regulations = []
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
            
    def add_regulation(self, from_specie, to_specie, reg_type):
        '''
        Adds a regulation to the model.
        Input:
            from_specie: str | The specie that regulates
            to_specie: str | The specie that is regulated
            reg_type: str | The type of regulation, either 'up' or 'down'
        '''
        if reg_type not in ['up', 'down']:
            raise ValueError("Regulation type must be either 'up' or 'down'")
        
        reg = Regulation(from_specie=from_specie, to_specie=to_specie, reg_type=reg_type)
        self.regulations.append(reg)

    def generate_specifications(self, num_cascades, num_regulations, random_seed=None, verbose=1):
        '''
        Extend the generate_specifications method to include the D -> R -> Intermediate -> O architecture.
        Returns a list of all species in the model, including drugs, receptors, intermediate layers, and outcomes.
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
        else: 
            rng = np.random.default_rng()
            
        logger.debug('--- Generating a random network ---')
        logger.debug(f'Random Seed: {random_seed_number}')
        logger.debug(f'Number of cascades: {num_cascades}')
        logger.debug(f'Number of Regulations: {num_regulations}')
        logger.debug('\n')
  
        # based on the number of cascades, generate the species, index from 1
        self.receptors = [f'R{i+1}' for i in range(num_cascades)]
        self.intermediate_layers = [[f'I{i+1}_{j+1}' for j in range(num_cascades)] for i in range(len(self.intermediate_layers))]
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
            
        logger.debug(f'Generated Ordinary Regulations:')
        for reg in self.regulations:
            logger.debug(f'From {reg.from_specie} to {reg.to_specie} of type {reg.reg_type}')
            
        # finally, based on num_regulations, generate regulations between different species, but outcome 
        # should not be regulating other species
        all_species = self.get_all_species(include_drugs=False, include_outcomes=False)
        # at this point, all regulations should be only the ordinary regulations
        all_regulations = [(reg.from_specie, reg.to_specie) for reg in self.regulations]
        self.ordinary_regulations = self.regulations.copy()
        feedback_regulations = []
        
        def check_species(species1, species2, feedback_regulations):
            '''
            Check if specie1 regulates specie2 in the current regulations.
            '''
            for reg in self.regulations:
                if (reg.from_specie == species1 and reg.to_specie == species2):
                    return True
            for reg in feedback_regulations:
                if (reg.from_specie == species1 and reg.to_specie == species2):
                    return True
            return False
        
        for _ in range(num_regulations):
            # randomly select two species to regulate each other
            species1 = rng.choice(all_species)
            species2 = rng.choice([s for s in all_species if s != species1])
            reg_type = rng.choice(['up', 'down'])
            total_species = len(all_species)
            max_iterations = total_species * 10  # arbitrary large number to avoid infinite loop
            valid = False
            for _ in range(max_iterations):
                if species1 != species2 and not check_species(species1, species2, feedback_regulations):
                    valid = True
                    break
                # Reselect maintaining biological constraints [1]
                species1 = rng.choice(all_species)
                species2 = rng.choice([s for s in all_species if s != species1])
                reg_type = rng.choice(['up', 'down'])
            if not valid:
                raise ValueError("Failed to find valid species pair")

            reg = Regulation(species1, species2, reg_type)
            feedback_regulations.append(reg)
            
        # add the feedback regulations to the model
        self.feedback_regulations = feedback_regulations
        self.regulations.extend(feedback_regulations)
        logger.debug(f'Generated Feedback Regulations:')
        for reg in feedback_regulations:
            logger.debug(f'From {reg.from_specie} to {reg.to_specie} of type {reg.reg_type}')
        
    ### Helpers 
    def generate_archtype_and_regulators(self, specie):

        all_regulations = [(reg.from_specie, reg.to_specie) for reg in self.regulations]
        all_regulation_types = [reg.reg_type for reg in self.regulations]            
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

        # print(f'Sorted regulators information: {regulators_for_specie}')
        # print(f'Final regulators for {specie}: {regulators_sorted_phos}')
        # print(f'Rate law for {specie}: {rate_law}')
        return rate_law, regulators_sorted                                            

    # generate random parameters informed by a scale
    def generate_random_parameters(self, reaction_archtype: ReactionArchtype, scale_range, multiplier_range, random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()

        assumed_values = reaction_archtype.assume_parameters_values
        # print(f'Assumed values: {assumed_values}')
        r_params = []
        for _, value in assumed_values.items():
            rand = rng.uniform(value*scale_range[0], value*scale_range[1])
            rand *= rng.uniform(multiplier_range[0], multiplier_range[1])
            r_params.append(rand)

        return tuple(r_params)

    def generate_network(self, network_name, mean_range_species, rangeScale_params, rangeMultiplier_params, random_seed=None) -> ModelBuilder:
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
        else:
            rng = np.random.default_rng()
        
        
        def add_reactions(specie, model: ModelBuilder):
            '''
            Returns the forward and reverse reactions for a given specie.
            '''
            forward_reaction = self.get_forward_reaction(specie, mean_range_species, rangeScale_params, rangeMultiplier_params, rng)
            reverse_reaction = self.get_reverse_reaction(specie, rangeScale_params, rangeMultiplier_params, rng)
            # add reverse reaction first, then forward reaction to ensure specie values are set correctly
            model.add_reaction(reverse_reaction)
            model.add_reaction(forward_reaction)
        
        # generate reaction classes layer by layer, starting from receptors
        # reactions are in the form of R1->R1a (which is an activation of R1)
        for receptor in self.receptors:
            add_reactions(receptor, model)
        
        # generate reactions for intermediate layers
        for layer in self.intermediate_layers:
            for specie in layer:
                add_reactions(specie, model)
                
        # generate reactions for outcomes
        for outcome in self.outcomes:
            add_reactions(outcome, model)
        for drug in self.drugs:
            model.add_simple_piecewise(0, drug.start_time, self.drug_values[drug.name], drug.name)
        model.precompile()
        logger = logging.getLogger(__name__)
        logger.info(f"Generated model {network_name} with {len(model.reactions)} reactions.")
        logger.info(f"Model States: {len(model.states)}")
        logger.info(f"Model Parameters: {len(model.parameters)}")
        logger.info(f"Model Reactions: {len(model.reactions)}")
        logger.debug('\n--- Antimony Model ---\n')
        logger.debug(model.get_antimony_model())
        return model

    def get_forward_reaction(self, specie, mean_range_species, rangeScale_params, rangeMultiplier_params, rng):
        forward_rate_law, regulators = self.generate_archtype_and_regulators(specie)
        forward_params = self.generate_random_parameters(forward_rate_law, rangeScale_params, rangeMultiplier_params, rng)
        forward_state_val = rng.integers(mean_range_species[0], mean_range_species[1])
        forward_reaction = Reaction(forward_rate_law, 
                                        (specie,), (specie + 'a',), 
                                        reactant_values=forward_state_val,
                                        extra_states=regulators,
                                        parameters_values=tuple(forward_params), 
                                        zero_init=False)
        return forward_reaction

    def get_reverse_reaction(self, specie, rangeScale_params, rangeMultiplier_params, rng):
        reverse_params = self.generate_random_parameters(michaelis_menten, rangeScale_params, rangeMultiplier_params, rng)
        reverse_reaction = Reaction(michaelis_menten, 
                                         (specie + 'a',), (specie,), 
                                         parameters_values=tuple(reverse_params), 
                                         zero_init=False)
        return reverse_reaction
    
    def get_regulations(self):
        '''
        extracts the regulations and their types from the model, as lists of tuples
        '''
        return self.regulations.copy()
    
    def get_ordinary_regulations(self):
        '''
        extracts the ordinary regulations and their types from the model, as lists of tuples
        '''
        return self.ordinary_regulations.copy()
    
    def get_feedback_regulations(self):
        '''
        extracts the feedback regulations and their types from the model, as lists of tuples
        '''
        return self.feedback_regulations.copy()
