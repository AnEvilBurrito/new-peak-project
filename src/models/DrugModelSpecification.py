from models.Utils import * 


class Drug: 
    
    def __init__(self, name, start_time, default_value, regulation=None, regulation_type=None):
        ''' 
        A drug class that represents a drug that can be applied to a model. 
        Input: 
            name: str | The name of the drug
            start_time: int | The time at which the drug is applied to the model.
            default_value: float | The default value of the drug in the model
            regulation: List[str] | The list of species that the drug regulates 
            regulation_type: List[str] | The type of regulation that the drug has on the species, either 'up' or 'down'
        '''
        self.name = name
        self.start_time = start_time
        self.default_value = default_value
        self.regulation = [] if regulation is None else regulation
        self.regulation_type = [] if regulation_type is None else regulation_type
        assert len(self.regulation) == len(self.regulation_type), "The regulation and regulation_type lists must be the same length"
        
    def __str__(self):
        return f"Drug({self.name}, {self.start_time}, {self.regulation}, {self.regulation_type})"
    
    def __repr__(self):
        return str(self)
    
    def add_regulation(self, specie, type): 
        ''' 
        Adds a regulation to the drug. 
        Input: 
            specie: str | The name of the specie to regulate
            type: str | The type of regulation, either 'up' or 'down'
        '''
        self.regulation.append(specie)
        self.regulation_type.append(type)
        
    def add_regulations(self, species, types):
        ''' 
        Adds multiple regulations to the drug. 
        Input: 
            species: List[str] | The list of species to regulate
            types: List[str] | The list of types of regulation, either 'up' or 'down'
        '''
        assert len(species) == len(types), "The species and types lists must be the same length"
        for i in range(len(species)): 
            self.add_regulation(species[i], types[i])
        
    def print_regulation(self): 
        ''' 
        Prints the regulation of the drug. 
        '''
        for i in range(len(self.regulation)): 
            print(f"{self.regulation[i]}: {self.regulation_type[i]}")
            
    

class DrugModelSpecification(ModelSpecification):
    
    def __init__(self):
        super().__init__()
        self.drug_list = []
        self.drug_values = {}
        self.D_species = []
        
        
        
    def add_drug(self, drug: Drug, value=None):
        ''' 
        Adds a drug to the model. 
        Input: 
            drug: Drug | The drug to add to the model
            value: float | if not None, the value of the drug to set in the model
        '''
        
        self.drug_list.append(drug)
        if value is not None: 
            self.drug_values[drug.name] = value
        else: 
            self.drug_values[drug.name] = drug.default_value
        ## update species and regulations based on drug information
        
        # update drug species
        self.D_species.append(drug.name)
        
        # update regulations based on species
        for i in range(len(drug.regulation)): 
            specie = drug.regulation[i]
            type = drug.regulation_type[i]
            if specie not in self.A_species and specie not in self.B_species and specie not in self.C_species: 
                raise ValueError(f"Drug model not compatible: Specie {specie} not found in the model")
            if type != 'up' and type != 'down': 
                raise ValueError(f"Drug model not compatible: Regulation type must be either 'up' or 'down'")
            
            reg = (drug.name, specie)
            self.regulations.append(reg)
            self.regulation_types.append(type)
        
    def generate_specifications(self, random_seed, NA, NR, verbose=1):
        return super().generate_specifications(random_seed, NA, NR, verbose)
    
    def generate_network(self, network_name, mean_range_species, rangeScale_params, rangeMultiplier_params, verbose=1, random_seed=None):
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
        model = super().generate_network(network_name, mean_range_species, rangeScale_params, rangeMultiplier_params, verbose, random_seed)
        for drug in self.drug_list:
            model.add_simple_piecewise(0, drug.start_time, self.drug_values[drug.name], drug.name)
        model.precompile()
        return model 