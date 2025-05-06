from models.Solver.Solver import Solver

import amici # type: ignore 
import logging
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

class AMICISolver(Solver):
    
    def __init__(self):
        super().__init__()
        self.model_module = None
        self.model = None   
        self.last_sim_result = None
        
        # for reset to origin, we need to set the model to the initial parameter values 
        self.initial_parameters = None
        self.initial_states = None
        
    def compile(self, compile_str: str, **kwargs) -> None:
        """
        AMICI Solver will only take in a filepath to a sbml file,
        """
        # Parse the antimony string to extract reactions, species, parameters, y0, and parameter_values
        model_dir = kwargs.get('model_dir', None)
        model_name = kwargs.get('model_name', None)
        if model_name is None:
            raise ValueError("Model name is not provided. Please provide a valid model name.")
        if model_dir is None:
            raise ValueError("Model directory is not provided. Please provide a valid model directory.")
        
        verbosity = kwargs.get('verbosity', logging.INFO)
        sbml_importer = amici.SbmlImporter(compile_str)
        sbml_importer.sbml2amici(model_name, model_dir, verbose=verbosity, generate_sensitivity_code=False, constant_parameters=False, simplify=None)
        self.model_module = amici.import_model_module(model_name, model_dir)
        
        # Initialise the parameter values in the model instance
        model = self.model_module.getModel()
        self.model = model
        
    def simulate(self, start: float, stop: float, step: float) -> pd.DataFrame:

        if self.model_module is None:
            raise ValueError("Model instance is not created. Please call compile() first.")
        
        # Create Model instance
        model = self.model
        # set timepoints for which we want to simulate the model
        model.setTimepoints(np.linspace(start, stop, step))
        # Create solver instance
        solver = model.getSolver()
        # Run simulation using default model parameters and solver options
        rdata = amici.runAmiciSimulation(model, solver)

        # Begin organising the data into a pandas dataframe
        time_points = rdata.ts
        res = rdata.x.T
        res_states = model.getStateNames()
        
        new_data = []
        new_data.append(time_points)
        for i, state in enumerate(res_states):
            new_data.append(res[i, :])
        # Convert the result to a pandas dataframe
        df = pd.DataFrame(new_data).T
        df.columns = ['time'] + list(res_states)
        
        # recreate the model as a reset to origin
        self.model = self.model_module.getModel()
        return df
        

    def set_state_values(self, state_values: Dict[str, float]) -> bool:
        """
        Hot swapping of state variables in the running instance of the model.
        Set the values of state variables in the model instance, this should only possible after compiling the model. 
        Not every solver will support this, so it is possible that this function to return an not implemented error.
        returns True if the state variable was set successfully, False otherwise
        """
        # Check if the model module is created
        if self.model is None:
            raise ValueError("Model instance is not created. Please call compile() first.")
        
        # Set the parameter values in the model instance
        model = self.model
        inits = list(model.getInitialStates())
        state_ids = model.getStateIds()
        for state, value in state_values.items():
            if state in state_ids:
                # search for the index of the state variable in the list of state variables
                index = state_ids.index(state)
                # change the initial state variable to the new value
                inits[index] = value
            else:
                raise ValueError(f"State variable {state} not found in the model.")
        # set the new initial states in the model instance
        model.setInitialStates(inits)
        return True

    def set_parameter_values(self, parameter_values: Dict[str, float]) -> bool:
        """
        Hot swapping of parameters in the running instance of the model.
        Set the values of parameter variables in the model instance, this should only possible after compiling the model. 
        Not every solver will support this, so it is possible that this function to return an not implemented error.
        returns True if the state variable was set successfully, False otherwise.
        """
        # Check if the model module is created
        if self.model is None:
            raise ValueError("Model instance is not created. Please call compile() first.")
        
        # Set the parameter values in the model instance
        model = self.model
        for param, value in parameter_values.items():
            if param in model.getParameterIds():
                model.setParameterByName(param, value)
            else:
                raise ValueError(f"Parameter {param} not found in the model.")
        return True
