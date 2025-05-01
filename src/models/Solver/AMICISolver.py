from models.Solver.Solver import Solver

import amici # type: ignore 
import logging
import pandas as pd
import numpy as np

class AMICISolver(Solver):
    
    def __init__(self):
        super().__init__()
        self.model_module = None
        self.last_sim_result = None
        
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
        
    def simulate(self, start: float, stop: float, step: float) -> pd.DataFrame:

        if self.model_module is None:
            raise ValueError("Model instance is not created. Please call compile() first.")
        
        # Create Model instance
        model = self.model_module.getModel()
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
        return df
