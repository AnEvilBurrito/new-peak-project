from models.Solver.Solver import Solver

from typing import Dict, Any, Tuple
import pandas as pd
from roadrunner import RoadRunner

class RoadrunnerSolver(Solver):
    
    def __init__(self):
        super().__init__()
        self.roadrunner_instance = None
        self.last_sim_result = None
        
    def compile(self, compile_str: str, **kwargs) -> RoadRunner:
        
        self.roadrunner_instance = RoadRunner(compile_str, **kwargs)
    
    def simulate(self, start: float, stop: float, step: float) -> pd.DataFrame:
        """
        Simulate the problem from start to stop with a given step size.
        Returns a pandas dataframe with the results with the following columns:
        - time: time points of the simulation
        - [species]: species names
        with each row corresponds to a time point and each column corresponds to a species.
        """
        
        # Check if the roadrunner instance is created
        if self.roadrunner_instance is None:
            raise ValueError("RoadRunner instance is not created. Please call compile() first.")
        
        runner = self.roadrunner_instance
        res = runner.simulate(start, stop, step)
        # Convert the result to a pandas dataframe, by default, this will not work 
        
        ## First step is to obtain all the state variables in the model
        state_vars = runner.model.getFloatingSpeciesIds()
        
        new_data = []
        new_data.append(res['time'])
        for state in state_vars:
            new_data.append(res[f'[{state}]'])
        
        # Convert the result to a pandas dataframe
        df = pd.DataFrame(new_data).T
        df.columns = ['time'] + list(state_vars)
        return df