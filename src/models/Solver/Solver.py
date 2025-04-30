# abstract class solver 
# This class is used to solve the problem

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

class Solver(ABC):
    """
    Abstract class for an ODE solver that can run simulations 
    """
    
    def compile_from_sbml(self, sbml: str) -> None:
        pass 
    
    def compile_from_sbml_file(self, sbml_file: str) -> None:
        """
        Compile the SBML file to a solver.
        """
        pass

    @abstractmethod
    def simulate(self, start: float, stop: float, step: float):
        """
        Simulate the problem from start to stop with a given step size.
        """
        pass