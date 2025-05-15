from models.Solver.Solver import Solver

import re
from typing import List, Dict, Any, Tuple

import pandas as pd
from scipy.integrate import odeint
import sympy as sp
import numpy as np
from numba import njit

class ScipySolver(Solver):
    """
    Solver using scipy to solve the ODEs. 
    """
    
    def __init__(self):
        super().__init__()
        # reactions, species, parameters, y0, parameter_values
        self.reactions = None
        self.species = None
        self.parameters = None
        self.y0 = None
        self.parameter_values = None
        self.assignment_rules = None
        self.func = None
        self.jit = False
        
        self.last_sim_result = None
        
    def compile(self, compile_str: str, **kwargs) -> None:
        """
        Takes an antimony string and use existing functions to set up the scipy ODE solver.
        """
        # Parse the antimony string to extract reactions, species, parameters, y0, and parameter_values
        reactions, species, parameters, y0, parameter_values = self._parse_antimony_model(compile_str)
        self.reactions = reactions
        self.species = species
        self.parameters = parameters
        self.y0 = y0
        self.parameter_values = parameter_values
        
        # depending on the keyword argument 'jit' being True, we will use the jit version to compile the function 
        # if keyword argument 'jit' is not provided, we will use the default version
        jit = kwargs.get('jit', False)
        if not jit:
            self.func = self._reactions_to_ode_func(self.reactions, self.species, self.parameters)
        else:
            # if jit is True, we will use the jit version to compile the function
            self.func = self._reactions_to_jit_ode_func(self.reactions, self.species, self.parameters)
            self.jit = True
        
        
    def simulate(self, start: float, stop: float, step: float) -> pd.DataFrame:
        """
        Simulate the problem from start to stop with a given step size.
        Returns a pandas dataframe with the results with the following columns:
        - time: time points of the simulation
        - [species]: species names
        with each row corresponds to a time point and each column corresponds to a species.
        """
        
        # Check if the model is created
        if self.func is None:
            raise ValueError("Model instance is not created. Please call compile() first.")
        
        t = np.linspace(start, stop, step)
        if self.jit:
            def ode_wrapper(y, t, *args):
                params = np.array(args)
                return self.func(y, t, params)
            
            sol = odeint(ode_wrapper, self.y0, t, args=tuple(self.parameter_values))
        else: 
            sol = odeint(self.func, self.y0, t, args=tuple(self.parameter_values))

        # set up the result dataframe
        result = pd.DataFrame(sol, columns=self.species)
        result.insert(0, 'time', t)
        
        self.last_sim_result = result
        return result

    def set_state_values(self, state_values: Dict[str, float]) -> bool:
        """
        Hot swapping of state variables in the running instance of the model, note this is setting the initial values of the state variables.
        Set the values of state variables in the model instance, this should only possible after compiling the model. 
        Not every solver will support this, so it is possible that this function to return an not implemented error.
        returns True if the state variable was set successfully, False otherwise.
        """
        # Check if the model is created
        if self.func is None:
            raise ValueError("Model instance is not created. Please call compile() first.")
        
        # Check if the state values are valid
        for key in state_values.keys():
            if key not in self.species:
                raise ValueError(f"State variable {key} is not valid. Valid state variables are: {self.species}")
        
        # Set the state values
        for key, value in state_values.items():
            index = self.species.index(key)
            self.y0[index] = value
        
        return True

    def set_parameter_values(self, parameter_values: Dict[str, float]) -> bool:
        """
        Hot swapping of parameters in the running instance of the model.
        Set the values of parameter variables in the model instance, this should only possible after compiling the model. 
        Not every solver will support this, so it is possible that this function to return an not implemented error.
        returns True if the state variable was set successfully, False otherwise.
        """
        # Check if the model is created
        if self.func is None:
            raise ValueError("Model instance is not created. Please call compile() first.")
        
        # Check if the parameter values are valid
        for key in parameter_values.keys():
            if key not in self.parameters:
                raise ValueError(f"Parameter {key} is not valid. Valid parameters are: {self.parameters}")
        
        # Set the parameter values
        for key, value in parameter_values.items():
            index = self.parameters.index(key)
            self.parameter_values[index] = value
        
        return True
        
        
    def _reactions_to_ode_func(self, reactions, species, parameters):
        """
        Convert a list of reactions to an ODE function using sympy.
        reactions: list of strings, each string is a reaction in the format "A + B -> C + D; rate_expr"
        species: list of strings, each string is a species name
        parameters: list of strings, each string is a parameter name
        """
        # Create symbols for species and parameters
        species_syms = {s: sp.Symbol(s) for s in species}
        param_syms = {p: sp.Symbol(p) for p in parameters}
        
        # Initialize derivative expressions
        derivs = {s: 0 for s in species}

        for rxn in reactions:
            # Format: "A + B -> C + D; rate_expr"
            reaction_part, rate_expr = map(str.strip, rxn.split(";"))
            rate = sp.sympify(rate_expr, locals={**species_syms, **param_syms})
            
            if "<->" in reaction_part:
                raise NotImplementedError("Reversible reactions not yet supported.")
            reactants_str, products_str = map(str.strip, reaction_part.split("->"))
            reactants = [r.strip() for r in reactants_str.split("+") if r.strip()]
            products = [p.strip() for p in products_str.split("+") if p.strip()]
            
            # Update derivatives
            for r in reactants:
                derivs[r] -= rate
            for p in products:
                derivs[p] += rate

        # Convert expressions to list in species order
        dydt_exprs = [derivs[s] for s in species]
        
        # Lambdify
        dydt_func = sp.lambdify((list(species_syms.values()), list(param_syms.values())), dydt_exprs, modules="numpy")

        # Define final ODE function
        def func(y, t, *params):
            return np.array(dydt_func(y, params)).flatten()

        return func
    
    def _parse_antimony_model(self, antimony_str):
        reactions = []
        species_set = set()
        species_dict = {}
        parameter_dict = {}

        for line in antimony_str.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("model ") or line == "end":
                continue

            # Reaction line
            if ":" in line and ";" in line:
                _, rxn = line.split(":", 1)
                rxn = rxn.strip()
                reactions.append(rxn)

                # Capture species from reaction components
                reaction_part, _ = map(str.strip, rxn.split(";"))
                if "->" in reaction_part:
                    lhs, rhs = map(str.strip, reaction_part.split("->"))
                    reactants = [s.strip() for s in lhs.split("+")]
                    products = [s.strip() for s in rhs.split("+")]
                    species_set.update(reactants + products)

            # Variable assignment
            elif "=" in line:
                var, val = map(str.strip, line.split("=", 1))
                try:
                    val = float(val)
                except ValueError:
                    continue

                if var in species_set:
                    species_dict[var] = val
                else:
                    parameter_dict[var] = val

        species = sorted(species_dict.keys())
        parameters = sorted(parameter_dict.keys())
        y0 = [species_dict[s] for s in species]
        parameter_values = [parameter_dict[p] for p in parameters]

        return reactions, species, parameters, y0, parameter_values
    
    def _reactions_to_jit_ode_func(self, reactions, species, parameters):
        # Create symbolic variables
        species_syms = {s: sp.Symbol(s) for s in species}
        param_syms = {p: sp.Symbol(p) for p in parameters}

        derivs = {s: 0 for s in species}

        for rxn in reactions:
            reaction_part, rate_expr = map(str.strip, rxn.split(";"))
            rate = sp.sympify(rate_expr, locals={**species_syms, **param_syms})

            if "<->" in reaction_part:
                raise NotImplementedError("Reversible reactions, in antimony '<->', not yet supported.")

            reactants_str, products_str = map(str.strip, reaction_part.split("->"))
            reactants = [r.strip() for r in reactants_str.split("+") if r.strip()]
            products = [p.strip() for p in products_str.split("+") if p.strip()]

            for r in reactants:
                derivs[r] -= rate
            for p in products:
                derivs[p] += rate

        dydt_exprs = [derivs[s] for s in species]

        # Substitutions
        species_subs = {species_syms[s]: sp.Symbol(f"y[{i}]") for i, s in enumerate(species)}
        param_subs = {param_syms[p]: sp.Symbol(f"params[{i}]") for i, p in enumerate(parameters)}

        func_lines = ["def generated_func(y, t, params):"]
        func_lines.append(f"    dydt = np.empty({len(dydt_exprs)})")

        for i, expr in enumerate(dydt_exprs):
            substituted = expr.subs({**species_subs, **param_subs})
            code_line = sp.ccode(substituted)
            func_lines.append(f"    dydt[{i}] = {code_line}")

        func_lines.append("    return dydt")

        func_code = "\n".join(func_lines)

        local_vars = {"np": np}
        exec(func_code, local_vars)
        generated_func = local_vars["generated_func"]

        # Compile with Numba
        return njit(generated_func)