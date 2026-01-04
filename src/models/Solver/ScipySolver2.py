from models.Solver.Solver import Solver

import re
from typing import List, Dict, Any, Tuple, Callable

import pandas as pd
from scipy.integrate import odeint
import sympy as sp
import numpy as np
from numba import njit

class ScipySolver2(Solver):
    """
    Enhanced Solver using scipy to solve ODEs with support for complex Antimony models.
    """
    
    def __init__(self):
        super().__init__()
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
        Compile the model from an Antimony string and generate ODE function.
        Supports complex assignment rules with time-dependent expressions.
        """
        result = self._parse_antimony_model(compile_str)
        reactions, species, parameters, y0, parameter_values, assignment_rules = result

        self.reactions = reactions
        self.species = species
        self.assignment_rules = assignment_rules
        self.assignment_rule_vars = sorted(assignment_rules.keys())
        self.parameters = parameters
        self.parameter_values = parameter_values  
        self.y0 = y0

        self.jit = kwargs.get("jit", False)
        if not self.jit:
            self.func = self._reactions_to_ode_func(
                reactions, species, self.parameters, self.assignment_rules
            )
        else:
            self.func = self._reactions_to_jit_ode_func(
                reactions, species, self.parameters, self.assignment_rules
            )
    
    def simulate(self, start: float, stop: float, steps: int) -> pd.DataFrame:
        """
        Simulate the ODE system from start to stop with given number of steps.
        """
        if self.func is None:
            raise ValueError("Model instance is not created. Please call compile() first.")

        t = np.linspace(start, stop, steps)
        
        try:
            sol = odeint(self.func, self.y0, t, args=(tuple(self.parameter_values),))
        except Exception as e:
            raise RuntimeError(f"ODE integration failed: {e}")

        df = pd.DataFrame(sol, columns=self.species)
        df.insert(0, "time", t)
        self.last_sim_result = df
        return df

    def set_state_values(self, state_values: Dict[str, float]) -> bool:
        """Set initial values of state variables."""
        if self.func is None:
            raise ValueError("Model instance is not created. Please call compile() first.")
        
        for key in state_values.keys():
            if key not in self.species:
                raise ValueError(f"State variable {key} is not valid. Valid: {self.species}")
        
        for key, value in state_values.items():
            index = self.species.index(key)
            self.y0[index] = value
        
        return True

    def set_parameter_values(self, parameter_values: Dict[str, float]) -> bool:
        """Set parameter values."""
        if self.func is None:
            raise ValueError("Model instance is not created. Please call compile() first.")
        
        for key in parameter_values.keys():
            if key not in self.parameters:
                raise ValueError(f"Parameter {key} is not valid. Valid: {self.parameters}")
        
        for key, value in parameter_values.items():
            index = self.parameters.index(key)
            self.parameter_values[index] = value
        
        return True
        
    def _parse_antimony_model(self, antimony_str: str) -> Tuple[
        List[str], List[str], List[str], List[float], List[float], Dict[str, str]
    ]:
        """
        Parse Antimony model string and extract all components.
        Returns: reactions, species, parameters, y0, parameter_values, assignment_rules
        """
        reactions = []
        species_set = set()
        species_dict = {}
        parameter_dict = {}
        assignment_rules = {}
        
        # Track species declarations
        declared_species = set()

        for line in antimony_str.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            # Skip model declaration lines
            if line.startswith("model ") or line == "end":
                continue
            
            # Skip compartment declarations
            if "compartment" in line and "is" in line:
                continue
            
            # Parse species declarations
            if line.startswith("substanceOnly species") or line.startswith("species"):
                match = re.match(r'(?:substanceOnly\s+)?species\s+(\w+)\s+in\s+\w+\s+is\s+"[^"]*"\s*=\s*([0-9.e+-]+)', line)
                if match:
                    species_name, init_val = match.groups()
                    declared_species.add(species_name)
                    species_dict[species_name] = float(init_val)
                continue
            
            # Parse parameter declarations (const keyword)
            if line.startswith("const "):
                match = re.match(r'const\s+(\w+)\s+is\s+"[^"]*"\s*=\s*([0-9.e+-]+)', line)
                if match:
                    param_name, value = match.groups()
                    parameter_dict[param_name] = float(value)
                continue
            
            # Parse non-const parameters
            if "is" in line and "=" in line and not line.startswith("const"):
                match = re.match(r'(\w+)\s+is\s+"[^"]*"\s*=\s*([0-9.e+-]+)', line)
                if match:
                    param_name, value = match.groups()
                    if param_name not in declared_species:
                        parameter_dict[param_name] = float(value)
                continue
            
            # Parse assignment rules
            if ":=" in line:
                parts = line.split(":=", 1)
                var = parts[0].strip()
                expr = parts[1].split(";")[0].strip()  # Remove comment
                assignment_rules[var] = expr
                # Assignment rule variables are treated as parameters
                if var not in parameter_dict:
                    parameter_dict[var] = 0.0  # Will be computed dynamically
                continue
            
            # Parse reactions
            if " is " in line and ":" in line and ("=>" in line or "->" in line):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    rxn = parts[1].strip()
                    reactions.append(rxn)
                    
                    # Extract species from reaction
                    reaction_part = rxn.split(";")[0].strip()
                    if "=>" in reaction_part or "->" in reaction_part:
                        sep = "=>" if "=>" in reaction_part else "->"
                        lhs, rhs = reaction_part.split(sep)
                        
                        # Parse reactants
                        reactants = [s.strip() for s in lhs.split("+") if s.strip() and s.strip() != ""]
                        # Parse products
                        products = [s.strip() for s in rhs.split("+") if s.strip() and s.strip() != ""]
                        
                        species_set.update(reactants + products)

        # Build final lists
        species = sorted([s for s in species_dict.keys() if s in declared_species])
        y0 = [species_dict[s] for s in species]
        
        parameters = sorted(parameter_dict.keys())
        parameter_values = [parameter_dict[p] for p in parameters]
        
        return reactions, species, parameters, y0, parameter_values, assignment_rules

    def _create_assignment_rule_functions(self, assignment_rules: Dict[str, str], 
                                         parameters: List[str]) -> Dict[str, Callable]:
        """
        Convert assignment rules to callable functions.
        Supports ifge0, pow, and other mathematical functions.
        """
        rule_functions = {}
        
        # Create symbol mapping
        param_syms = {p: sp.Symbol(p) for p in parameters}
        t_sym = sp.Symbol('time')
        all_syms = {**param_syms, 'time': t_sym}
        
        # Define custom functions
        def ifge0_impl(cond, val_true, val_false):
            """If greater or equal to 0: ifge0(x, a, b) = a if x >= 0 else b"""
            return sp.Piecewise((val_true, cond >= 0), (val_false, True))
        
        for var, expr_str in assignment_rules.items():
            try:
                # Replace ifge0 with Piecewise
                expr_str_processed = expr_str
                
                # Handle ifge0 function
                while 'ifge0' in expr_str_processed:
                    match = re.search(r'ifge0\(([^,]+),\s*([^,]+),\s*([^)]+)\)', expr_str_processed)
                    if match:
                        cond, val_true, val_false = match.groups()
                        # Create piecewise: if cond >= 0 then val_true else val_false
                        replacement = f"Piecewise(({val_true}, ({cond}) >= 0), ({val_false}, True))"
                        expr_str_processed = expr_str_processed[:match.start()] + replacement + expr_str_processed[match.end():]
                    else:
                        break
                
                # Parse with sympy
                expr = sp.sympify(expr_str_processed, locals=all_syms)
                
                # Create lambdified function
                # Args: time, then all parameters in order
                func = sp.lambdify([t_sym] + [param_syms[p] for p in parameters], 
                                  expr, modules=['numpy', {'Piecewise': sp.Piecewise}])
                rule_functions[var] = func
                
            except Exception as e:
                raise ValueError(f"Failed to parse assignment rule '{var} := {expr_str}': {e}")
        
        return rule_functions

    def _reactions_to_ode_func(self, reactions: List[str], species: List[str], 
                               parameters: List[str], assignment_rules: Dict[str, str]):
        """
        Convert reactions to ODE function with assignment rule support.
        """
        # Create symbols
        species_syms = {s: sp.Symbol(s) for s in species}
        param_syms = {p: sp.Symbol(p) for p in parameters}
        t_sym = sp.Symbol('time')
        
        # Combine all symbols for expression parsing
        all_syms = {**species_syms, **param_syms, 'time': t_sym}
        
        # Add pow as a recognized function
        all_syms['pow'] = sp.Pow
        
        # Create assignment rule functions
        rule_funcs = self._create_assignment_rule_functions(assignment_rules, parameters)
        
        # Build derivative expressions
        derivs = {s: 0 for s in species}

        for rxn in reactions:
            try:
                reaction_part, rate_expr = map(str.strip, rxn.split(";"))
                
                # Parse rate expression
                rate = sp.sympify(rate_expr, locals=all_syms)
                
                # Handle reversible reactions
                if "<=>" in reaction_part:
                    raise NotImplementedError("Reversible reactions not yet supported.")
                
                # Parse reactants and products
                sep = "=>" if "=>" in reaction_part else "->"
                reactants_str, products_str = map(str.strip, reaction_part.split(sep))
                
                reactants = [r.strip() for r in reactants_str.split("+") if r.strip()]
                products = [p.strip() for p in products_str.split("+") if p.strip()]
                
                # Update derivatives
                for r in reactants:
                    if r in species:
                        derivs[r] -= rate
                for p in products:
                    if p in species:
                        derivs[p] += rate
                        
            except Exception as e:
                raise ValueError(f"Failed to parse reaction '{rxn}': {e}")

        # Create derivative expressions list
        dydt_exprs = [derivs[s] for s in species]
        
        # Lambdify with all symbols
        all_inputs = [t_sym] + list(species_syms.values()) + list(param_syms.values())
        dydt_func = sp.lambdify(all_inputs, dydt_exprs, modules=['numpy', 'scipy'])

        def func(y, t, params):
            """ODE function with assignment rule evaluation."""
            # Evaluate assignment rules at current time
            params_with_rules = list(params)
            for i, p in enumerate(parameters):
                if p in rule_funcs:
                    # Evaluate rule: func(time, *all_params)
                    params_with_rules[i] = rule_funcs[p](t, *params)
            
            # Evaluate derivatives: dydt_func(t, *y, *params_with_rules)
            result = dydt_func(t, *y, *params_with_rules)
            return np.array(result, dtype=float).flatten()

        return func

    def _reactions_to_jit_ode_func(self, reactions: List[str], species: List[str], 
                                   parameters: List[str], assignment_rules: Dict[str, str]):
        """
        Create Numba-compiled ODE function (simplified, may not support all features).
        """
        # For complex models with assignment rules, JIT compilation is challenging
        # This is a simplified version - for full support, use non-JIT mode
        raise NotImplementedError(
            "JIT compilation not fully supported for complex assignment rules. "
            "Please use jit=False in compile()."
        )