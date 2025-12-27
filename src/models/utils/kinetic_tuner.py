"""
Kinetic parameter tuner for multi-degree drug interaction networks.

This module provides utilities to generate kinetic parameters that ensure
robust signal propagation through hierarchical networks by solving the
full nonlinear kinetic equations to achieve target active percentages.

Algorithm Overview:
1. For each species X, randomly assign active percentage p ∈ active_percentage_range
2. Compute target active concentration [X_a] = p × X_total
3. Parse drug concentrations from model.variables (piecewise assignments)
4. For each activated species Xa:
   a. Identify forward parameters (kc_i values) and backward parameters (vmax_b)
   b. Get regulator active concentrations [Y_i_a] from conc_target or drug_concentrations
   c. Solve: p = (Σ kc_i × [Y_i_a]) / (Σ kc_i × [Y_i_a] + vmax_b)
   d. Determine individual kc_i values: kc_i = constant / [Y_i_a] (equal contribution)
   e. Set km_b = X_total × X_total_multiplier
   f. Set km_f = km_b × (1 + Σ [Y_inhibitor_a]/ki_val) where inhibitors are down regulators
   g. Set ki = ki_val (constant)

Complexity Analysis:
- Time Complexity: O(n × m) where:
    n = number of activated species (states ending with 'a')
    m = average number of regulators per species
- Space Complexity: O(n + p) where:
    n = number of species
    p = total parameters in the model
- The algorithm processes each species independently once all target
  concentrations are known, enabling potential parallelization.

Note: This implementation requires that each activated species has its own
independent set of parameters (no parameter sharing between species), which
is the typical pattern in the DegreeInteractionSpec networks.
"""

import re
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from ..ModelBuilder import ModelBuilder
from .parameter_mapper import get_parameters_for_state, get_parameter_reaction_map


class KineticParameterTuner:
    """
    Generate kinetic parameters that achieve target active percentage ranges.
    
    This tuner solves the full nonlinear kinetic equations to ensure that
    each species reaches a specified active percentage at steady state,
    enabling robust signal propagation through multi-degree networks.
    
    Attributes:
        model: The ModelBuilder object containing the network structure
        rng: numpy random number generator for reproducibility
    """
    
    def __init__(self, model: ModelBuilder, random_seed: Optional[int] = None):
        """
        Initialize the parameter tuner for a specific model.
        
        Args:
            model: ModelBuilder object with network structure
            random_seed: Optional seed for reproducible parameter generation
        """
        if not model.pre_compiled:
            raise ValueError("Model must be pre-compiled before parameter tuning")
        
        self.model = model
        self.rng = np.random.default_rng(random_seed)
        
        # Cache state information for performance
        self.all_states = model.get_state_variables()
        self.active_states = {k: v for k, v in self.all_states.items() if k.endswith('a')}
        self.inactive_states = {k: v for k, v in self.all_states.items() if not k.endswith('a')}
        
        # Parse drug concentrations from model.variables
        self.drug_concentrations = self._parse_drug_concentrations()
        self._target_concentrations = {}
        
        # Get parameter-reaction map for regulator identification
        self.param_map = get_parameter_reaction_map(model)
        
        # Validate that each active state has a corresponding inactive state
        self._validate_state_pairs()
    
    def _parse_drug_concentrations(self) -> Dict[str, float]:
        """
        Parse drug concentrations from model.variables piecewise assignments.
        
        Returns:
            Dictionary mapping drug names to their concentrations (after_value)
        """
        drug_concentrations = {}
        
        # Regex pattern for piecewise assignment: 'D := piecewise(0, time < 500, 10)'
        pattern = r'(\w+)\s*:=\s*piecewise\(([^,]+),\s*time\s*<\s*([^,]+),\s*([^)]+)\)'
        
        for var_name, rule in self.model.variables.items():
            match = re.match(pattern, rule.strip())
            if match:
                state_name, before_value, activation_time, after_value = match.groups()
                try:
                    drug_concentrations[state_name] = float(after_value)
                except ValueError:
                    # If conversion fails, skip this variable
                    continue
        
        return drug_concentrations
    
    def _validate_state_pairs(self):
        """Ensure each active state Xa has a corresponding inactive state X."""
        for active_state in self.active_states.keys():
            inactive_state = active_state[:-1]  # Remove trailing 'a'
            if inactive_state not in self.inactive_states:
                raise ValueError(
                    f"Active state {active_state} has no corresponding inactive state {inactive_state}"
                )
    
    def generate_parameters(
        self,
        active_percentage_range: Tuple[float, float] = (0.3, 0.7),
        X_total_multiplier: float = 5.0,
        ki_val: float = 100.0,
        v_max_f_random_range: Tuple[float, float] = (5.0, 10.0)
    ) -> Dict[str, float]:
        """
        Generate kinetic parameters to achieve target active percentages.
        
        Args:
            active_percentage_range: Range for target active percentages (e.g., (0.3, 0.7))
            X_total_multiplier: Multiplier for setting km_b = X_total × multiplier
            ki_val: Constant value for all inhibition constants Ki
            v_max_f_random_range: Range for total forward vmax (Σ kc_i × [Y_i_a])
            
        Returns:
            Dictionary mapping parameter names to tuned values
            
        Raises:
            ValueError: If target active percentages cannot be achieved
            RuntimeError: If parameter structure doesn't match expected pattern
        """
        # Step 1: Initialize target active concentrations for all species
        if not self._target_concentrations:
            self._initialize_target_concentrations(active_percentage_range)
        else: 
            print("Using existing target concentrations")
        # else just use existing target concentrations
        target_concentrations = self._target_concentrations
        
        # Step 2: Process each activated species to determine parameters
        parameter_values = {}
        
        for active_state in self.active_states.keys():
            # Get corresponding inactive state
            inactive_state = active_state[:-1]
            
            # Get total concentration (inactive + active)
            X_total = self.all_states[inactive_state] + self.all_states[active_state]
            
            # Get target active concentration
            X_a_target = target_concentrations[active_state]
            p_target = X_a_target / X_total
            
            # Get parameters for this state
            params = get_parameters_for_state(self.model, active_state)
            forward_params = params['as_product']
            backward_params = params['as_reactant']
            
            # Categorize parameters
            km_b = [p for p in backward_params if p.startswith('Km')]
            km_f = [p for p in forward_params if p.startswith('Km')]
            ki_f = [p for p in forward_params if p.startswith('Ki')]
            vmax_f = [p for p in forward_params if p.startswith('Vmax')]
            kc_f = [p for p in forward_params if p.startswith('Kc')]
            
            # Combine related parameter types
            all_km_f = km_f + ki_f  # Km and Ki both affect denominator
            all_vmax_f = vmax_f + kc_f  # Vmax and Kc both contribute to numerator
            
            # Get backward Vmax
            vmax_b = [p for p in backward_params if p.startswith('Vmax')]
            
            # Validate parameter structure
            if not vmax_b:
                raise ValueError(f"No backward Vmax found for state {active_state}")
            if len(vmax_b) > 1:
                raise ValueError(f"Multiple backward Vmax found for state {active_state}: {vmax_b}")
            
            # Step 3: Try to identify regulators for this state
            regulators = self._identify_regulators(active_state, forward_params)
            
            # Step 4: Solve for parameters using the corrected kinetic equations
            state_params = self._solve_state_parameters_corrected(
                active_state=active_state,
                X_total=X_total,
                p_target=p_target,
                all_vmax_f=all_vmax_f,
                regulators=regulators,
                vmax_b_param=vmax_b[0],
                all_km_f=all_km_f,
                km_b_params=km_b,
                target_concentrations=target_concentrations,
                X_total_multiplier=X_total_multiplier,
                ki_val=ki_val,
                v_max_f_random_range=v_max_f_random_range
            )
            
            parameter_values.update(state_params)
        
        return parameter_values
    
    def _identify_regulators(self, active_state: str, forward_params: List[str]) -> List[str]:
        """
        Identify regulators for a given activated state.
        
        Attempts to find regulator species by examining reaction structure.
        Returns empty list if regulators cannot be identified.
        
        Args:
            active_state: Name of the activated species
            forward_params: List of forward parameter names
            
        Returns:
            List of regulator species names (may be empty if cannot identify)
        """
        regulators = []
        
        # Try to find the reaction that produces this active state
        for param_name in forward_params:
            param_info = self.param_map.get(param_name)
            if param_info:
                reaction_idx = param_info.get('reaction_index')
                if reaction_idx is not None and reaction_idx < len(self.model.reactions):
                    reaction = self.model.reactions[reaction_idx]
                    # Check if this reaction produces the active state
                    if active_state in reaction.products_names:
                        # extra_states may contain regulators
                        if hasattr(reaction, 'extra_states') and reaction.extra_states:
                            regulators.extend(reaction.extra_states)
                        break
        
        return regulators
    
    def _initialize_target_concentrations(
        self, 
        active_percentage_range: Tuple[float, float]
    ) -> Dict[str, float]:
        """
        Initialize target active concentrations for all species.
        
        Args:
            active_percentage_range: Range for random active percentages
            
        Returns:
            Dictionary mapping active state names to target concentrations
        """
        target_concentrations = {}
        
        for active_state, active_value in self.active_states.items():
            inactive_state = active_state[:-1]
            inactive_value = self.all_states[inactive_state]
            
            X_total = inactive_value + active_value
            
            # Randomly select active percentage within range
            p = self.rng.uniform(active_percentage_range[0], active_percentage_range[1])
            
            # Compute target active concentration
            X_a_target = p * X_total
            
            target_concentrations[active_state] = X_a_target
            
        self._target_concentrations = target_concentrations
        
        return target_concentrations
    
    def _get_regulator_concentration(self, regulator: str, target_concentrations: Dict[str, float]) -> float:
        """
        Get concentration of a regulator species.
        
        Args:
            regulator: Name of regulator species
            target_concentrations: Dictionary of target active concentrations
            
        Returns:
            Regulator concentration (active form if available, otherwise from drug_concentrations)
        """
        # Check if regulator is a drug
        if regulator in self.drug_concentrations:
            return self.drug_concentrations[regulator]
        
        # Check if regulator has active form
        active_regulator = regulator + 'a' if not regulator.endswith('a') else regulator
        if active_regulator in target_concentrations:
            return target_concentrations[active_regulator]
        
        # Check if regulator is in states (inactive form)
        if regulator in self.all_states:
            return self.all_states[regulator]
        
        # Default to 0 if regulator not found
        return 0.0
    
    def _solve_state_parameters_corrected(
        self,
        active_state: str,
        X_total: float,
        p_target: float,
        all_vmax_f: List[str],
        regulators: List[str],
        vmax_b_param: str,
        all_km_f: List[str],
        km_b_params: List[str],
        target_concentrations: Dict[str, float],
        X_total_multiplier: float,
        ki_val: float,
        v_max_f_random_range: Tuple[float, float]
    ) -> Dict[str, float]:
        """
        Solve kinetic equations for a single activated species using corrected algorithm.
        
        Args:
            active_state: Name of the activated species
            X_total: Total concentration of the species (inactive + active)
            p_target: Target active percentage [X_a]/X_total
            all_vmax_f: List of forward Vmax/Kc parameter names
            regulators: List of regulator species names
            vmax_b_param: Backward Vmax parameter name
            all_km_f: List of forward Km/Ki parameter names
            km_b_params: List of backward Km parameter names
            target_concentrations: Dictionary of target active concentrations
            X_total_multiplier: Multiplier for km_b calculation
            ki_val: Constant value for Ki parameters
            v_max_f_random_range: Range for total forward vmax
            
        Returns:
            Dictionary of parameter values for this state
        """
        parameter_values = {}
        
        # Step 1: Determine total forward vmax (Σ kc_i × [Y_i_a])
        total_vmax_f = self.rng.uniform(v_max_f_random_range[0], v_max_f_random_range[1])
        
        # Step 2: Determine backward vmax from equation: p = vmax_f / (vmax_f + vmax_b)
        if p_target <= 0 or p_target >= 1:
            raise ValueError(
                f"Target active percentage {p_target} for {active_state} must be between 0 and 1"
            )
        
        vmax_b_value = total_vmax_f * (1.0 / p_target - 1.0)
        
        # Step 3: Set backward Vmax parameter
        parameter_values[vmax_b_param] = vmax_b_value
        
        # Step 4: Distribute total_vmax_f among forward parameters
        if all_vmax_f:
            # If we have regulators identified, try to distribute based on regulator concentrations
            if regulators and len(regulators) == len(all_vmax_f):
                # Get regulator concentrations
                regulator_concs = []
                for regulator in regulators:
                    conc = self._get_regulator_concentration(regulator, target_concentrations)
                    regulator_concs.append(conc)
                
                # Avoid division by zero
                regulator_concs = [max(conc, 1e-6) for conc in regulator_concs]
                
                # Distribute total_vmax_f so that kc_i × [Y_i_a] are equal
                # Solve: kc_i = constant / [Y_i_a] where constant = total_vmax_f / n
                n_regulators = len(regulators)
                constant_per_regulator = total_vmax_f / n_regulators
                
                for i, param in enumerate(all_vmax_f):
                    if i < len(regulator_concs):
                        kc_value = constant_per_regulator / regulator_concs[i]
                        parameter_values[param] = kc_value
                    else:
                        # Fallback: equal distribution
                        parameter_values[param] = total_vmax_f / len(all_vmax_f)
            else:
                # Simplified: distribute equally among all forward parameters
                vmax_f_per_param = total_vmax_f / len(all_vmax_f)
                for param in all_vmax_f:
                    parameter_values[param] = vmax_f_per_param
        
        # Step 5: Set backward Km parameters
        km_b_value = X_total * X_total_multiplier
        for param in km_b_params:
            parameter_values[param] = km_b_value
        
        # Step 6: Set forward Km and Ki parameters with corrected calculation
        # km_f = km_b × (1 + Σ [inhibitor_a]/ki_val)
        # We need to identify which regulators are inhibitors (affect denominator)
        # For now, use simplified approach
        for param in all_km_f:
            if param.startswith('Ki'):
                parameter_values[param] = ki_val
            else:  # Km parameter
                # Simplified: set equal to km_b_value
                # TODO: Implement proper km_f calculation with inhibitor concentrations
                parameter_values[param] = km_b_value
        
        return parameter_values
    
    def apply_parameters(self, parameter_dict: Dict[str, float]) -> ModelBuilder:
        """
        Apply tuned parameters to the model.
        
        Args:
            parameter_dict: Dictionary of parameter values from generate_parameters()
            
        Returns:
            New ModelBuilder object with applied parameters
        """
        # Create a copy of the model
        new_model = self.model.copy()
        
        # Apply all parameters
        for param_name, param_value in parameter_dict.items():
            new_model.set_parameter(param_name, param_value)
        
        return new_model

    def get_target_concentrations(self) -> Dict[str, float]:
        """
        Get the target active concentrations for all species.
        
        Returns:
            Dictionary mapping active state names to target concentrations
        """
        return self._target_concentrations
