"""
Unit tests for parameter_mapper module.
"""
import pytest
from models.utils.parameter_mapper import (
    get_parameter_reaction_map,
    find_parameter_by_role,
    explain_reaction_parameters,
    get_parameters_for_state
)


class TestParameterMapper:
    """Test parameter mapping utilities."""
    
    def test_get_parameter_reaction_map_basic(self, simple_model_builder):
        """Test basic parameter-reaction mapping."""
        param_map = get_parameter_reaction_map(simple_model_builder)
        
        # Should have parameters
        assert len(param_map) > 0
        
        # Check structure of first parameter
        first_param = list(param_map.values())[0]
        assert 'parameter_name' in first_param
        assert 'parameter_type' in first_param
        assert 'reaction_index' in first_param
        assert 'reactants' in first_param
        assert 'products' in first_param
    
    def test_get_parameter_reaction_map_multi_reaction(self, multi_reaction_model):
        """Test mapping with multiple reactions."""
        param_map = get_parameter_reaction_map(multi_reaction_model)
        
        # Should have 6 parameters (Km_J0, Vmax_J0, Km_J1, Vmax_J1, Km_J2, Vmax_J2)
        assert len(param_map) == 6
        
        # Check parameter names
        param_names = list(param_map.keys())
        assert any('Km_J0' in name for name in param_names)
        assert any('Vmax_J0' in name for name in param_names)
        
        # Check reaction indices
        for param_info in param_map.values():
            assert param_info['reaction_index'] in [0, 1, 2]
            
            # Reactions should have correct reactants/products
            if param_info['reaction_index'] == 0:
                assert 'R1' in param_info['reactants']
                assert 'R1a' in param_info['products']
            elif param_info['reaction_index'] == 2:
                assert 'O' in param_info['reactants']
                assert 'Oa' in param_info['products']
    
    def test_find_parameter_by_role_vmax(self, multi_reaction_model):
        """Test finding Vmax parameters."""
        vmax_params = find_parameter_by_role(multi_reaction_model, 'Vmax')
        
        # Should find Vmax parameters
        assert len(vmax_params) == 3  # Vmax_J0, Vmax_J1, Vmax_J2
        assert all('Vmax' in param for param in vmax_params)
    
    def test_find_parameter_by_role_km(self, multi_reaction_model):
        """Test finding Km parameters."""
        km_params = find_parameter_by_role(multi_reaction_model, 'Km')
        
        # Should find Km parameters
        assert len(km_params) == 3  # Km_J0, Km_J1, Km_J2
        assert all('Km' in param for param in km_params)
    
    def test_find_parameter_by_role_case_insensitive(self, multi_reaction_model):
        """Test case-insensitive role matching."""
        vmax_lower = find_parameter_by_role(multi_reaction_model, 'vmax')
        vmax_upper = find_parameter_by_role(multi_reaction_model, 'VMAX')
        
        # Should get same results
        assert len(vmax_lower) == len(vmax_upper)
        assert set(vmax_lower) == set(vmax_upper)
    
    def test_find_parameter_by_role_with_state(self, multi_reaction_model):
        """Test finding parameters affecting specific state."""
        # Find parameters affecting O
        o_params = find_parameter_by_role(multi_reaction_model, None, 'O')
        
        # Should find parameters for reaction 2 (O -> Oa)
        assert len(o_params) == 2  # Km_J2, Vmax_J2
        assert all('_J2' in param for param in o_params)
        
        # Find Vmax parameters for O
        o_vmax_params = find_parameter_by_role(multi_reaction_model, 'Vmax', 'O')
        assert len(o_vmax_params) == 1
        assert 'Vmax_J2' in o_vmax_params[0]
    
    def test_find_parameter_by_role_activated_forms(self, multi_reaction_model):
        """Test finding parameters for activated/deactivated forms."""
        # Find parameters affecting R1 (base form)
        r1_params = find_parameter_by_role(multi_reaction_model, None, 'R1')
        
        # Should find parameters for reaction 0 (R1 -> R1a)
        assert len(r1_params) == 2  # Km_J0, Vmax_J0
        assert all('_J0' in param for param in r1_params)
        
        # Find parameters affecting R1a (activated form)
        r1a_params = find_parameter_by_role(multi_reaction_model, None, 'R1a')
        
        # Should also find parameters for reaction 0
        assert len(r1a_params) == 2
        assert all('_J0' in param for param in r1a_params)
    
    def test_explain_reaction_parameters(self, multi_reaction_model):
        """Test explaining reaction parameters."""
        explanation = explain_reaction_parameters(multi_reaction_model, 0)
        
        # Should contain reaction information
        assert "Reaction 0" in explanation
        assert "R1" in explanation
        assert "R1a" in explanation
        
        # Should contain parameter explanations
        assert "Km_J0" in explanation
        assert "Vmax_J0" in explanation
        assert "Maximum rate" in explanation or "Michaelis constant" in explanation
    
    def test_explain_reaction_parameters_invalid_index(self, multi_reaction_model):
        """Test explaining parameters with invalid reaction index."""
        with pytest.raises(IndexError):
            explain_reaction_parameters(multi_reaction_model, 10)
    
    def test_get_parameters_for_state(self, multi_reaction_model):
        """Test getting all parameters affecting a state."""
        state_params = get_parameters_for_state(multi_reaction_model, 'O')
        
        # Should have correct structure
        assert 'as_reactant' in state_params
        assert 'as_product' in state_params
        assert 'all' in state_params
        
        # O is reactant in reaction 2
        assert len(state_params['as_reactant']) == 2
        assert len(state_params['all']) == 2
        assert all('_J2' in param for param in state_params['all'])
    
    def test_get_parameters_for_state_activated_form(self, multi_reaction_model):
        """Test getting parameters for activated form."""
        state_params = get_parameters_for_state(multi_reaction_model, 'R1a')
        
        # R1a is product in reaction 0
        assert len(state_params['as_product']) == 2
        assert len(state_params['all']) == 2
        assert all('_J0' in param for param in state_params['all'])
    
    def test_get_parameters_for_state_nonexistent(self, multi_reaction_model):
        """Test getting parameters for non-existent state."""
        with pytest.raises(ValueError, match="not found"):
            get_parameters_for_state(multi_reaction_model, 'Nonexistent')
    
    def test_parameter_mapping_uncompiled_model(self):
        """Test that uncompiled models raise error."""
        uncompiled_model = pytest.importorskip('models.ModelBuilder').ModelBuilder("uncompiled")
        
        with pytest.raises(ValueError, match="must be pre-compiled"):
            get_parameter_reaction_map(uncompiled_model)
        
        with pytest.raises(ValueError, match="must be pre-compiled"):
            find_parameter_by_role(uncompiled_model, 'Vmax')
    
    def test_model_spec4_integration(self, model_spec4_example):
        """Test parameter mapping with real ModelSpec4 model."""
        param_map = get_parameter_reaction_map(model_spec4_example)
        
        # Should have parameters
        assert len(param_map) > 0
        
        # Find Vmax parameters
        vmax_params = find_parameter_by_role(model_spec4_example, 'Vmax')
        assert len(vmax_params) > 0
        
        # Find Km parameters
        km_params = find_parameter_by_role(model_spec4_example, 'Km')
        assert len(km_params) > 0
        
        # Test state-specific search
        if 'R1' in model_spec4_example.states:
            r1_params = find_parameter_by_role(model_spec4_example, None, 'R1')
            assert len(r1_params) > 0
