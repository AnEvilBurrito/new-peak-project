"""
Unit tests for regulator-parameter mapping in Reaction and ModelBuilder.
"""
import pytest
from models.Reaction import Reaction
from models.ArchtypeCollections import (
    create_archtype_michaelis_menten_v2,
    create_archtype_michaelis_menten,
    create_archtype_basal_michaelis,
    create_archtype_mass_action,
)


class TestRegulatorParameterMapping:
    """Test mapping between regulators and parameters."""
    
    def test_mapping_v2_stimulator_weak_and_inhibitors(self):
        """Test mapping for archtype with stimulator_weak and allosteric inhibitors."""
        arch = create_archtype_michaelis_menten_v2(
            stimulator_weak=2,
            allosteric_inhibitors=1
        )
        # Create reaction with dummy extra_state names
        reaction = Reaction(
            arch,
            reactants=('S',),
            products=('P',),
            extra_states=('Reg1', 'Reg2', 'Reg3'),
            zero_init=False
        )
        # Expected mapping: Kc0 -> Reg1, Kc1 -> Reg2, Ki0 -> Reg3
        assert reaction.get_regulator_for_parameter('Kc0') == 'Reg1'
        assert reaction.get_regulator_for_parameter('Kc1') == 'Reg2'
        assert reaction.get_regulator_for_parameter('Ki0') == 'Reg3'
        assert reaction.get_regulator_for_parameter('Km') == ''  # Km not associated
        
    def test_mapping_v2_stimulators(self):
        """Test mapping for archtype with stimulators (strong)."""
        arch = create_archtype_michaelis_menten_v2(stimulators=2)
        reaction = Reaction(
            arch,
            reactants=('S',),
            products=('P',),
            extra_states=('A1', 'A2'),
            zero_init=False
        )
        assert reaction.get_regulator_for_parameter('Ka0') == 'A1'
        assert reaction.get_regulator_for_parameter('Ka1') == 'A2'
        
    def test_mapping_basal_michaelis(self):
        """Test mapping for basal archtype."""
        arch = create_archtype_basal_michaelis(
            stimulator_weak=1,
            competitive_inhibitors=1
        )
        reaction = Reaction(
            arch,
            reactants=('S',),
            products=('P',),
            extra_states=('W1', 'I1'),
            zero_init=False
        )
        # Note: archtype parameters include Km, Kc, Kc0, Kic0
        # Kc0 -> W1, Kic0 -> I1
        assert reaction.get_regulator_for_parameter('Kc0') == 'W1'
        assert reaction.get_regulator_for_parameter('Kic0') == 'I1'
        
    def test_mapping_mass_action(self):
        """Test mapping for mass action archtype with regulators."""
        arch = create_archtype_mass_action(
            reactant_count=1,
            product_count=1,
            allo_stimulators=1,
            allo_inhibitors=1
        )
        reaction = Reaction(
            arch,
            reactants=('A',),
            products=('B',),
            extra_states=('Stim', 'Inh'),
            zero_init=False
        )
        # parameters: Ka, Kd, Ks0, Ki0
        assert reaction.get_regulator_for_parameter('Ks0') == 'Stim'
        assert reaction.get_regulator_for_parameter('Ki0') == 'Inh'
        
    def test_modelbuilder_mapping(self):
        """Test ModelBuilder aggregator mapping."""
        from models.ModelBuilder import ModelBuilder
        arch = create_archtype_michaelis_menten_v2(stimulator_weak=1)
        model = ModelBuilder('test')
        reaction = Reaction(
            arch,
            reactants=('S',),
            products=('P',),
            extra_states=('Reg1',),
            zero_init=False
        )
        model.add_reaction(reaction)
        model.precompile()
        regulator_map = model.get_regulator_parameter_map()
        param_map = model.get_parameter_regulator_map()
        # Expect mapping: Reg1 -> ['Kc0_J0']
        assert 'Reg1' in regulator_map
        assert regulator_map['Reg1'] == ['Kc0_J0']
        assert param_map['Kc0_J0'] == 'Reg1'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
