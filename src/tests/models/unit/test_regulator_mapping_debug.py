"""
Debug test for regulator-parameter mapping.
"""
import sys
sys.path.insert(0, 'src')
import re
from models.Reaction import Reaction
from models.ArchtypeCollections import (
    create_archtype_michaelis_menten_v2,
    create_archtype_michaelis_menten,
    create_archtype_basal_michaelis,
    create_archtype_mass_action,
)


def test_mapping_debug():
    """Debug mapping for various archtypes."""
    # Test 1
    arch = create_archtype_michaelis_menten_v2(
        stimulator_weak=2,
        allosteric_inhibitors=1
    )
    reaction = Reaction(arch, reactants=('S',), products=('P',), 
                       extra_states=('Reg1', 'Reg2', 'Reg3'), zero_init=False)
    print('Test 1: stimulator_weak=2, allosteric_inhibitors=1')
    print('Archtype extra_states:', arch.extra_states)
    print('Archtype parameters:', arch.parameters)
    print('Reaction extra_states:', reaction.extra_states)
    print('extra_states_names_to_archtype_names:', reaction.extra_states_names_to_archtype_names)
    print('regulator_parameters:', reaction.regulator_parameters)
    print('parameter_regulators:', reaction.parameter_regulators)
    
    # Check mapping
    expected = {'Kc0': 'Reg1', 'Kc1': 'Reg2', 'Ki0': 'Reg3'}
    for param, reg in expected.items():
        actual = reaction.get_regulator_for_parameter(param)
        print(f'{param} -> {actual} (expected {reg})')
        assert actual == reg, f'Mapping mismatch for {param}'
    
    # Test 2: stimulators
    arch2 = create_archtype_michaelis_menten_v2(stimulators=2)
    reaction2 = Reaction(arch2, reactants=('S',), products=('P',),
                        extra_states=('A1', 'A2'), zero_init=False)
    print('\nTest 2: stimulators=2')
    print('Archtype extra_states:', arch2.extra_states)
    print('Archtype parameters:', arch2.parameters)
    print('regulator_parameters:', reaction2.regulator_parameters)
    print('parameter_regulators:', reaction2.parameter_regulators)
    expected2 = {'Ka0': 'A1', 'Ka1': 'A2'}
    for param, reg in expected2.items():
        actual = reaction2.get_regulator_for_parameter(param)
        print(f'{param} -> {actual} (expected {reg})')
        assert actual == reg, f'Mapping mismatch for {param}'
    
    # Test 3: basal michaelis
    arch3 = create_archtype_basal_michaelis(
        stimulator_weak=1,
        competitive_inhibitors=1
    )
    reaction3 = Reaction(arch3, reactants=('S',), products=('P',),
                        extra_states=('W1', 'I1'), zero_init=False)
    print('\nTest 3: basal with weak stimulator and competitive inhibitor')
    print('Archtype extra_states:', arch3.extra_states)
    print('Archtype parameters:', arch3.parameters)
    print('regulator_parameters:', reaction3.regulator_parameters)
    print('parameter_regulators:', reaction3.parameter_regulators)
    expected3 = {'Kc0': 'W1', 'Kic0': 'I1'}
    for param, reg in expected3.items():
        actual = reaction3.get_regulator_for_parameter(param)
        print(f'{param} -> {actual} (expected {reg})')
        assert actual == reg, f'Mapping mismatch for {param}'
    
    print('\nAll debug tests passed.')


if __name__ == '__main__':
    test_mapping_debug()
