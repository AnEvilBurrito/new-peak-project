from .ReactionArchtype import ReactionArchtype

michaelis_menten = ReactionArchtype(
    'Michaelis Menten',
    ('&S',), ('&E',),
    ('Km', 'Vmax'),
    'Vmax*&S/(Km + &S)',
    assume_parameters_values={'Km': 100, 'Vmax': 10},
    assume_reactant_values={'&S': 100},
    assume_product_values={'&E': 0})

michaelis_menten_fixed = ReactionArchtype(
    'Michaelis Menten',
    ('&S',), ('&E',),
    (),
    '100*&S/(1 + &S)',
    assume_reactant_values={'&S': 100},
    assume_product_values={'&E': 0})

mass_action_21 = ReactionArchtype(
    'Mass Action',
    ('&A', '&B'), ('&C',),
    ('ka', 'kd'),
    'ka*&A*&B - kd*&C',
    assume_parameters_values={'ka': 0.001, 'kd': 0.01},
    assume_reactant_values={'&A': 100, '&B': 100},
    assume_product_values={'&C': 0},
    reversible=True,
    reverse_rate_law='kd*&C- ka*&A*&B')

michaelis_menten_stim = ReactionArchtype(
    'Michaelis Menten',
    ('&S',), ('&E',),
    ('Km', 'Vmax'),
    'Vmax*&S*&I/(Km + &S)',
    extra_states=('&I',),
    assume_parameters_values={'Km': 100, 'Vmax': 10},
    assume_reactant_values={'&S': 100},
    assume_product_values={'&E': 0})

michaelis_menten_inh_allosteric = ReactionArchtype(
    'Michaelis Menten',
    ('&S',), ('&E',),
    ('Km', 'Vmax', 'Ki'),
    'Vmax*&S/(Km + &S)*(1+&I*Ki)',
    extra_states=('&I',),
    assume_parameters_values={'Km': 100, 'Vmax': 10, 'Ki': 0.01},
    assume_reactant_values={'&S': 100},
    assume_product_values={'&E': 0})

michaelis_menten_inh_competitive_1 = ReactionArchtype(
    'Michaelis Menten',
    ('&S',), ('&E',),
    ('Km', 'Vmax', 'Ki'),
    'Vmax*&S/(Km*(1+&I*Ki) + &S)',
    extra_states=('&I',),
    assume_parameters_values={'Km': 100, 'Vmax': 10, 'Ki': 0.01},
    assume_reactant_values={'&S': 100},
    assume_product_values={'&E': 0})


def create_archtype_michaelis_menten(stimulators=0, stimulator_weak=0, allosteric_inhibitors=0, competitive_inhibitors=0):

    if stimulators + allosteric_inhibitors + competitive_inhibitors + stimulator_weak == 0:
        return michaelis_menten

    # create the archtype

    archtype_name = 'Michaelis Menten General'

    reactants = ('&S',)
    products = ('&E',)
    upper_equation = 'Vmax*&S'
    lower_equation = '(Km + &S)'
    total_extra_states = ()
    parameters = ('Km', 'Vmax')
    assume_parameters_values={'Km': 100, 'Vmax': 10}

    if stimulators > 0:
        # add the stimulators to the equation
        stim_str = '*('
        for i in range(stimulators):
            stim_str += f'&A{i}*Ka{i}+'
        
        upper_equation += stim_str[:-1] + ')'

        # fill extra states
        extra_states = tuple([f'&A{i}' for i in range(stimulators)])
        total_extra_states += extra_states

        # fill parameters
        parameters += tuple([f'Ka{i}' for i in range(stimulators)])

        # fill assume parameters values
        assume_parameters_values.update({f'Ka{i}': 0.01 for i in range(stimulators)})

    if stimulator_weak > 0:
        # weak stimulators represent that stimulant is not required for the reaction to occur
        stim_weak_str = '(Vmax+'
        for i in range(stimulator_weak):
            stim_weak_str += f'&W{i}*Kw{i}+'
        
        upper_equation = stim_weak_str[:-1] + ')' + upper_equation[4:]

        # fill extra states
        extra_states = tuple([f'&W{i}' for i in range(stimulator_weak)])
        total_extra_states += extra_states

        # fill parameters
        parameters += tuple([f'Kw{i}' for i in range(stimulator_weak)])

        # fill assume parameters values
        assume_parameters_values.update({f'Kw{i}': 100 for i in range(stimulator_weak)})

    if allosteric_inhibitors > 0:
        # add the allosteric inhibitors to the equation
        inhb_allo_str = '*(1+'
        for i in range(allosteric_inhibitors):
            inhb_allo_str += f'&L{i}*Kil{i}+'
        
        inhb_allo_str = inhb_allo_str[:-1] + ')'

        lower_equation += inhb_allo_str

        # fill extra states
        extra_states = tuple([f'&L{i}' for i in range(allosteric_inhibitors)])
        total_extra_states += extra_states

        # fill parameters
        parameters += tuple([f'Kil{i}' for i in range(allosteric_inhibitors)])

        # fill assume parameters values
        assume_parameters_values.update({f'Kil{i}': 0.01 for i in range(allosteric_inhibitors)})
    
    if competitive_inhibitors > 0:
        # add the competitive inhibitors to the equation
        inhb_comp_str = '(Km*(1+'
        for i in range(competitive_inhibitors):
            inhb_comp_str += f'&I{i}*Kic{i}+'
        
        inhb_comp_str = inhb_comp_str[:-1] + '))'

        lower_equation = inhb_comp_str + lower_equation[3:]

        # fill extra states
        extra_states = tuple([f'&I{i}' for i in range(competitive_inhibitors)])
        total_extra_states += extra_states

        # fill parameters
        parameters += tuple([f'Kic{i}' for i in range(competitive_inhibitors)])

        # fill assume parameters values
        assume_parameters_values.update({f'Kic{i}': 0.01 for i in range(competitive_inhibitors)})

    full_equation = f'{upper_equation}/{lower_equation}'

    general_reaction = ReactionArchtype(
        archtype_name,
        reactants, products,
        parameters,
        full_equation,
        extra_states=total_extra_states,
        assume_parameters_values=assume_parameters_values,
        assume_reactant_values={'&S': 100},
        assume_product_values={'&E': 0})

    return general_reaction
