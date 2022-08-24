from .ReactionArchtype import ReactionArchtype

michaelis_menten = ReactionArchtype(
    'Michaelis Menten',
    ('&S',), ('&E',),
    ('Km', 'Vmax'),
    'Vmax*&S/(Km + &S)',
    assume_parameters_values={'Km': 100, 'Vmax': 10},
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