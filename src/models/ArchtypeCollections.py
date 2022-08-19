from .ReactionArchtype import ReactionArchtype

michaelis_menten = ReactionArchtype(
    'Michaelis Menten',
    ('S',), ('E',),
    ('Km', 'Vmax'),
    'Vmax*S/(Km + S)',
    assume_parameters_values={'Km': 100, 'Vmax': 10},
    assume_reactant_values={'S': 100},
    assume_product_values={'E': 0})

mass_action_21 = ReactionArchtype(
    'Mass Action',
    ('A', 'B'), ('C'),
    ('ka', 'kd'),
    'ka*A*B - kd*C',
    assume_parameters_values={'ka': 0.001, 'kd': 0.01},
    assume_reactant_values={'A': 100, 'B': 100},
    assume_product_values={'C': 0})

mass_action_12 = ReactionArchtype(
    'Mass Action',
    ('C',), ('A', 'B'),
    ('ka', 'kd'),
    'ka*A*B - kd*C',
    assume_parameters_values={'ka': 0.001, 'kd': 0.01},
    assume_reactant_values={'C': 0},
    assume_product_values={'A': 100, 'B': 100})
