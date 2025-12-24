"""
Test fixtures for ModelBuilder utility tests.
"""
import pytest
import numpy as np
import sys
import os

# Ensure we can import from models
# Add src to path if not already there
src_path = os.path.join(os.path.dirname(__file__), '..', '..')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from models.ModelBuilder import ModelBuilder
from models.Reaction import Reaction
from models.ReactionArchtype import ReactionArchtype
from models.Specs.ModelSpec4 import ModelSpec4


@pytest.fixture
def simple_reaction_archtype():
    """Create a simple Michaelis-Menten reaction archtype."""
    return ReactionArchtype(
        "MichaelisMenten",
        ("&S",),
        ("&E",),
        ("Km", "Vmax"),
        "Vmax*&S/(Km + &S)",
        assume_parameters_values={"Km": 100.0, "Vmax": 10.0},
        assume_reactant_values={"&S": 100.0},
        assume_product_values={"&E": 0.0}
    )


@pytest.fixture
def simple_model_builder(simple_reaction_archtype):
    """Create a simple ModelBuilder with one reaction."""
    model = ModelBuilder("test_model")
    
    # Add a simple reaction O -> Oa
    reaction = Reaction(
        simple_reaction_archtype,
        ("O",),
        ("Oa",),
        reactant_values={"O": 100.0},
        product_values={"Oa": 0.0},
        parameters_values={"Km": 50.0, "Vmax": 5.0}
    )
    
    model.add_reaction(reaction)
    model.precompile()
    return model


@pytest.fixture
def multi_reaction_model():
    """Create ModelBuilder with multiple reactions for comprehensive testing."""
    model = ModelBuilder("multi_reaction_model")
    
    # Reaction 1: R1 -> R1a
    archtype1 = ReactionArchtype(
        "MichaelisMenten",
        ("&S",),
        ("&E",),
        ("Km", "Vmax"),
        "Vmax*&S/(Km + &S)",
        assume_parameters_values={"Km": 100.0, "Vmax": 10.0},
        assume_reactant_values={"&S": 100.0},
        assume_product_values={"&E": 0.0}
    )
    reaction1 = Reaction(
        archtype1,
        ("R1",),
        ("R1a",),
        reactant_values={"R1": 100.0},
        product_values={"R1a": 0.0},
        parameters_values={"Km": 100.0, "Vmax": 8.0}
    )
    
    # Reaction 2: R2 -> R2a
    archtype2 = ReactionArchtype(
        "MichaelisMenten",
        ("&S",),
        ("&E",),
        ("Km", "Vmax"),
        "Vmax*&S/(Km + &S)",
        assume_parameters_values={"Km": 100.0, "Vmax": 10.0},
        assume_reactant_values={"&S": 100.0},
        assume_product_values={"&E": 0.0}
    )
    reaction2 = Reaction(
        archtype2,
        ("R2",),
        ("R2a",),
        reactant_values={"R2": 100.0},
        product_values={"R2a": 0.0},
        parameters_values={"Km": 80.0, "Vmax": 12.0}
    )
    
    # Reaction 3: O -> Oa
    archtype3 = ReactionArchtype(
        "MichaelisMenten",
        ("&S",),
        ("&E",),
        ("Km", "Vmax"),
        "Vmax*&S/(Km + &S)",
        assume_parameters_values={"Km": 100.0, "Vmax": 10.0},
        assume_reactant_values={"&S": 100.0},
        assume_product_values={"&E": 0.0}
    )
    reaction3 = Reaction(
        archtype3,
        ("O",),
        ("Oa",),
        reactant_values={"O": 100.0},
        product_values={"Oa": 0.0},
        parameters_values={"Km": 120.0, "Vmax": 6.0}
    )
    
    model.add_reaction(reaction1)
    model.add_reaction(reaction2)
    model.add_reaction(reaction3)
    model.precompile()
    return model


@pytest.fixture
def model_spec4_example():
    """Create a ModelSpec4 example model for integration testing."""
    spec = ModelSpec4(num_intermediate_layers=2)
    spec.generate_specifications(
        num_cascades=3,
        num_regulations=2,
        random_seed=42
    )
    
    model = spec.generate_network(
        "test_network",
        mean_range_species=(100, 200),
        rangeScale_params=(0.5, 2.0),
        rangeMultiplier_params=(0.9, 1.1),
        random_seed=42,
        receptor_basal_activation=True
    )
    return model


@pytest.fixture
def rng_seed():
    """Provide a fixed random seed for reproducible tests."""
    return 42


@pytest.fixture
def expected_parameter_names():
    """Expected parameter names for multi_reaction_model."""
    return ['Km_J0', 'Vmax_J0', 'Km_J1', 'Vmax_J1', 'Km_J2', 'Vmax_J2']


@pytest.fixture
def expected_state_names():
    """Expected state names for multi_reaction_model."""
    return ['R1', 'R1a', 'R2', 'R2a', 'O', 'Oa']
