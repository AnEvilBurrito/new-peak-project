# Python script to generate degree specification models for network analysis

# Append the right path for module imports
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "../../..")
sys.path.insert(0, src_dir)

import numpy as np # noqa: E402
from models.Specs.DegreeInteractionSpec import DegreeInteractionSpec # noqa: E402
from models.Specs.Drug import Drug # noqa: E402
from models.Solver.RoadrunnerSolver import RoadrunnerSolver # noqa: E402 
from models.utils.kinetic_tuner import KineticParameterTuner  # noqa: E402
from models.utils.s3_config_manager import S3ConfigManager # noqa: E402

from dotenv import dotenv_values
config = dotenv_values(".env")

# Initialize degree interaction specification
degree_spec = DegreeInteractionSpec(degree_cascades=[1, 2])

# Generate complete specifications with moderate feedback density
degree_spec.generate_specifications(
    random_seed=42,
    feedback_density=1,  # 30% of cascades get upward feedback (mandatory downward always present)
)

# Create drug D that down-regulates R1_1
drug_d = Drug(
    name="D",
    start_time=5000.0,  # Drug applied at time 5000
    default_value=100.0,  # Drug concentration
    regulation=["R1_1"],  # Regulates R1_1 (degree 1, cascade 1)
    regulation_type=["down"],  # Down-regulation decreases activation
)

# Add drug to model
degree_spec.add_drug(drug_d)
print(f"Drug added: {drug_d.name} targeting {drug_d.regulation}")

# Verify drug validation works
print(f"Drug species list: {[d.name for d in degree_spec.drugs]}")
print(f"Total regulations: {len(degree_spec.regulations)}")

# Generate the model
model = degree_spec.generate_network(
    network_name="MultiDegree_Kinetics",
    mean_range_species=(50, 150),  # Initial concentrations
    rangeScale_params=(0.8, 1.2),  # ±20% variation
    rangeMultiplier_params=(0.9, 1.1),  # Small additional variation
    random_seed=42,
    receptor_basal_activation=False,  # Receptors have basal activation
)

print(f"Model created: {model.name}")
print(f"Total reactions: {len(model.reactions)}")
print(f"Total states: {len(model.states)}")
print(f"Total parameters: {len(model.parameters)}")

# Show a few key states
key_states = ["R1_1", "I1_1", "R2_1", "I2_1", "R3_1", "I3_1", "O"]
available_states = [s for s in key_states if s in model.states]
print(f"Key states available: {available_states}")


tuner = KineticParameterTuner(model, random_seed=42)
updated_params = tuner.generate_parameters(active_percentage_range=(0.3, 0.7))

print("Tuned Parameters:")
for param, value in updated_params.items():
    print(f"  {param}: {value:.3f}")
    
for param in updated_params:
    # param is a dict key, so we need to get its value
    print(f"Setting parameter {param} to {updated_params[param]:.3f}")
    model.set_parameter(param, updated_params[param])

# model.set_parameter("Ki0_J1", 0.01)

print(model.get_antimony_model())
target_concentrations = tuner.get_target_concentrations()
for t in target_concentrations.items():
    print(f"Target concentration for {t[0]}: {t[1]:.3f}")
    
    
regulator_parameter_map = model.get_regulator_parameter_map()
drug_map = regulator_parameter_map.get("D", {})
drug_param = drug_map[0]
print(f"Drug D regulates parameters: {drug_map[0]}")

model.set_parameter(drug_param, 10)  # Set to 0 to simulate drug effect

# Save model_spec and model_builder to S3 

S3_manager = S3ConfigManager()
S3_manager.save_data()