# Python script to generate ModelSpec4 models for network analysis using simple path methods

# Append the right path for module imports
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "../../..")
sys.path.insert(0, src_dir)
from dotenv import dotenv_values
config = dotenv_values(dotenv_path=src_dir)

import numpy as np # noqa: E402
from models.Specs.ModelSpec4 import ModelSpec4 # noqa: E402
from models.Specs.Drug import Drug # noqa: E402
from models.Solver.RoadrunnerSolver import RoadrunnerSolver # noqa: E402 
from models.utils.kinetic_tuner import KineticParameterTuner  # noqa: E402
from models.utils.s3_config_manager import S3ConfigManager # noqa: E402

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize ModelSpec4 with 2 intermediate layers
model_spec = ModelSpec4(num_intermediate_layers=2)

# Generate specifications with custom regulations
model_spec.generate_specifications(
    num_cascades=3,
    num_regulations=0,
    random_seed=42
)

# Add custom regulations (from original config)
custom_regulations = [
    ['R1', 'R2', 'up'],
    ['R3', 'I1_2', 'up'],
    ['I1_1', 'I2_2', 'up'],
    ['I1_2', 'I2_1', 'down'],
    ['I1_2', 'I2_3', 'down'],
    ['I1_3', 'I2_2', 'up'],
    ['I2_1', 'R1', 'down'],
    ['I2_3', 'R3', 'up']
]

for regulation in custom_regulations:
    model_spec.add_regulation(*regulation)

# Create drug D that down-regulates R1
drug_d = Drug(
    name="D",
    start_time=5000.0,  # Drug applied at time 500
    default_value=500.0,  # Drug concentration
    regulation=["R1"],  # Regulates R1
    regulation_type=["down"],  # Down-regulation decreases activation
)

# Add drug to model
model_spec.add_drug(drug_d)
logger.info(f"Drug added: {drug_d.name} targeting {drug_d.regulation}")

# Verify drug validation works
logger.info(f"Drug species list: {[d.name for d in model_spec.drugs]}")
logger.info(f"Total regulations: {len(model_spec.regulations)}")

# Generate the model
model = model_spec.generate_network(
    network_name="ModelSpec4_Network",
    mean_range_species=(200, 1000),  # Initial concentrations range
    rangeScale_params=(0.7, 1.3),  # Parameter scale range
    rangeMultiplier_params=(0.99, 1.01),  # Small additional variation
    random_seed=42,
    receptor_basal_activation=True,  # Receptors have basal activation
)

logger.info(f"Model created: {model.name}")
logger.info(f"Total reactions: {len(model.reactions)}")
logger.info(f"Total states: {len(model.states)}")
logger.info(f"Total parameters: {len(model.parameters)}")

# Show a few key states
key_states = ["R1", "R1a", "I1_1", "I1_1a", "I2_1", "I2_1a", "O", "Oa"]
available_states = [s for s in key_states if s in model.states]
logger.info(f"Key states available: {available_states}")

tuner = KineticParameterTuner(model, random_seed=42)

### Results saving block using simple path methods

S3_manager = S3ConfigManager()
gen_path = S3_manager.save_result_path
folder_name = "sy_simple"  # Use the specified folder name
file_name = 'model_spec.pkl'
S3_manager.save_data_from_path(f"{gen_path}/models/{folder_name}/{file_name}", model_spec, data_format="pkl")

file_name = 'model_builder.pkl'
S3_manager.save_data_from_path(f"{gen_path}/models/{folder_name}/{file_name}", model, data_format="pkl")

file_name = 'model_tuner.pkl'
S3_manager.save_data_from_path(f"{gen_path}/models/{folder_name}/{file_name}", tuner, data_format="pkl")

item_list = S3_manager.list_files_from_path(f"{gen_path}/models/{folder_name}/")

logger.info("Files in S3:")
for item in item_list:
    logger.info(f"S3 item: {item}")
