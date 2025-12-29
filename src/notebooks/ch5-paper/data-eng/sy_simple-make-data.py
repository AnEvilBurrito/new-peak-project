# create datasets for the sy_simple model, saving to S3 using S3ConfigManager

import sys
import os
from dotenv import dotenv_values

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "../../..")
sys.path.insert(0, src_dir)
config = dotenv_values(dotenv_path=src_dir)

from models.utils.data_generation_helpers import make_data_extended 
from models.utils.s3_config_manager import S3ConfigManager
from models.Solver.RoadrunnerSolver import RoadrunnerSolver
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

s3_manager = S3ConfigManager()

model_name = "sy_simple"
gen_path = s3_manager.save_result_path 
model_spec = s3_manager.load_data_from_path(f"{gen_path}/models/{model_name}/model_spec.pkl", data_format='pkl')
model = s3_manager.load_data_from_path(f"{gen_path}/models/{model_name}/model_builder.pkl", data_format='pkl')
tuner = s3_manager.load_data_from_path(f"{gen_path}/models/{model_name}/model_tuner.pkl", data_format='pkl')


solver = RoadrunnerSolver()
solver.compile(model.get_sbml_model())
state_variables = model.get_state_variables()

# get state variables that end with 'a' (active forms)
active_state_variables = {k: v for k, v in state_variables.items() if k.endswith("a")}

# filter out all active state variables
inactive_state_variables = {
    k: v for k, v in state_variables.items() if not k.endswith("a")
}
# further filter out 'O' if present
if "O" in inactive_state_variables:
    del inactive_state_variables["O"]
    

kinetic_parameters = model.get_parameters()
results = make_data_extended(
    initial_values=inactive_state_variables,
    perturbation_type="lognormal",
    perturbation_params={"shape": 0.5},
    parameter_values=kinetic_parameters,
    param_perturbation_type="lognormal",
    param_perturbation_params={"shape": 0.5},
    n_samples=10,
    model_spec=model_spec,
    solver=solver,
    simulation_params={"start": 0, "end": 10000, "points": 101},
    seed=42,
    outcome_var="Oa",
    capture_all_species=True,
)

X, y, parameters, timecourses, metadata = (
    results["features"],
    results["targets"],
    results["parameters"],
    results["timecourse"],
    results["metadata"],
)

gen_path = s3_manager.save_result_path
folder_name = model_name+"_data_v1"

file_name = 'features.pkl'
s3_manager.save_data_from_path(f"{gen_path}/models/{folder_name}/{file_name}", X, data_format="pkl")

file_name = 'targets.pkl'
s3_manager.save_data_from_path(f"{gen_path}/models/{folder_name}/{file_name}", y, data_format="pkl")

file_name = 'parameter_sets.pkl'
s3_manager.save_data_from_path(f"{gen_path}/models/{folder_name}/{file_name}", parameters, data_format="pkl")

file_name = 'timecourses.pkl'
s3_manager.save_data_from_path(f"{gen_path}/models/{folder_name}/{file_name}", timecourses, data_format="pkl")

file_name = 'metadata.pkl'
s3_manager.save_data_from_path(f"{gen_path}/models/{folder_name}/{file_name}", metadata, data_format="pkl")

item_list = s3_manager.list_files_from_path(f"{gen_path}/models/{folder_name}/")

logger.info("Files in S3:")
for item in item_list:
    logger.info(f"S3 item: {item}")