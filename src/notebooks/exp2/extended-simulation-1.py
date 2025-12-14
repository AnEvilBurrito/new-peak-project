# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: new-peak-project
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Init

# %%
import os

path = os.getcwd()
# find the string 'project' in the path, return index
index_project = path.find("project")
# slice the path from the index of 'project' to the end
project_path = path[: index_project + 7]
# set the working directory
os.chdir(project_path + "/src")
print(f"Project path set to: {os.getcwd()}")


# %%
from dotenv import dotenv_values
config = dotenv_values(".env")
print(config["DATA_PATH"])

# %%
new_path = config["NEW_DATA_PATH"]
print(f'New data path set to: {new_path}')

# %% [markdown]
# # Config

# %%
from models.utils.config_manager import initialise_config

initialise_config(folder_name="extended-simulation-1", verbose=1)


# %%
# or load existing config
from models.utils.config_manager import load_configs, print_config

loaded_config = load_configs(folder_name="extended-simulation-1", config_suffix="CuratedModel1")
print_config(loaded_config)

# %% [markdown]
# # Run

# %% [markdown]
# ### Setup

# %%
from models.Specs.ModelSpec4 import ModelSpec4

notebook_config = loaded_config["notebook"]
config_name = notebook_config['version']
exp_config = loaded_config["exp"]
spec_config = exp_config['spec']
n_layers = spec_config['n_layers']
new_spec = ModelSpec4(num_intermediate_layers=n_layers)


# %%
from models.Specs.Drug import Drug

drug_config = spec_config['drug']
drug_name = drug_config['name']
drug_start = drug_config['start']
drug_dose = drug_config['dose']
drug_regulations = drug_config['regulations']


n_cascades = spec_config["n_cascades"]
n_regs = spec_config["n_regs"]
seed = spec_config["gen_seed"]

for regulation in spec_config["custom_regulations"]:
    new_spec.add_regulation(*regulation)

new_drug = Drug(name=drug_name, start_time=drug_start, default_value=drug_dose)

# check if target_all exists in drug_config, if not set to False
drug_target_all = drug_config.get('target_all', False)

if drug_target_all:
    # If the drug targets all receptors, we don't need to add specific regulations
    for n in range(n_cascades):
        target = f'R{n+1}' # assuming receptors are named R1, R2, ..., Rn
        new_drug.add_regulation(target, 'down') # assuming the type is 'down' for all
else: 
    for regs in drug_regulations:
        target, type = regs[0], regs[1]
        new_drug.add_regulation(target, type)

new_spec.generate_specifications(n_cascades, n_regs, seed)
new_spec.add_drug(new_drug)

# %%
import numpy as np

param_sampling_config = exp_config["parameter_sampling"]
base_sampling_seed = param_sampling_config["sampling_seed"]
num_models = param_sampling_config["num_models"]
# based on base seed create 1000 different seeds
seeds = np.random.default_rng(base_sampling_seed).integers(0, 1000000, size=num_models)

param_gen_config = exp_config['parameter_generation']
basal_activation = spec_config["basal_activation"]
specie_range = param_gen_config['ic_range']
param_range = param_gen_config['param_range']
param_mul_range = param_gen_config['param_mul_range']

builder_models = []
for seed in seeds:
    builder = new_spec.generate_network(
        config_name,
        specie_range,
        param_range,
        param_mul_range,
        seed,
        receptor_basal_activation=basal_activation,
    )
    builder_models.append(builder)
print(f"Generated {len(builder_models)} models.")

# %% [markdown]
# ### Simulations

# %%
from models.Solver.RoadrunnerSolver import RoadrunnerSolver
from models.utils.config_manager import save_data
from tqdm import tqdm
import pandas as pd


sim_config = exp_config["simulation"]
sim_start = sim_config["start"]
sim_stop = sim_config["stop"]
sim_step = sim_config["step"]

# --- Simulations ---
solver_models = []
for i, builder in enumerate(tqdm(builder_models, desc="Creating solvers")):
    solver = RoadrunnerSolver()
    solver.compile(builder.get_sbml_model())
    solver_models.append(solver)


sim_results = []
for i, solver in enumerate(tqdm(solver_models, desc="Simulating models")):
    res = solver.simulate(sim_start, sim_stop, sim_step)
    sim_results.append(res)


# Concatenate all simulation results into a single DataFrame with 'seed' as its column using seeds list
sim_results_df = pd.concat(
    [pd.DataFrame(res).assign(seed=seed) for res, seed in zip(sim_results, seeds)],
    ignore_index=True,
)




# %%
# save the results
save_data(
    notebook_config=notebook_config,
    data=sim_results_df,
    data_name="sim_results_df",
    data_format="pkl",
    verbose=1,
)

# save the results
save_data(
    notebook_config=notebook_config,
    data=sim_results,
    data_name="sim_results",
    data_format="pkl",
    verbose=1,
)
