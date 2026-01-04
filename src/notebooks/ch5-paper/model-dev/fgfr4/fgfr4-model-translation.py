# Python script to load FGFR4 model in sbml format directly into a Roadrunner object

# Append the right path for module imports
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "../../../..")
sys.path.insert(0, src_dir)
from dotenv import dotenv_values

config = dotenv_values(dotenv_path=src_dir)

import numpy as np  # noqa: E402

from models.Solver.RoadrunnerSolver import RoadrunnerSolver  # noqa: E402
from models.Solver.ScipySolver2 import ScipySolver2  # noqa: E402
from models.utils.s3_config_manager import S3ConfigManager  # noqa: E402

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

solver = RoadrunnerSolver()

# Open the FGFR4 model file directly as an string file, the file is located within the same directory
model_file_path = os.path.join(current_dir, "FGFR4_model_rev2a.xml")
with open(model_file_path, "r") as file:
    model_content = file.read()
# print the first 2 lines of the model file to verify it is read correctly
print("\n".join(model_content.splitlines()[:2]))
solver.compile(model_content)
print("Model loaded successfully into Roadrunner.")

# loading in parameters 

import pandas as pd
params_vals = pd.read_csv(os.path.join(current_dir, "fitted_paramsets_rev2_STEP3.csv"))
param_names = [
    'kc01f', 'Vm01r', 'ki01r', 'kc02f', 'Ki02f', 'Vm02r', 'ki02r1', 'ki02r2',
    'kc03f', 'Vm03r', 'ki03r', 'kc04f', 'ki04f', 'Vm04r', 'kc05f1', 'kc05f2',
    'kc05f3', 'Vm05r', 'kc06f', 'Ki06f', 'kc06r', 'kc07f', 'Vm07r', 'kc08f',
    'Vm08r', 'kc09f', 'Vm09r', 'kc10f', 'Vm10r', 'kc11f', 'ki11f', 'ki11r',
    'Vm11r', 'kc12f1', 'kc12f2', 'kc12f3', 'ki12f', 'Vm12r', 'kc13f', 'Vm13r',
    'kc14f1', 'kc14f2', 'Vm14r', 'kc15f1', 'kc15f2', 'ki15f', 'Vm15r', 'kc16f',
    'ki16f1', 'ki16f2', 'Vm16r', 'kc17f', 'Vm17r', 'kc18f', 'Vm18r', 'kc19f',
    'alpha19f', 'ki19f', 'Vm19r', 'kc20f', 'alpha20f', 'ki20f1', 'ki20f2',
    'Vm20r', 'vs21', 'kc21', 'Km21', 'kc23', 'Km23', 'kc24', 'kc25f', 'kc25r',
    'kc26', 'vs27', 'kc27a', 'Km27a', 'kc27b', 'Km27b', 'kc29', 'Km29', 'kc31f',
    'Vm31r', 'n02', 'Kmf02', 'n06', 'Kmf06', 'kc03f2', 'ki03f', 'Ki03f2',
    'kc05f4', 'kc32f', 'Vm32r', 'Ki08f', 'Ki17f', 'kc33f', 'Vm33r', 'kc08f1',
    'ki33f', 'kc09f1', 'Kmf08', 'Kmf17', 'Kmf03', 'PTEN', 'IGF0', 'HRG0',
    'FGF0', 'IGF_on', 'FGF_on', 'HRG_on', 'FGFR4i_0', 'PI3Ki_0', 'ERBBi_0',
    'AKTi_0', 'MEKi_0', 'inh_on'
]

# Use parameter set at index 0
# Test all parameter sets to find successful simulations
successful_indices = []

for idx in range(len(params_vals)):
    try:
        candidate_paramset = params_vals.iloc[idx]
        param_dict = {name: candidate_paramset[i] for i, name in enumerate(param_names)}
        solver.set_parameter_values(param_dict)
        res = solver.simulate(0, 5000, 100)
        successful_indices.append(idx)
        logger.info(f"Successfully simulated with parameter set {idx}")
    except Exception as e:
        logger.warning(f"Failed to simulate with parameter set {idx}: {str(e)}")

logger.info(f"Successful parameter sets: {successful_indices}")

# Create a quick plot of the simulation results
# res is a dataframe with 'time' as a column and other species as columns

myreadouts = [
    "pAkt",
    "pIGFR",
    "pFGFR4",
    "pERBB",
    "pIRS",
    "aPI3K",
    "PIP3",
    "pFRS2",
    "aGrb2",
    "aPDK1",
    "amTORC1",
    "pS6K",
    "aSos",
    "aShp2",
    "aRas",
    "aRaf",
    "pMEK",
    "pERK",
    "aGAB1",
    "aGAB2",
    "SPRY2",
    "pSPRY2",
    "PTP",
    "aCbl",
    "FOXO",
    "amTORC2",
]

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
for readout in myreadouts:
    plt.plot(res['time'], res[readout], label=readout)
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('FGFR4 Model Simulation Results')
plt.legend()
plt.show()