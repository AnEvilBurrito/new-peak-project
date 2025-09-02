import logging
from models.Specs.ModelSpec3 import ModelSpec3
from models.Specs.Drug import Drug

def generate_model_spec(spec_config, seed=None):
    """
    Generate a model specification with drug configuration.

    Args:
        spec_config (dict): The spec configuration containing:
            - n_layers: number of intermediate layers
            - n_cascades: number of cascades
            - n_regs: number of regulations
            - drug: drug configuration dict
            - gen_seed: generation seed (optional, can be overridden)
        seed (int, optional): Override seed for generation

    Returns:
        ModelSpec3: The complete model specification
    """
    logging.basicConfig(level=logging.INFO)

    # Extract parameters from spec_config
    n_layers = spec_config['n_layers']
    n_cascades = spec_config['n_cascades']
    n_regs = spec_config['n_regs']
    drug_config = spec_config['drug']
    seed = seed or spec_config.get('gen_seed')

    # Create ModelSpec3
    model_spec = ModelSpec3(num_intermediate_layers=n_layers)

    # Configure Drug
    drug_name = drug_config['name']
    drug_start = drug_config['start']
    drug_dose = drug_config['dose']
    drug_regulations = drug_config['regulations']
    drug_target_all = drug_config.get('target_all', False)

    drug = Drug(name=drug_name, start_time=drug_start, default_value=drug_dose)

    if drug_target_all:
        for n in range(n_cascades):
            target = f'R{n+1}'  # Assuming receptors are named R1, R2, ..., Rn
            drug.add_regulation(target, 'down')  # Assuming the type is 'down' for all
    else:
        for regs in drug_regulations:
            target, reg_type = regs[0], regs[1]
            drug.add_regulation(target, reg_type)

    # Generate specifications and add drug
    model_spec.generate_specifications(n_cascades, n_regs, seed)
    model_spec.add_drug(drug)

    return model_spec

def generate_network(builder_config, model_spec, seed=None):
    """
    Generate a network builder from the model specification.

    Args:
        builder_config (dict): The builder configuration containing:
            - ic_range: initial condition range
            - param_range: parameter range
            - param_mul_range: parameter multiplier range
        model_spec (ModelSpec3): The model specification
        seed (int, optional): Override seed for generation

    Returns:
        Builder: The network builder
    """
    param_gen_config = builder_config
    specie_range = param_gen_config['ic_range']
    param_range = param_gen_config['param_range']
    param_mul_range = param_gen_config['param_mul_range']

    builder = model_spec.generate_network('test', specie_range, param_range, param_mul_range, seed)
    return builder



import logging
import numpy as np 
from models.SyntheticGen import generate_feature_data_v3
from models.SyntheticGen import generate_target_data
from models.Solver.RoadrunnerSolver import RoadrunnerSolver
logging.basicConfig(level=logging.INFO)

spec_config = loaded_config['exp']['spec']
param_gen_config = loaded_config['exp']['parameter_generation']
dose_range_config = loaded_config['exp']['dose_range']
sim_config = loaded_config['exp']['simulation']
sim_start, sim_stop, sim_step = sim_config['start'], sim_config['stop'], sim_config['step']

n_cascades_perturb = dose_range_config['n_cascades_perturb']
start, stop, step = n_cascades_perturb['start'], n_cascades_perturb['stop'], n_cascades_perturb['step']
n_reps = dose_range_config['n_reps']
dr_seed = dose_range_config['dr_seed']
rng = np.random.default_rng(dr_seed)
seeds = rng.integers(0, 1e6, size=n_reps)

for n_cascades in range(start, stop + 1, step):
    for rep in range(n_reps):
        spec_config['n_cascades'] = n_cascades
        spec_config['gen_seed'] = int(seeds[rep])
        model_spec = generate_model_spec(spec_config, seed=42)
        logging.info(f"Generated model with {n_cascades} cascades.")
        builder = generate_network(param_gen_config, model_spec, seed=42)
        logging.info(f"Generated network builder for model with {n_cascades} cascades.")

        initial_values = builder.get_state_variables()
        # exclude all activated species from initial_values
        initial_values = {k: v for k, v in initial_values.items() if not k.endswith("a")}
        # exclude 'O' and 'Oa' from perturbation
        initial_values = {k: v for k, v in initial_values.items() if k not in ["O", "Oa"]}

        feature_data = generate_feature_data_v3(
            model_spec, initial_values, "lhs", {"min": 200, "max": 1000}, 500, spec_config['gen_seed']
        )
        solver = RoadrunnerSolver()
        solver.compile(builder.get_sbml_model())
        sim_params = {"start": sim_start, "end": sim_stop, "points": sim_step}

        target_data, timecourse_data = generate_target_data(
            model_spec, solver, feature_data, sim_params, outcome_var="Oa", verbose=True
        )
        
        # calculate the mean and std of target_data
        target_mean = np.mean(target_data, axis=0)
        target_std = np.std(target_data, axis=0)
    