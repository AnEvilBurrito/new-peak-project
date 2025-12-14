# ODE System Usage Guide

## Table of Contents
1. [Quick Start Example](#quick-start-example)
2. [Model Specification Workflow](#model-specification-workflow)
3. [Simulation Execution](#simulation-execution)
4. [Parameter Sampling](#parameter-sampling)
5. [Drug Integration](#drug-integration)
6. [Configuration Management](#configuration-management)
7. [Advanced Patterns](#advanced-patterns)
8. [Troubleshooting](#troubleshooting)

## Quick Start Example

### Complete Workflow Based on `extended-simulation-1.py`

```python
import dotenv
import numpy as np
import pandas as pd
from tqdm import tqdm
from models.utils.config_manager import load_configs, save_data, initialise_config
from models.Specs.ModelSpec4 import ModelSpec4
from models.Specs.Drug import Drug
from models.Solver.RoadrunnerSolver import RoadrunnerSolver

# 1. Environment Setup
dotenv.load_dotenv()
initialise_config(folder_name="extended-simulation-1", verbose=1)

# 2. Load Configuration
loaded_config = load_configs(folder_name="extended-simulation-1", config_suffix="CuratedModel1")
notebook_config = loaded_config["notebook"]
exp_config = loaded_config["exp"]
spec_config = exp_config['spec']

# 3. Model Specification
new_spec = ModelSpec4(num_intermediate_layers=spec_config['n_layers'])
new_drug = Drug(
    name=spec_config['drug']['name'],
    start_time=spec_config['drug']['start'],
    default_value=spec_config['drug']['dose']
)

# 4. Generate Network
new_spec.generate_specifications(
    num_cascades=spec_config["n_cascades"],
    num_regulations=spec_config["n_regs"],
    random_seed=spec_config["gen_seed"]
)

# 5. Parameter Sampling
param_gen_config = exp_config['parameter_generation']
seeds = np.random.default_rng(
    exp_config["parameter_sampling"]["sampling_seed"]
).integers(0, 1000000, size=exp_config["parameter_sampling"]["num_models"])

builder_models = []
for seed in seeds:
    builder = new_spec.generate_network(
        notebook_config['name'],
        param_gen_config['ic_range'],
        param_gen_config['param_range'],
        param_gen_config['param_mul_range'],
        seed,
        receptor_basal_activation=spec_config["basal_activation"]
    )
    builder_models.append(builder)

# 6. Simulation Execution
solver_models = []
sim_results = []
sim_config = exp_config["simulation"]

for builder in tqdm(builder_models, desc="Creating solvers"):
    solver = RoadrunnerSolver()
    solver.compile(builder.get_sbml_model())
    solver_models.append(solver)

for solver in tqdm(solver_models, desc="Simulating models"):
    res = solver.simulate(sim_config["start"], sim_config["stop"], sim_config["step"])
    sim_results.append(res)

# 7. Data Management
sim_results_df = pd.concat(
    [pd.DataFrame(res).assign(seed=seed) for res, seed in zip(sim_results, seeds)],
    ignore_index=True,
)

save_data(notebook_config, sim_results_df, "sim_results_df", "pkl", verbose=1)
save_data(notebook_config, sim_results, "sim_results", "pkl", verbose=1)
```

## Model Specification Workflow

### Basic Model Creation

```python
from models.Specs.ModelSpec4 import ModelSpec4

# Create a model with 2 intermediate layers
spec = ModelSpec4(num_intermediate_layers=2)

# Generate basic network topology
spec.generate_specifications(
    num_cascades=3,       # Number of parallel pathways
    num_regulations=5,    # Additional feedback regulations
    random_seed=42        # For reproducibility
)
```

### Custom Regulations

```python
# Add specific regulations before network generation
spec.add_regulation("R1", "I1_2", "up")    # R1 up-regulates I1_2
spec.add_regulation("I1_1", "R2", "down")  # I1_1 down-regulates R2

# Generate network with custom regulations
spec.generate_specifications(3, 3, 42)
```

## Simulation Execution

### Single Model Simulation

```python
from models.Solver.RoadrunnerSolver import RoadrunnerSolver
import pandas as pd

# Generate and compile model
builder = spec.generate_network(
    "test_model",
    [50, 200],    # Initial condition range
    [0.5, 2.0],   # Parameter scale range
    [0.8, 1.2],   # Parameter multiplier range
    seed=42
)

# Create solver and simulate
solver = RoadrunnerSolver()
solver.compile(builder.get_sbml_model())

# Run simulation
result = solver.simulate(
    start=0,     # Start time
    stop=100,    # End time
    step=1       # Time step
)

print(f"Simulated {len(result)} time points")
```

### Batch Simulation

```python
# Generate multiple models with different parameters
models = []
for seed in range(10):
    builder = spec.generate_network(
        f"model_{seed}",
        [50, 200],
        [0.5, 2.0],
        [0.8, 1.2],
        seed=seed
    )
    models.append(builder)

# Parallel simulation (conceptual)
results = []
for builder in models:
    solver = RoadrunnerSolver()
    solver.compile(builder.get_sbml_model())
    result = solver.simulate(0, 100, 1)
    results.append(result)
```

## Parameter Sampling

### Controlled Parameter Generation

```python
import numpy as np

# Create reproducible parameter sets
rng = np.random.default_rng(42)  # Fixed seed for reproducibility
num_models = 100
model_seeds = rng.integers(0, 1000000, num_models)

# Generate models with different parameters
for seed in model_seeds:
    builder = spec.generate_network(
        f"model_seed_{seed}",
        specie_range=[50, 200],      # Initial concentration range
        param_range=[0.5, 2.0],      # Parameter scaling range
        param_mul_range=[0.8, 1.2], # Parameter multiplier range
        random_seed=seed
    )
```

## Drug Integration

### Single Drug Application

```python
from models.Specs.Drug import Drug

# Create drug specification
drug = Drug(
    name="InhibitorX",
    start_time=20,      # Apply at time 20
    default_value=1.0   # Concentration
)

# Add specific regulations
drug.add_regulation("R1", "down")  # Inhibit receptor 1
drug.add_regulation("I1_1", "up")  # Activate intermediate 1_1

# Add to model
spec.add_drug(drug)
```

### Complex Drug Scenarios

```python
# Multiple drugs with different timing
drug1 = Drug("EarlyInhibitor", start_time=10, default_value=0.5)
drug1.add_regulation("R1", "down")

drug2 = Drug("LateActivator", start_time=50, default_value=2.0)
drug2.add_regulation("O", "up")

spec.add_drug(drug1)
spec.add_drug(drug2)
```

### Global Drug Effects

```python
# Drug that targets all receptors
drug = Drug("BroadSpectrum", start_time=15, default_value=1.0)

if target_all_receptors:
    for i in range(spec.n_cascades):
        target = f'R{i+1}'
        drug.add_regulation(target, 'down')
else:
    # Specific targets
    drug.add_regulation("R1", "down")
    drug.add_regulation("R3", "up")
```

## Configuration Management

### YAML Configuration Structure

```yaml
notebook:
  name: "extended-simulation-1"
  version: "CuratedModel1"

exp:
  spec:
    n_layers: 2
    n_cascades: 3
    n_regs: 4
    gen_seed: 42
    basal_activation: true
    
    drug:
      name: "TestDrug"
      start: 20
      dose: 1.0
      target_all: false
      regulations:
        - ["R1", "down"]
        - ["I1_1", "up"]

  parameter_sampling:
    sampling_seed: 12345
    num_models: 100

  parameter_generation:
    ic_range: [50, 200]
    param_range: [0.5, 2.0]
    param_mul_range: [0.8, 1.2]

  simulation:
    start: 0
    stop: 100
    step: 1
```

### Loading and Using Configurations

```python
from models.utils.config_manager import load_configs, print_config

# Load configuration
config = load_configs("experiment-name", "config-version")
print_config(config)  # Display configuration

# Extract specific sections
notebook_cfg = config["notebook"]
spec_cfg = config["exp"]["spec"]
param_cfg = config["exp"]["parameter_generation"]
```

## Advanced Patterns

### Custom Reaction Types

```python
# Extend existing archtype patterns
from models.ArchtypeCollections import create_archtype_michaelis_menten_v2

# Create customized reaction type
custom_archtype = create_archtype_michaelis_menten_v2(
    stimulators=2,          # 2 stimulators
    stimulator_weak=1,      # 1 weak stimulator
    allosteric_inhibitors=1, # 1 allosteric inhibitor
    competitive_inhibitors=0
)
```

### Dynamic Parameter Modification

```python
# Modify parameters after model compilation
solver = RoadrunnerSolver()
solver.compile(builder.get_sbml_model())

# Change specific parameter values
parameter_changes = {
    "k_R1_forward": 0.5,    # Reduce forward rate for R1
    "k_I1_1_reverse": 2.0   # Increase reverse rate for I1_1
}
solver.set_parameter_values(parameter_changes)

# Run simulation with modified parameters
result = solver.simulate(0, 100, 1)
```

## Troubleshooting

### Common Issues and Solutions

#### 1. "Drug model not compatible" Error
```python
# Ensure target species exist in the model
all_species = spec.get_all_species(include_drugs=False)
print("Available species:", all_species)

# Only regulate existing species
if "TargetSpecies" in all_species:
    drug.add_regulation("TargetSpecies", "down")
```

#### 2. Simulation Convergence Issues
```python
# Adjust simulation parameters
result = solver.simulate(
    start=0,
    stop=100,
    step=0.1  # Smaller time step for stability
)

# Check for numerical stability
print("Max concentration:", result.max().max())
```

#### 3. Parameter Range Problems
```python
# Use biologically meaningful ranges
builder = spec.generate_network(
    "stable_model",
    specie_range=[10, 500],      # Reasonable concentrations
    param_range=[0.1, 10.0],     # Reasonable kinetic rates
    param_mul_range=[0.5, 2.0],  # Moderate variation
    random_seed=42
)
```

#### 4. Model Validation
```python
# Check model structure before simulation
print("Model species:", builder.get_species_names())
print("Model reactions:", len(builder.reactions))

# Verify SBML compilation
sbml_model = builder.get_sbml_model()
print("SBML model generated successfully")
```

### Debugging Tips

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check intermediate steps
print("Network topology:")
print("Receptors:", spec.receptors)
print("Intermediates:", spec.intermediate_layers)
print("Outcomes:", spec.outcomes)

# Validate drug regulations
for drug in spec.drugs:
    print(f"Drug {drug.name} regulates:", drug.regulation)
```

## Best Practices

1. **Version Control**: Always use specific configuration versions
2. **Reproducibility**: Fix random seeds for consistent results
3. **Parameter Ranges**: Use biologically meaningful value ranges
4. **Model Validation**: Check model structure before long simulations
5. **Progressive Testing**: Start with small networks and scale up
6. **Documentation**: Record all configuration parameters for reproducibility
