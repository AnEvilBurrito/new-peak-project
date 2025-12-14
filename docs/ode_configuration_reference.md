# ODE Configuration Reference

## Introduction

This document provides comprehensive reference for the YAML configuration files used to define ODE modeling experiments. These configurations enable reproducible research by capturing all experimental parameters in a structured format.

## Configuration File Structure

### Complete Configuration Example

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
      name: "EGFR_Inhibitor"
      start: 20
      dose: 1.0
      target_all: false
      regulations:
        - ["R1", "down"]
        - ["I1_1", "up"]
    
    custom_regulations:
      - ["R1", "I1_2", "up"]
      - ["I1_1", "R2", "down"]
      
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

## Configuration Sections

### 1. Notebook Configuration (`notebook`)

Defines experiment identification and versioning:

```yaml
notebook:
  name: "experiment-identifier"  # Unique experiment name
  version: "config-version"      # Configuration version
```

**Parameters:**
- **`name`** (string): Unique identifier for the experiment folder
- **`version`** (string): Version suffix for configuration files (e.g., "v1", "CuratedModel1")

### 2. Experiment Specification (`exp.spec`)

Defines the network architecture and topology:

```yaml
exp:
  spec:
    n_layers: 2               # Number of intermediate layers
    n_cascades: 3             # Number of parallel pathways
    n_regs: 5                 # Number of feedback regulations
    gen_seed: 42              # Random seed for network generation
    basal_activation: true    # Receptor basal activation
```

#### Network Topology Parameters

**`n_layers`** (integer, required)
- Number of intermediate processing layers
- Range: 1-5 (biologically meaningful)
- Example: `n_layers: 2` creates I1 and I2 layers

**`n_cascades`** (integer, required)  
- Number of parallel signalling pathways
- Range: 1-10 (computationally feasible)
- Example: `n_cascades: 3` creates R1, R2, R3 receptors

**`n_regs`** (integer, required)
- Number of additional feedback regulations
- Range: 0-20 (complexity control)
- Example: `n_regs: 5` adds 5 cross-regulations

**`gen_seed`** (integer, optional)
- Random seed for reproducible network generation
- Default: None (non-deterministic)
- Example: `gen_seed: 42` ensures identical networks

**`basal_activation`** (boolean, optional)
- Whether receptors have constitutive activation
- Default: true (biologically realistic)
- Example: `basal_activation: false` for no basal activity

### 3. Drug Configuration (`exp.spec.drug`)

Defines drug properties and target interactions:

```yaml
drug:
  name: "DrugName"          # Drug identifier
  start: 20                 # Application time
  dose: 1.0                 # Concentration
  target_all: false         # Global targeting
  regulations:              # Specific regulations
    - ["R1", "down"]
    - ["I1_1", "up"]
```

#### Drug Parameters

**`name`** (string, required)
- Unique identifier for the drug
- Example: `"EGFR_Inhibitor"`, `"p38_Activator"`

**`start`** (integer, required)
- Time when drug is applied (simulation time units)
- Example: `start: 20` applies drug at time 20

**`dose`** (float, required)
- Drug concentration or effect strength
- Example: `dose: 1.0` for unit concentration

**`target_all`** (boolean, optional)
- Whether drug targets all receptors globally
- Default: false (specific targeting)
- Example: `target_all: true` for broad-spectrum drugs

**`regulations`** (list of lists, conditional)
- Specific target regulations (required if `target_all: false`)
- Format: `[["target_species", "regulation_type"], ...]`
- Regulation types: "up" (activation), "down" (inhibition)
- Example: `[["R1", "down"], ["I1_1", "up"]]`

### 4. Custom Regulations (`exp.spec.custom_regulations`)

Additional specific regulations beyond auto-generated ones:

```yaml
custom_regulations:
  - ["R1", "I1_2", "up"]     # R1 up-regulates I1_2
  - ["I1_1", "R2", "down"]   # I1_1 down-regulates R2
```

**Parameters:**
- **List of [from, to, type]** tuples
- **from**: Regulating species (must exist in model)
- **to**: Target species (must exist in model)  
- **type**: "up" or "down" regulation

### 5. Parameter Sampling (`exp.parameter_sampling`)

Controls Monte Carlo parameter sampling:

```yaml
parameter_sampling:
  sampling_seed: 12345     # Seed for parameter generation
  num_models: 100          # Number of parameter sets
```

**Parameters:**

**`sampling_seed`** (integer, required)
- Random seed for reproducible parameter sampling
- Example: `sampling_seed: 12345`

**`num_models`** (integer, required)
- Number of different parameter sets to generate
- Range: 1-1000 (computational consideration)
- Example: `num_models: 100` generates 100 models

### 6. Parameter Generation (`exp.parameter_generation`)

Defines parameter value ranges:

```yaml
parameter_generation:
  ic_range: [50, 200]        # Initial concentration range
  param_range: [0.5, 2.0]    # Parameter scaling range
  param_mul_range: [0.8, 1.2] # Parameter multiplier range
```

#### Parameter Range Definitions

**`ic_range`** (list of 2 floats, required)
- Range for initial species concentrations
- Format: `[min, max]` (molecule counts)
- Example: `[50, 200]` for 50-200 initial molecules

**`param_range`** (list of 2 floats, required)
- Scaling range for base kinetic parameters
- Applied multiplicatively to assumed parameter values
- Example: `[0.5, 2.0]` for 0.5x to 2x baseline rates

**`param_mul_range`** (list of 2 floats, required)
- Additional multiplier range for parameter variation
- Provides fine-grained control over parameter diversity
- Example: `[0.8, 1.2]` for 80%-120% additional variation

### 7. Simulation Configuration (`exp.simulation`)

Defines simulation time course parameters:

```yaml
simulation:
  start: 0      # Simulation start time
  stop: 100     # Simulation end time
  step: 1       # Time step size
```

**Parameters:**

**`start`** (float, required)
- Simulation starting time (typically 0)
- Example: `start: 0`

**`stop`** (float, required)
- Simulation ending time
- Should be sufficient to observe dynamics
- Example: `stop: 100`, `stop: 500`

**`step`** (float, required)
- Time step for numerical integration
- Smaller for accuracy, larger for speed
- Example: `step: 0.1`, `step: 1.0`

## Advanced Configuration Patterns

### Multiple Drug Configurations

```yaml
# For complex multi-drug scenarios
drugs:
  - name: "EarlyInhibitor"
    start: 10
    dose: 0.5
    regulations:
      - ["R1", "down"]
      
  - name: "LateActivator"  
    start: 50
    dose: 2.0
    regulations:
      - ["O", "up"]
```

### Complex Regulation Networks

```yaml
custom_regulations:
  # Cross-pathway regulations
  - ["R1", "I1_2", "up"]     # Pathway 1→2 activation
  - ["R2", "I1_1", "down"]   # Pathway 2→1 inhibition
  
  # Feedback loops  
  - ["I2_1", "R1", "up"]     # Positive feedback
  - ["I2_2", "I1_2", "down"] # Negative feedback
  
  # Lateral connections
  - ["I1_1", "I1_2", "down"] # Competitive inhibition
```

### Parameter Sensitivity Studies

```yaml
# For systematic parameter exploration
parameter_sampling:
  sampling_seed: 42
  num_models: 500            # Large sample for sensitivity

parameter_generation:
  ic_range: [10, 500]        # Wide concentration range
  param_range: [0.1, 10.0]   # Broad kinetic range
  param_mul_range: [0.5, 2.0] # High parameter diversity
```

## Configuration Validation

### Required Fields Checklist

Ensure your configuration includes:

- [ ] `notebook.name` - Experiment identifier
- [ ] `notebook.version` - Configuration version  
- [ ] `exp.spec.n_layers` - Network depth
- [ ] `exp.spec.n_cascades` - Pathway count
- [ ] `exp.spec.n_regs` - Regulation complexity
- [ ] `exp.parameter_sampling.num_models` - Sample size
- [ ] `exp.parameter_generation.ic_range` - Concentration bounds
- [ ] `exp.parameter_generation.param_range` - Parameter scaling
- [ ] `exp.simulation` - Time course parameters

### Value Range Guidelines

**Biologically Meaningful Ranges:**

- **Initial concentrations**: `[10, 500]` molecules
- **Kinetic parameters**: `[0.1, 10.0]` relative to baseline
- **Time parameters**: Start typically 0, stop sufficient for dynamics
- **Drug doses**: `[0.01, 10.0]` relative effect strengths

**Computational Considerations:**

- **Network size**: `n_cascades ≤ 10`, `n_layers ≤ 5` for performance
- **Sample size**: `num_models ≤ 1000` for reasonable computation time
- **Time steps**: `step ≥ 0.01` for numerical stability

## Configuration File Naming

### Standard Naming Convention

```
config_{suffix}.yml
```

Where `{suffix}` corresponds to `notebook.version`:

- Example: `config_CuratedModel1.yml` for version "CuratedModel1"
- Example: `config_v1.yml` for version "v1"

### File Location

Configuration files are stored in:
```
{NEW_DATA_PATH}/{notebook.name}/config_{version}.yml
```

## Integration with Code

### Loading Configuration

```python
from models.utils.config_manager import load_configs

config = load_configs(
    folder_name="extended-simulation-1",
    config_suffix="CuratedModel1"  # Matches notebook.version
)

# Access specific sections
spec_config = config["exp"]["spec"]
param_config = config["exp"]["parameter_generation"]
```

### Using Configuration Values

```python
# Network generation
spec.generate_specifications(
    num_cascades=spec_config["n_cascades"],
    num_regulations=spec_config["n_regs"],
    random_seed=spec_config.get("gen_seed")
)

# Parameter sampling
builder = spec.generate_network(
    config_name=config["notebook"]["name"],
    specie_range=param_config["ic_range"],
    param_range=param_config["param_range"],
    param_mul_range=param_config["param_mul_range"]
)
```

## Best Practices

### 1. Version Control
- Always use descriptive version names
- Record configuration changes in version history
- Maintain backward compatibility when possible

### 2. Reproducibility
- Use fixed random seeds for deterministic results
- Document all parameter choices
- Include configuration files with results

### 3. Modular Design
- Create configurations for specific research questions
- Use inheritance for related experiments
- Maintain separate configurations for different scenarios

### 4. Documentation
- Include comments in YAML files
- Document biological rationale for parameter choices
- Record computational constraints and considerations
