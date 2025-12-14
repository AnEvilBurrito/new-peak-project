# ODE Network Architecture: D → R → Intermediate → O

## Introduction

The core architectural pattern of the ODE modelling system is the **Drug → Receptor → Intermediate → Outcome** (D→R→I→O) paradigm. This biologically-inspired structure provides a flexible framework for modeling cell signalling pathways with realistic constraints and meaningful biological interpretations.

## Core Architectural Pattern

### D → R → Intermediate → O Hierarchy

```
Drugs (D) → Receptors (R1, R2, ... Rn) → Intermediate Layers (I1_1, I2_1, ...) → Outcome (O)
     ↓              ↓                            ↓
External      Signal Reception        Signal Processing      Cellular Decision
Stimuli                                 & Integration
```

### Species Naming Convention

- **D**: Drug species (external stimuli) - e.g., `Drug1`, `InhibitorX`
- **R**: Receptor species (signal entry points) - e.g., `R1`, `R2`, `R3`
- **I**: Intermediate species (processing layers) - e.g., `I1_1`, `I1_2`, `I2_1`, `I2_2`
- **O**: Outcome species (final decisions) - e.g., `O`, `Response`, `Decision`

## Architecture Components

### 1. Drug Layer (D)

Drugs represent external stimuli that can regulate internal species:

```python
from models.Specs.Drug import Drug

# Single drug application
drug = Drug(
    name="EGF",           # Drug identifier
    start_time=10,        # Application time
    default_value=1.0,    # Concentration
    regulation=["R1"],    # Target species
    regulation_type=["up"] # Regulation type
)
```

**Key Features:**
- **Timing control**: Precise application schedules
- **Concentration effects**: Dose-response modeling  
- **Specificity**: Targeted regulation of species
- **Piecewise application**: Step-function concentration changes

### 2. Receptor Layer (R)

Receptors serve as signal entry points with biological constraints:

```python
# Auto-generated receptor species
receptors = [f'R{i+1}' for i in range(num_cascades)]
# Result: ['R1', 'R2', 'R3'] for 3 cascades
```

**Biological Properties:**
- **Basal activation**: Constitutive signalling capability
- **Drug regulation**: Primary targets for external stimuli
- **Forward signal propagation**: Connect to intermediate layers
- **Activation states**: Active (`Ra`) and inactive (`R`) forms

### 3. Intermediate Layers (I)

Multi-layered processing systems with configurable depth:

```python
# Configurable intermediate layers
intermediate_layers = [
    [f'I1_{i+1}' for i in range(num_cascades)],  # Layer 1
    [f'I2_{i+1}' for i in range(num_cascades)],  # Layer 2
    # ... additional layers
]
```

**Processing Capabilities:**
- **Signal integration**: Combine multiple input signals
- **Cross-talk**: Inter-layer communication
- **Feedback loops**: Regulatory connections within layers
- **Signal amplification**: Multi-stage processing

### 4. Outcome Layer (O)

Final signalling endpoints representing cellular decisions:

```python
# Single outcome species (typically)
outcomes = ['O']
```

**Decision Characteristics:**
- **Integration point**: Collects signals from all intermediates
- **Cellular response**: Represents final biological outcome
- **Quantifiable output**: Measurable endpoint for analysis

## Regulation Patterns

### Ordinary Regulations (Default Connectivity)

Automatically generated connections that form the backbone structure:

```python
# Receptor → First Intermediate
for receptor in receptors:
    reg = Regulation(receptor, f'I1_{index}', 'up')
    
# Intermediate Layer Connections  
for layer_index in range(num_layers - 1):
    for species_index in range(num_cascades):
        from_specie = f'I{layer_index+1}_{species_index+1}'
        to_specie = f'I{layer_index+2}_{species_index+1}'
        reg = Regulation(from_specie, to_specie, 'up')

# Final Intermediate → Outcome
for species in last_intermediate_layer:
    reg = Regulation(species, 'O', 'up')
```

### Feedback Regulations (Custom Connectivity)

Additional regulatory connections that introduce complexity:

```python
# During network generation
spec.generate_specifications(
    num_cascades=3,
    num_regulations=5,  # Number of feedback connections
    random_seed=42
)
```

**Feedback Types:**
- **Cross-pathway regulation**: R1 → I1_2 (inter-pathway)
- **Feedback loops**: I2_1 → R1 (closed loops)
- **Lateral inhibition**: I1_1 → I1_2 (competition)
- **Reinforcement**: I2_1 → I1_1 (positive feedback)

## Kinetic Architecture

### Reaction Generation Pattern

Each species has paired forward and reverse reactions:

```python
def add_reactions(specie, model, basal=False):
    # Forward reaction (activation)
    forward_reaction = get_forward_reaction(specie, basal=basal)
    
    # Reverse reaction (deactivation)  
    reverse_reaction = get_reverse_reaction(specie)
    
    model.add_reaction(reverse_reaction)
    model.add_reaction(forward_reaction)
```

### Activation States

Species exist in inactive and active forms:

```
R (inactive) ⇌ Ra (active)
I1_1 (inactive) ⇌ I1_1a (active)
O (inactive) ⇌ Oa (active)
```

### Rate Law Generation

Automatic rate law selection based on regulation patterns:

```python
def generate_forward_archtype_and_regulators(specie):
    # Count regulatory inputs
    regulators = get_regulators_for_specie(specie)
    total_up = count_up_regulations(regulators)
    total_down = count_down_regulations(regulators)
    
    # Select appropriate rate law
    rate_law = create_archtype_michaelis_menten_v2(
        stimulators=0,
        stimulator_weak=total_up,
        allosteric_inhibitors=total_down,
        competitive_inhibitors=0
    )
    return rate_law, regulators
```

## Network Generation Algorithm

### Step 1: Species Creation

```python
# Based on configuration parameters
receptors = [f'R{i+1}' for i in range(num_cascades)]
intermediate_layers = [
    [f'I{layer+1}_{cascade+1}' for cascade in range(num_cascades)]
    for layer in range(num_intermediate_layers)
]
outcomes = ['O']
```

### Step 2: Ordinary Regulation Generation

```python
# Vertical connections (flow through layers)
for i, receptor in enumerate(receptors):
    add_regulation(receptor, intermediate_layers[0][i], 'up')

for layer_idx in range(len(intermediate_layers) - 1):
    for cascade_idx in range(num_cascades):
        from_specie = intermediate_layers[layer_idx][cascade_idx]
        to_specie = intermediate_layers[layer_idx+1][cascade_idx]
        add_regulation(from_specie, to_specie, 'up')

for intermediate in intermediate_layers[-1]:
    add_regulation(intermediate, outcomes[0], 'up')
```

### Step 3: Feedback Regulation Generation

```python
# Random feedback connections with constraints
for _ in range(num_regulations):
    specie1 = random.choice(all_species_except_outcomes)
    specie2 = random.choice([s for s in all_species_except_outcomes if s != specie1])
    
    # Avoid duplicate regulations and self-regulation
    while regulation_exists(specie1, specie2) or specie1 == specie2:
        specie1, specie2 = select_new_pair()
    
    reg_type = random.choice(['up', 'down'])
    add_regulation(specie1, specie2, reg_type)
```

## Biological Constraints

### Regulation Validation

```python
def validate_regulation(from_specie, to_specie, reg_type):
    # Outcome cannot regulate other species
    if from_specie.startswith('O'):
        raise ValueError("Outcome species cannot regulate other species")
    
    # Regulation type must be 'up' or 'down'
    if reg_type not in ['up', 'down']:
        raise ValueError("Regulation type must be 'up' or 'down'")
    
    # Species must exist in the model
    if to_specie not in all_species:
        raise ValueError(f"Target species {to_specie} not found")
```

### Network Topology Constraints

- **No self-regulation**: Species cannot regulate themselves
- **Outcome isolation**: Outcome species only receive regulations
- **Biological plausibility**: Prevents impossible connections
- **Parameter constraints**: Biologically meaningful value ranges

## Parameter Sampling Strategy

### Biologically Informed Ranges

```python
def generate_biologically_reasonable_parameters():
    # Initial concentrations (molecule counts)
    ic_range = [50, 200]           # 50-200 molecules
    
    # Kinetic parameters (rate constants)
    param_range = [0.5, 2.0]       # 0.5x-2x baseline rates
    param_mul_range = [0.8, 1.2]   # Additional variation
    
    # Basal activation rates
    basal_activation = True         # Receptors have constitutive activity
```

### Reproducible Sampling

```python
# Controlled random number generation
rng = np.random.default_rng(seed=42)
parameters = rng.uniform(
    low=min_value * scale_range[0] * multiplier_range[0],
    high=max_value * scale_range[1] * multiplier_range[1]
)
```

## Integration with Solver System

### Model Compilation

```python
# Convert to standard formats
sbml_model = builder.get_sbml_model()    # Systems Biology Markup Language
antimony_model = builder.get_antimony_model()  # Human-readable format
```

### Simulation Execution

```python
# Roadrunner solver integration
solver = RoadrunnerSolver()
solver.compile(sbml_model)
results = solver.simulate(start=0, stop=100, step=1)
```

## Example Network Configurations

### Minimal Network (2 cascades, 1 layer)
```
Drug → R1 → I1_1 → O
     → R2 → I1_2 → O
```

### Complex Network (3 cascades, 3 layers)
```
Drug → R1 → I1_1 → I2_1 → I3_1 → O
     → R2 → I1_2 → I2_2 → I3_2 → O  
     → R3 → I1_3 → I2_3 → I3_3 → O
     
With cross-regulations: R1→I1_2, I2_1→R2, I3_1→I1_3
```

## Advantages of D→R→I→O Architecture

### Biological Relevance
- **Modular design**: Mirrors real signalling pathway organization
- **Constraint enforcement**: Prevents biologically impossible connections
- **Interpretable results**: Clear mapping to biological concepts

### Computational Efficiency
- **Structured generation**: Systematic network construction
- **Parameter optimization**: Meaningful parameter space exploration
- **Scalable complexity**: Easy adjustment of network size

### Research Flexibility
- **Hypothesis testing**: Easy modification of specific connections
- **Drug screening**: Systematic evaluation of intervention points  
- **Parameter sensitivity**: Controlled exploration of system behavior
