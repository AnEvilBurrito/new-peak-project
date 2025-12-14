# Data Loading System Usage Guide

## Table of Contents
1. [Setup and Initialization](#setup-and-initialization)
2. [Working with Configurations](#working-with-configurations)
3. [Saving and Loading Data](#saving-and-loading-data)
4. [Managing Figures](#managing-figures)
5. [Common Patterns](#common-patterns)
6. [Troubleshooting](#troubleshooting)

## Setup and Initialization

### Environment Configuration
The system relies on a `.env` file containing:
```
NEW_DATA_PATH=C:/path/to/your/data
```

### Basic Setup in Notebooks

```python
# Import required modules
import dotenv
from models.utils.config_manager import load_configs, load_data, initialise_config, save_figure

# Load environment variables
dotenv.load_dotenv()  # Loads from .env file in project root

# Initialize a new experiment folder structure
initialise_config(
    folder_name='my-experiment',
    verbose=1  # Optional: print status messages
)

### Folder Structure Creation
When `initialise_config()` is called, it creates:
```
{NEW_DATA_PATH}/my-experiment/
├── data/      # For storing simulation results
└── figures/   # For saving plots
```

## Working with Configurations

### Loading Configuration Files

```python
# Load configuration from YAML file
config = load_configs(
    folder_name='extended-simulation-1',
    config_suffix='CuratedModel1'  # Without 'config_' prefix
)

# Access configuration sections
notebook_config = config.get('notebook', {})
model_config = config.get('model', {})

print(f"Notebook configuration: {notebook_config}")
print(f"Model configuration: {model_config}")
```

### Configuration Versioning
Configurations are versioned to track changes:

```python
# Version is specified in config_suffix or defaults to 'v1'
config_v1 = load_configs('my-experiment', 'v1')
config_v2 = load_configs('my-experiment', 'v2')
```

## Saving and Loading Data

### Saving Simulation Results

```python
from models.utils.config_manager import save_data

# Save simulation results
save_data(
    notebook_config=notebook_config,
    data=simulation_results,
    data_name='sim_results',
    data_format='pkl',  # Options: 'pkl', 'csv', 'txt'
    verbose=1  # Optional
)
```

### Loading Saved Data

```python
from models.utils.config_manager import load_data

# Load previously saved data
loaded_results = load_data(
    notebook_config=notebook_config,
    data_name='sim_results',
    data_format='pkl'
)
```

### Format-Specific Examples

#### Pickle Format (Complex Objects)
```python
# Save model objects, simulation results
save_data(notebook_config, complex_object, 'model_output', 'pkl')
```

#### CSV Format (Tabular Data)
```python
import pandas as pd

# Create a DataFrame
results_df = pd.DataFrame({
    'time': simulation_time,
    'value': simulation_values
})

save_data(notebook_config, results_df, 'sim_results_df', 'csv')
```

## Managing Figures

### Saving Plots

```python
import matplotlib.pyplot as plt

# Create a plot
fig, ax = plt.subplots()
ax.plot(time, values)
ax.set_xlabel('Time')
ax.set_ylabel('Value')

save_figure(
    notebook_config=notebook_config,
    fig=fig,
    fig_name='simulation_plot',
    fig_format='png',  # Options: 'png', 'pdf', 'svg'
    dpi=300  # Additional savefig parameters
)
```

### Clearing Data and Figures

```python
from models.utils.config_manager import clear_data_and_figure

# Clear all data and figures for a specific version
clear_data_and_figure(
    notebook_config=notebook_config,
    data=True,     # Clear data files
    figure=True,   # Clear figure files
    verbose=1
)
```

## Common Patterns

### Complete Experiment Workflow

```python
import dotenv
import matplotlib.pyplot as plt
from models.utils.config_manager import (
    initialise_config,
    load_configs,
    save_data,
    load_data,
    save_figure
)

# 1. Initialize experiment
initialise_config('new-experiment')

# 2. Load experiment configuration
config = load_configs('new-experiment', 'v1')

# 3. Run simulations
simulation_results = run_simulation(config['model']))

# 4. Save results
save_data(
    notebook_config=config['notebook'],
    data=simulation_results,
    data_name='initial_sim',
    data_format='pkl'
)

# 5. Create and save visualization
fig = create_plot(simulation_results))
save_figure(
    notebook_config=config['notebook'],
    fig=fig,
    fig_name='initial_results',
    fig_format='png',
    dpi=300
)

# 6. Later: Reload data for analysis
reloaded_data = load_data(
    notebook_config=config['notebook'],
    data_name='initial_sim',
    data_format='pkl'
)
```

### Working with Multiple Configurations

```python
# Compare different model configurations
config_base = load_configs('experiment', 'base_config')
config_modified = load_configs('experiment', 'modified_config')
```

## Troubleshooting

### Common Issues

#### 1. "FileNotFoundError: Data file not found"
- Check that the configuration version matches the data file name
- Verify `NEW_DATA_PATH` is correctly set in `.env`

#### 2. "ValueError: Unsupported data format"
- Ensure `data_format` is one of: 'pkl', 'csv', 'txt'

#### 3. Path Configuration Errors
```python
# Verify environment setup
import os
from dotenv import load_dotenv

load_dotenv()
new_path = os.getenv("NEW_DATA_PATH"))
if not new_path:
    print("ERROR: NEW_DATA_PATH not set in .env file")
```

### Debugging Tips

```python
# Enable verbose mode for debugging
initialise_config('test', verbose=2)
save_data(notebook_config, data, 'test_data', 'pkl', verbose=2)
```

## Best Practices

1. **Version everything**: Always use version suffixes in configuration files
2. **Consistent naming**: Use the same naming conventions across experiments
3. **Document configurations**: Include comments in YAML files
4. **Test data loading**: Verify you can load data after saving it
5. **Environment isolation**: Use separate paths for dev, test, and prod

### Example Configuration YAML

```yaml
notebook:
  name: extended-simulation-1
  version: CuratedModel1
  
model:
  name: TwoStepDrugModel
  parameters:
    k1: 0.1
    k2: 0.2
  
simulation:
  time_points: [0, 1, 2, 3, 4, 5)
  
  initial_conditions:
    s0: 100
    s1: 0
    s2: 0
```

### Integration with Other Systems

```python
# Example: Loading data for optimization
from src.optimisation.DataHelper import DataHelper

# Create data helper
data_helper = DataHelper()

# Add experimental data
data_helper.add_exp_data(
    time=[0, 1, 2, 3, 4, 5],
]
```

### DataHelper Integration

```python
from src.optimisation.DataHelper import DataHelper

# Initialize helper
helper = DataHelper()

# Add experimental data
helper.add_exp_data(
    time=exp_time,
    value=exp_values,
    std_error=std_errors,
    name='measurement_1'
)

# Get data for optimization
exp_std, exp_value, sim_value = helper.get_data_for_optimisation()
```

### ExpDataLoader Integration

```python
from src.optimisation.ExpDataLoader import ExpDataLoader

# Load experimental data from CSV
loader = ExpDataLoader()
loader.load_data('experimental_data.csv')

# Get average for a state
state_average = loader.get_state_average_for_each_time_point('pERK'))
