# Data Loading System Architecture

## Overview
The data loading system provides a structured approach to managing experimental configurations, simulation results, and processed data. It uses YAML configuration files, environment variables, and a set of utility functions to standardize data access throughout the project.

## Core Components

### 1. Configuration Management (`config_manager.py`)
Located at `src/models/utils/config_manager.py`, this is the central module that handles:

- **Environment configuration**: Loads `NEW_DATA_PATH` from `.env` file using `dotenv`
- **YAML config loading**: `load_configs()` loads configuration files from `{NEW_DATA_PATH}/{folder_name}/config_{config_suffix}.yml`
- **Data persistence**: `save_data()` and `load_data()` functions for saving/loading data with versioning
- **File system organization**: Creates standardized folder structures with `data/` and `figures/` subdirectories
- **Version control**: Configurations are versioned (e.g., "v1", "CuratedModel1")

### 2. Key Functions

#### `load_data(notebook_config, data_name, data_format)`
```python
def load_data(notebook_config: dict, data_name: str, data_format: str = 'pkl') -> any
```
Loads data from `{NEW_DATA_PATH}/{folder_name}/data/{config_version}_{data_name}.{format}`
- Supports multiple formats: pickle (`.pkl`), CSV (`.csv`), and text (`.txt`)
- Uses environment-based paths for flexibility across development environments

#### `initialise_config(folder_name)`
Creates the required folder structure:
```
{NEW_DATA_PATH}/{folder_name}/
├── data/
│   └── {version}_{data_name}.pkl
└── figures/
    └── {version}_{fig_name}.png
```

#### `save_figure(notebook_config, fig, fig_name, fig_format)`
Saves matplotlib figures to the standardized figures folder.

### 3. File Structure
The system organizes data according to a standardized hierarchy:

```
{NEW_DATA_PATH}/
├── extended-simulation-1/          # Experiment folder
│   ├── config_CuratedModel1.yml     # YAML configuration
│   ├── data/
│   │   ├── v1_sim_results.pkl
│   │   └── v1_sim_results_df.pkl
└── other-experiment/                # Additional experiment folders
    ├── config_v1.yml
    └── data/
        └── v1_sim_results.pkl
```

### 4. Related Utilities

#### `DataHelper.py` (Wrapper for Experimental Data)
```python
class DataHelper:
    """
    A dictionary wrapper that stores experimental data with methods for 
    adding/retrieving experimental data with time series, values, and standard errors.
```

- Provides methods for:
  - `add_exp_data(time, value, std_error, name)`: Add experimental data
  - `get_exp_data(name)`: Retrieve experimental data
  - `match_sim_data(result)`: Match simulation data to experimental time points
  - `get_data_for_optimisation()`: Returns data formatted for optimization algorithms

#### `ExpDataLoader.py`
```python
class ExpDataLoader:
    """
    Loads and processes experimental data from CSV files.
    
    Expected CSV format:
    experimental_state, replicate_number, time_point, value
    ```
}

### 5. Environment Integration

- Uses `.env` file for path configuration (referenced in code, blocked by `.clineignore`)
- Configurations are experiment-specific and version-controlled
- Data files are named with version prefixes to prevent conflicts

### 6. Key Design Principles

1. **Separation of concerns**: 
   - Configuration loading vs. data loading
   - Experiment-specific configurations

2. **Environment awareness**: 
   - Works across dev/test/prod environments with path configuration
   - Paths configured via environment variables

3. **Version isolation**: 
   - Different configuration versions keep data separate
   - Prevents accidental data corruption or mixing

4. **Flexible formats**: 
   - Supports pickle for complex objects (simulation results, model objects)
   - CSV for tabular data and text for simple string data

### 7. Example Data Flow

From the notebook `visualise-extended-simulation-1.ipynb`:

```python
import dotenv
from models.utils.config_manager import load_configs, load_data

# Set up paths from .env
config = dotenv_values(".env")  # Typically handled via dotenv.load_dotenv()
# new_path = config["NEW_DATA_PATH"]

# Load configuration
loaded_config = load_configs(
    folder_name='extended-simulation-1',
    config_suffix='CuratedModel1'
)

# Load simulation data
sim_results = load_data(
    notebook_config=loaded_config["notebook"],
    data_name="sim_results",
    data_format="pkl"
)
```

### 8. System Advantages

- **Consistency**: All experiments follow the same structure
- **Reproducibility`: Versioned configurations and data files
- **Scalability**: Easy to add new experiments or versions
- **Maintainability**: Clear separation of configuration and data loading logic
