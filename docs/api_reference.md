# API Reference

## config_manager.py

### Core Functions

#### `load_configs(folder_name: str, config_suffix: str = "v1") -> dict`
Loads configuration from YAML file.

**Parameters:**
- `folder_name`: Name of the experiment folder
- `config_suffix`: Suffix for configuration file (default: "v1")

**Returns:**
- Configuration dictionary containing notebook and model settings

**Example:**
```python
config = load_configs('extended-simulation-1', 'CuratedModel1')```

#### `save_configs(folder_name: str, config: dict, config_suffix: str = "v1") -> None`
Saves configuration to YAML file.

**Parameters:**
- `folder_name`: Name of the experiment folder
- `config`: Configuration dictionary to save
- `config_suffix`: Suffix for configuration file

#### `initialise_config(folder_name: str, verbose: int = 0) -> None`
Creates folder structure for an experiment.

**Parameters:**
- `folder_name`: Name of the experiment folder
- `verbose`: Verbosity level (0-2)

#### `save_figure(notebook_config: dict, fig: plt.Figure, fig_name: str, fig_format: str = "png", verbose: int = 0, **kwargs) -> None`
Saves matplotlib figures to the standardized figures folder.

**Parameters:**
- `notebook_config`: Notebook configuration dictionary
- `fig`: Matplotlib figure object
- `fig_name`: Name for the figure file (without extension)
- `fig_format`: Format to save figure in (default: "png")
- `verbose`: Verbosity level
- `**kwargs`: Additional keyword arguments for `fig.savefig()`

**Example:**
```python
save_figure(
    notebook_config=config["notebook"],
    fig=my_plot,
    fig_name="simulation_results"
)
```

### Data Management Functions

#### `save_data(notebook_config: dict, data: any, data_name: str, data_format: str = 'pkl', verbose: int = 0, **kwargs) -> None`
Saves data in various formats.

**Parameters:**
- `notebook_config`: Notebook configuration dictionary
- `data`: Data to save (can be any Python object)
- `data_name`: Name for the data file (without extension)
- `data_format`: Format to save data in ('pkl', 'csv', 'txt')
- `verbose`: Verbosity level
- `**kwargs`: Format-specific keyword arguments

**Supported Formats:**
- `'pkl'`: Pickle format for complex objects
(" Writer uses Pickle serialization ))))
- `'csv'`: CSV format for tabular data (requires pandas DataFrame)
- `'txt'`: Text format for string data

**Example:**
```python
save_data(
    notebook_config=config["notebook"],
    data=simulation_results,
    data_name='sim_results',
    data_format='pkl'
)
```

#### `load_data(notebook_config: dict, data_name: str, data_format: str = 'pkl', verbose: int = 0, **kwargs) -> any`
Loads previously saved data.

**Parameters:**
- `notebook_config`: Notebook configuration dictionary
- `data_name`: Name of the data file (without extension)
- `data_format`: Format of the data file

**Returns:**
- The loaded data

**Example:**
```python
sim_results = load_data(
    notebook_config=config["notebook"],
    data_name='sim_results',
    data_format='pkl'
)
```

#### `clear_data_and_figure(notebook_config: dict, data: bool = True, figure: bool = True, verbose: int = 0) -> None`
Clears all data and figures for a specific configuration version.

**Parameters:**
- `notebook_config`: Notebook configuration dictionary
- `data`: Whether to clear data files (default: True)
- `figure`: Whether to clear figure files (default: True)
- `verbose`: Verbosity level

### Utility Functions

#### `print_config(d: dict, indent: int = 0) -> None`
Prints configuration dictionary in a readable format.

**Parameters:**
- `d`: Configuration dictionary to print
- `indent`: Initial indentation level

## DataHelper.py

### DataHelper Class

#### `__init__()` 
Initializes empty data storage.

**Example:**
```python
data_helper = DataHelper()
```

#### `add_exp_data(time: Iterable, value: Iterable, std_error: Iterable, name: str) -> None`
Adds experimental data to the storage.

**Parameters:**
- `time`: Time points
- `value`: Experimental values
- `std_error`: Standard errors
- `name`: Unique name for this dataset

**Example:**
```python
data_helper.add_exp_data(
    time=time_points,
    value=measurements,
    std_error=std_errors,
    name='my_experiment'
)
```

#### `get_exp_state_names() -> List[str]`
Returns list of experimental data names.

#### `get_exp_data(name: str) -> Dict`
Retrieves specific experimental data.

**Parameters:**
- `name`: Name of the dataset to retrieve

#### `get_exp_data_time(name: str) -> Iterable`
Returns time points for specific experimental data.

#### `match_sim_data(result: dict) -> None`
Matches simulation data to experimental time points.

#### `get_data_for_optimisation() -> Tuple[np.ndarray, np.ndarray, np.ndarray]
Returns data formatted for optimization algorithms.

#### `get_data_tuple(name: str) -> Tuple[Iterable, Iterable, Iterable]`
Returns experimental data as a tuple.

#### `get_exp_data_time(name: str) -> Iterable`
Returns time points for specific experimental data.

**Parameters:**
- `name`: Name of the dataset to retrieve

**Returns:**
- Tuple containing (time, value, std_error)

## ExpDataLoader.py

### ExpDataLoader Class

#### `__init__()` 
Initializes empty data loader.

#### `load_data(filename: str, delimiter: str = ',') -> None`
Loads experimental data from CSV file.

**Expected CSV Format:**
```
experimental_state, replicate_number, time_point, value
EGFR, 1, 0, 100
EGFR, 1, 1, 85
EGFR, 2, 0, 98
```

### Data Access Methods

#### `get_data() -> pd.DataFrame`
Returns the loaded experimental data.

#### `get_experimental_states() -> List[str]`
Returns list of unique experimental states.

#### `get_state_average_for_each_time_point(state: str) -> pd.Series`
Returns average values for each time point of a specific state.

#### `get_state_std_for_each_time_point(state: str) -> pd.Series`
Returns standard deviation for each time point of a specific state.

#### `get_state_modified_time(state: str, stim_time: float) -> np.ndarray`
Returns modified time points for stimulation experiments.

#### `get_state_raw_data(state: str) -> pd.Series`
Returns raw data values for a specific state.

## Environment Variables

### Required Configuration

#### `NEW_DATA_PATH`
Path to the root data directory.

**Example .env file:**
```
NEW_DATA_PATH=C:/Users/l8105/Documents/research/data
```

### Path Resolution

Data is stored at: `{NEW_DATA_PATH}/{folder_name}/data/{config_version}_{data_name}.{format}`
Figures are stored at: `{NEW_DATA_PATH}/{folder_name}/figures/{config_version}_{fig_name}.{format}`
