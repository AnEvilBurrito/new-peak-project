# Data Loading System Documentation

Welcome to the documentation for the data loading system used in the new-peak-project repository. This system provides a structured approach to managing experimental configurations, simulation results, and processed data.

## Table of Contents

1. **[Architecture Overview](data_loading_system.md)**: In-depth explanation of the system design and components
2. **[Usage Guide](usage_guide.md)**: Practical examples and step-by-step instructions
3. **[API Reference](api_reference.md)**: Detailed function and class references

## Quick Start

### For New Users
1. Read the [Architecture Overview](data_loading_system.md) to understand the system design
2. Follow the [Usage Guide](usage_guide.md) for common tasks and workflows
3. Consult the [API Reference](api_reference.md) for detailed technical information

### Basic Usage Pattern

```python
import dotenv
from models.utils.config_manager import load_configs, load_data

# Load environment configuration
dotenv.load_dotenv()

# Load experiment configuration
config = load_configs(
    folder_name='extended-simulation-1',
    config_suffix='CuratedModel1'
)

# Load simulation data
sim_results = load_data(
    notebook_config=config["notebook"],
    data_name='sim_results',
    data_format='pkl'
)
```

## System Overview

The data loading system is built around three core principles:

1. **Configuration-driven**: All experiments are defined by YAML configuration files
2. **Version-controlled**: All configurations and data are versioned
3. **Environment-aware**: Works across development, testing, and production environments

## Key Features

- **Standardized folder structure** for all experiments
- **Multiple data formats** support (Pickle, CSV, Text)
- **Automatic figure management**
- **Experimental data wrappers** for optimization workflows

## Directory Structure

```
{NEW_DATA_PATH}/
├── {experiment_folder}/          # e.g., 'extended-simulation-1'
│   ├── config_{suffix}.yml     # Configuration files
│   ├── data/                      # Simulation results
│   └── figures/                   # Visualizations
```

## Getting Started

### 1. Set up your environment
Create a `.env` file with:
```
NEW_DATA_PATH=C:/path/to/your/data
```

## Documentation Files

### [data_loading_system.md](data_loading_system.md)
Comprehensive explanation of the system architecture, design principles, and core components.

### 2. Configure your experiment
Use YAML configuration files to define:
- Model parameters
- Simulation settings
- Notebook configurations

### [usage_guide.md](usage_guide.md)
Step-by-step instructions for:
- Setting up new experiments
- Saving and loading data
- Managing figures

### 3. Use the API
Detailed function signatures, parameters, and examples for:
- `config_manager.py` - Main configuration and data management
- `DataHelper.py` - Experimental data storage and retrieval
- `ExpDataLoader.py` - CSV data loading and processing

## Contributing to Documentation

If you find issues or have improvements for this documentation:
1. Review existing files
2. Update relevant sections
3. Ensure examples remain functional and current

## Related Code

The system implementation is located in:
- `src/models/utils/config_manager.py`
- `src/optimisation/DataHelper.py`
- `src/optimisation/ExpDataLoader.py`

## Support

For questions about using the data loading system, check:
- The example notebooks in `src/notebooks/`
- Existing experiment configurations
- Integration with optimization workflows

---

**Last Updated**: December 2025  
**Version**: 1.0.0
