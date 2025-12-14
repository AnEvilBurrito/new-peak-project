# ODE Modelling System Documentation

Welcome to the comprehensive documentation for the ODE Modelling System used in the new-peak-project repository. This system provides a structured framework for investigating cell signalling pathways and decision-making processes using ordinary differential equations.

## Documentation Structure

### Core Documentation Files

1. **[ODE System Overview](ode_system_overview.md)** - High-level architecture and design principles
2. **[ODE Usage Guide](ode_usage_guide.md)** - Practical examples and step-by-step workflows
3. **[Network Architecture](ode_network_architecture.md)** - D→R→I→O paradigm details
4. **[Configuration Reference](ode_configuration_reference.md)** - YAML file specifications

### Related Documentation

- **[Data Loading System](../data_loading_system.md)** - Configuration and data management
- **[Usage Guide](../usage_guide.md)** - General system usage patterns
- **[API Reference](../api_reference.md)** - Technical function references

## Quick Start

### For New Users
1. Read the **[System Overview](ode_system_overview.md)** to understand the architecture
2. Follow the **[Usage Guide](ode_usage_guide.md)** for practical implementation
3. Consult the **[Configuration Reference](ode_configuration_reference.md)** for experiment setup

### Basic Workflow Pattern

```python
import dotenv
from models.utils.config_manager import load_configs
from models.Specs.ModelSpec4 import ModelSpec4
from models.Solver.RoadrunnerSolver import RoadrunnerSolver

# 1. Load configuration
config = load_configs("experiment-name", "config-version")

# 2. Generate model specification
spec = ModelSpec4(num_intermediate_layers=config["exp"]["spec"]["n_layers"])
spec.generate_specifications(
    num_cascades=config["exp"]["spec"]["n_cascades"],
    num_regulations=config["exp"]["spec"]["n_regs"]
)

# 3. Create and simulate model
builder = spec.generate_network(
    config["notebook"]["name"],
    config["exp"]["parameter_generation"]["ic_range"],
    config["exp"]["parameter_generation"]["param_range"]
)

solver = RoadrunnerSolver()
solver.compile(builder.get_sbml_model())
result = solver.simulate(
    config["exp"]["simulation"]["start"],
    config["exp"]["simulation"]["stop"],
    config["exp"]["simulation"]["step"]
)
```

## System Features

### Biological Modeling Capabilities
- **Initial cell expression** capture through configurable initial conditions
- **Protein dynamics** simulation using biologically meaningful kinetics
- **Mechanistic drug response** determination in virtual cellular systems
- **Multi-layered signalling** pathways with realistic constraints

### Key Architectural Patterns
- **D → R → Intermediate → O** structured hierarchy
- **Regulatory network generation** with biological constraints
- **Parameter sampling** with reproducible random seeds
- **SBML/Antimony model export** for standard compatibility

### Integration Ecosystem
- **Configuration-driven experiments** via YAML files
- **Seamless data management** integration
- **Notebook workflow support** for research pipelines
- **Multiple solver implementations** for flexibility

## Documentation Navigation

### Based on Your Needs

**Getting Started with Basic Experiments**
- Start with **[Usage Guide](ode_usage_guide.md)** Quick Start section
- Follow the complete workflow example
- Progress to custom model configurations

**Understanding System Architecture**
- Read **[System Overview](ode_system_overview.md)** for high-level concepts
- Dive into **[Network Architecture](ode_network_architecture.md)** for D→R→I→O details
- Explore component interactions and design principles

**Advanced Configuration and Customization**
- Use **[Configuration Reference](ode_configuration_reference.md)** for YAML specifications
- Implement complex drug scenarios and regulation patterns
- Set up parameter sensitivity studies

**Technical Implementation Details**
- Refer to **[API Reference](../api_reference.md)** for function signatures
- Understand solver integration and model compilation
- Explore parameter generation and sampling strategies

## Example Experiment Types

### Drug Response Prediction
Model how specific drugs affect signalling pathways with precise timing and concentration effects.

### Network Topology Exploration
Generate different network architectures to understand pathway organization principles.

### Parameter Sensitivity Analysis
Explore how kinetic parameter variations affect system behavior and robustness.

### Multi-Drug Combination Studies
Investigate synergistic or antagonistic effects of drug combinations.

## Integration Examples

### With Data Loading System
```python
from models.utils.config_manager import save_data, load_data

# Save simulation results
save_data(notebook_config, results, "simulation_results", "pkl")

# Load for analysis
loaded_results = load_data(notebook_config, "simulation_results", "pkl")
```

### With Visualization Tools
```python
from models.utils.plotting import plot_simulation_results
import matplotlib.pyplot as plt

fig = plot_simulation_results(results)
plt.show()
```

## Getting Help

### Common Starting Points
- Check the **[Troubleshooting](../usage_guide.md#troubleshooting)** section for common issues
- Review example configurations in existing experiment folders
- Examine notebook implementations in `src/notebooks/`

### Configuration Validation
- Use the checklist in **[Configuration Reference](ode_configuration_reference.md)**
- Ensure biologically meaningful parameter ranges
- Verify random seed settings for reproducibility

### Performance Considerations
- Start with small networks (`n_cascades ≤ 3`, `n_layers ≤ 2`)
- Use moderate parameter sampling (`num_models ≤ 100`)
- Consider computational constraints for large-scale studies

## Related Code Components

The ODE modelling system is implemented in:
- `src/models/Specs/ModelSpec4.py` - Core model specification
- `src/models/Specs/Drug.py` - Drug modeling functionality
- `src/models/ArchtypeCollections.py` - Reaction kinetics patterns
- `src/models/Solver/` - Simulation engine implementations
- `src/models/utils/config_manager.py` - Configuration integration

## Contributing to Documentation

If you find issues or have improvements:
1. Check existing documentation for coverage
2. Update relevant sections with clear examples
3. Ensure consistency with actual system behavior
4. Include practical use cases and troubleshooting tips

---

**Last Updated**: December 2025  
**System Version**: ModelSpec4-based implementation  
**Related Project**: new-peak-project cell signalling analysis
