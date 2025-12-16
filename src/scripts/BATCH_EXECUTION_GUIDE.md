# Batch Execution System Guide

## Overview

This system provides batch execution scripts for iterative dataset generation with sequential re-assembly capabilities. The system allows you to run multi-day batch processes and later re-assemble all generated data into a single cohesive dataset.

## Core Components

### 1. Batch Framework (`batch_framework.py`)
The main framework that handles:
- Assembly ID generation and tracking
- Sequential re-assembly of multiple data files
- Storage abstraction (S3 or local)
- Progress tracking and resume capabilities

### 2. Specific Batch Scripts
- `4_01_v1_batch-parameter-distortion.py` - Parameter distortion experiments
- `4_02_v1_batch-expression-noise.py` - Expression noise experiments  
- `4_03_v1_batch-response-noise.py` - Response noise experiments

## Quick Start

### Running a Single Batch Process

```python
# Execute parameter distortion batch
python src/scripts/4_01_v1_batch-parameter-distortion.py

# Execute expression noise batch  
python src/scripts/4_02_v1_batch-expression-noise.py

# Execute response noise batch
python src/scripts/4_03_v1_batch-response-noise.py
```

### Sequential Re-assembly Example

```python
from scripts.batch_framework import create_batch_executor

# Re-assemble all parameter distortion results
executor = create_batch_executor(
    notebook_name='batch-parameter-distortion',
    exp_number='01',
    version_number='v1'
)

try:
    combined_data = executor.sequential_reassembly('parameter_distortion_results')
    print(f"Re-assembled {len(combined_data)} records")
except Exception as e:
    print(f"Re-assembly failed: {e}")
```

## Configuration Management

Each script uses a configuration system that can be customized:

### Default Configuration Structure
```python
config = {
    'spec': {
        'n_layers': 2,
        'n_cascades': 3,
        # ... model specifications
    },
    'parameter_generation': {
        'ic_range': [200, 1000],
        # ... parameter ranges
    },
    # ... other settings
}
```

### Customizing Configurations

Edit the `load_experiment_config()` function in each script to modify:
- Model architecture parameters
- Noise levels or distortion factors
- Sampling sizes and repetitions
- Simulation parameters

## Assembly ID System

### How Assembly IDs Work
- Each batch execution generates a unique Assembly ID
- Format: `YYYYMMDD_HHMMSS_runNNN` (e.g., `20251216_142230_run001`)
- Assembly IDs are tracked in `assembly_list.csv`
- Files are named with Assembly ID suffixes for identification

### Assembly List Management
The assembly list tracks:
- Assembly ID
- Timestamp
- Status (started/completed)

### Checking Assembly Status
```python
executor = create_batch_executor('batch-parameter-distortion', '01', 'v1')
status = executor.get_assembly_status('20251216_142230_run001')
print(f"Status: {status}")  # 'started', 'completed', or 'not_found'
```

## Storage Backends

### S3 Storage (Default)
- Uses the project's S3ConfigManager
- Files stored at: `{SAVE_RESULT_PATH}/{exp-number}_{version-number}_{experiment-title}/`
- Requires proper S3 environment configuration

### Local Storage
```python
# Use local storage instead of S3
executor = create_batch_executor(
    notebook_name='batch-parameter-distortion',
    exp_number='01',
    version_number='v1',
    storage_backend='local'  # Use local file system
)
```

## Data Management Patterns

### Sequential Re-assembly Pattern
The core innovation of this system:

1. **Batch Execution**: Run scripts multiple times over several days
2. **Data Storage**: Each run saves data with unique Assembly ID
3. **Assembly Tracking**: Assembly list tracks all completed runs
4. **Re-assembly**: Combine all data files into single DataFrame

```python
# Complete workflow example
executor = create_batch_executor('batch-parameter-distortion', '01', 'v1')

# Run multiple batches over time...
# (Each run generates new Assembly ID and saves data)

# Later, re-assemble all data
combined_data = executor.sequential_reassembly('parameter_distortion_results')
```

### File Naming Convention
- Data files: `parameter_distortion_results_20251216_142230_run001.pkl`
- Assembly list: `assembly_list.csv`
- Supports easy identification and combination

## Error Handling and Recovery

### Resume Capabilities
The system includes built-in error handling:
- Assembly status tracking prevents duplicate processing
- Individual batch failures don't affect previous runs
- Failed assemblies can be marked for retry

### Monitoring Batch Progress
```python
executor = create_batch_executor('batch-parameter-distortion', '01', 'v1')
assembly_list = executor.load_assembly_list()

# Check completed vs started assemblies
completed = assembly_list[assembly_list['status'] == 'completed']
started = assembly_list[assembly_list['status'] == 'started']

print(f"Completed: {len(completed)}, Started: {len(started)}")
```

## Customization Guide

### Creating New Batch Scripts

1. **Copy Template**: Use existing scripts as templates
2. **Modify Configuration**: Update `load_experiment_config()`
3. **Implement Perturbation**: Add your specific noise/distortion logic
4. **Update Naming**: Change experiment names and numbers

### Example: Creating a New Variation
```python
# Based on parameter distortion template
def apply_custom_perturbation(parameters, custom_factor, seed):
    # Implement your specific perturbation logic
    pass

def run_batch_custom_experiment():
    # Copy structure from existing batch functions
    # Modify perturbation application and naming
    pass
```

## Performance Considerations

### Memory Optimization
- Batch scripts use reasonable default sample sizes (n=100)
- Adjust `n_samples` and `n_reps` based on available resources
- Sequential re-assembly loads data files one at a time

### Parallel Processing
- Machine learning evaluation uses Joblib parallelization
- Dynamic feature calculation uses multi-core processing
- Adjust `n_cores` parameters based on system capabilities

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `src` is in Python path
2. **S3 Connection**: Verify S3 environment variables
3. **Memory Issues**: Reduce sample sizes in configuration
4. **Assembly Tracking**: Check assembly list file permissions

### Debug Mode
Add debug prints to track execution:
```python
def run_batch_parameter_distortion():
    print("🚀 Starting batch execution...")
    # Add progress prints throughout function
    print(f"✅ Step completed: {step_name}")
```

## Best Practices

1. **Version Control**: Use different version numbers for major configuration changes
2. **Documentation**: Update configuration comments when modifying experiments
3. **Testing**: Run small-scale tests before large batch executions
4. **Monitoring**: Regularly check assembly list for failed runs
5. **Backup**: Consider local backups before large S3 operations

## Example Workflow

### Multi-Day Experiment
```bash
# Day 1: Run first batch
python src/scripts/4_01_v1_batch-parameter-distortion.py

# Day 2: Run second batch  
python src/scripts/4_01_v1_batch-parameter-distortion.py

# Day 3: Run final batch and re-assemble
python -c "
from scripts.batch_framework import create_batch_executor
executor = create_batch_executor('batch-parameter-distortion', '01', 'v1')
combined = executor.sequential_reassembly('parameter_distortion_results')
print(f'Combined data: {combined.shape}')
"
```

This system enables robust, scalable batch processing for intensive data generation tasks while maintaining data integrity through the sequential re-assembly pattern.
