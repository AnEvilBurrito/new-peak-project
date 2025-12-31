# Configuration-Based Scripts Usage Guide

## Overview

This directory contains **consolidated configuration-based versions** of all major data generation and ML batch loading scripts. These scripts are designed for **remote batch job execution** where modifying script variables is more practical than using CLI arguments.

## Available Configuration-Based Scripts

1. **`expression-noise-v1.py`** - Expression noise data generation
2. **`parameter-distortion-v2.py`** - Parameter distortion data generation  
3. **`response-noise-v1.py`** - Response noise data generation
4. **`create-ml-loader-v1.py`** - ML batch task list generation

## Key Features

### 1. Configuration Variables at Top
Each script has a clear `CONFIGURATION SECTION` at the top with all adjustable parameters:

```python
# ===== CONFIGURATION SECTION =====
MODEL_NAME = "sy_simple"  # Can be string or list for multiplexing
NOISE_LEVELS = [0, 0.1, 0.2, 0.3, 0.5]
N_SAMPLES = 2000
SEED = 42
UPLOAD_S3 = True
SEND_NOTIFICATIONS = True
# ===== END CONFIGURATION =====
```

### 2. Model Multiplexing Support
Single model (string):
```python
MODEL_NAME = "sy_simple"
```

Multiple models (list):
```python
MODEL_NAME = ["sy_simple", "fgfr4_model", "model_v3"]
```

### 3. Remote Batch Job Friendly
- No CLI arguments needed - modify configuration and run
- Easy to copy and modify for different batch job submissions
- Consistent naming convention (no more `*-config.py` suffix needed)

## Usage Instructions

### Basic Usage
1. **Open** any script (`expression-noise-v1.py`, etc.)
2. **Modify** configuration variables at the top of the file
3. **Run** the script:
```bash
python expression-noise-v1.py
```

### Example: Custom Batch Job
```python
# ===== CONFIGURATION SECTION =====
MODEL_NAME = ["sy_simple", "fgfr4_model"]  # Process 2 models
NOISE_LEVELS = [0, 0.1, 0.2, 0.3]          # Custom noise levels
N_SAMPLES = 1000                           # Reduced for testing
UPLOAD_S3 = True                           # Upload to S3
SEND_NOTIFICATIONS = False                  # Disable for batch jobs
# ===== END CONFIGURATION =====
```

### Example: Quick Local Test
```python
# ===== CONFIGURATION SECTION =====
MODEL_NAME = "sy_simple"                   # Single model
NOISE_LEVELS = [0, 0.3]                    # Only 2 noise levels
N_SAMPLES = 100                            # Small sample size
UPLOAD_S3 = False                          # Skip S3 upload
SEND_NOTIFICATIONS = False                 # Skip notifications
# ===== END CONFIGURATION =====
```

## Configuration Variables Reference

### Common to All Scripts
| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MODEL_NAME` | str or list | `"sy_simple"` | Model name(s) to process |
| `UPLOAD_S3` | bool | `True` | Upload results to S3 |
| `SEND_NOTIFICATIONS` | bool | `True` | Send ntfy notifications |
| `N_SAMPLES` | int | `2000` | Number of samples per level |
| `SEED` | int | `42` | Random seed for reproducibility |

### Experiment-Specific Variables
| Script | Variable | Default | Description |
|--------|----------|---------|-------------|
| Expression Noise | `NOISE_LEVELS` | `[0, 0.1, 0.2, 0.3, 0.5, 1.0]` | Gaussian noise levels |
| Parameter Distortion | `DISTORTION_FACTORS` | `[0, 1.1, 1.3, 1.5, 2.0, 3.0]` | Parameter distortion factors |
| Response Noise | `NOISE_LEVELS` | `[0, 0.05, 0.1, 0.2, 0.3, 0.5]` | Response noise levels |
| All | `SIMULATION_PARAMS` | `{'start': 0, 'end': 10000, 'points': 101}` | Simulation parameters |
| All | `OUTCOME_VAR` | `"Oa"` | Target outcome variable |

### ML Loader Specific
| Variable | Default | Description |
|----------|---------|-------------|
| `EXPERIMENT_TYPES` | `["expression-noise-v1", "parameter-distortion-v2", "response-noise-v1"]` | Experiment types to include |
| `OUTPUT_CSV` | `"ml_batch_tasks.csv"` | Output CSV file path |
| `VERIFY_EXISTS` | `True` | Verify files exist in S3 |
| `GENERATE_ONLY` | `False` | Only generate CSV, don't show examples |

## Workflow Examples

### 1. Generate Data for Multiple Models
```python
# expression-noise-v1.py configuration
MODEL_NAME = ["sy_simple", "fgfr4_model", "peak3_model"]
NOISE_LEVELS = [0, 0.2, 0.5]
N_SAMPLES = 5000  # Larger for production
UPLOAD_S3 = True
```

### 2. Create ML Batch Task List
```python
# create-ml-loader-v1.py configuration  
MODEL_NAME = ["sy_simple", "fgfr4_model"]
EXPERIMENT_TYPES = ["expression-noise-v1", "parameter-distortion-v2"]
OUTPUT_CSV = "production_tasks_2024.csv"
VERIFY_EXISTS = True
```

### 3. Quick Validation Run
```python
# Any script - validation setup
MODEL_NAME = "sy_simple"
N_SAMPLES = 100  # Small for quick validation
UPLOAD_S3 = False  # Skip S3 for testing
SEND_NOTIFICATIONS = False  # Quiet mode
```

## File Structure

```
data-eng/
├── Consolidated configuration-based scripts:
│   ├── expression-noise-v1.py
│   ├── parameter-distortion-v2.py
│   ├── response-noise-v1.py
│   └── create-ml-loader-v1.py
│
├── Test files (moved to src/tests):
│   ├── test_all_scripts.py
│   └── test_consolidated_scripts.py
│
├── Shell scripts for batch execution:
│   ├── expression-noise-v1.sh
│   ├── parameter-distortion-v2.sh
│   └── response-noise-v1.sh
│
└── Shared utilities:
    ├── ml_task_utils.py
    ├── sy_simple-make-data-v1.py
    ├── sy_simple-parameter-distortion-v1.py
    └── topology-noise-v1.py
```

## Benefits Over CLI Approach

1. **Easier batch job submission**: Modify variables instead of complex CLI commands
2. **Better documentation**: Configuration is self-documented at top of file
3. **Reduced errors**: No typos in long CLI argument strings
4. **Version control friendly**: Easy to track configuration changes
5. **Template reuse**: Copy files for different experiments
6. **Simplified maintenance**: No duplicate code between CLI and config versions

## Testing

Test scripts are located in `src/tests/`:
- `test_all_scripts.py` - Original test script
- `test_consolidated_scripts.py` - Tests for consolidated configuration-based scripts

Run tests from the project root:
```bash
cd c:\Github\dev2
python -m pytest src/tests/test_consolidated_scripts.py
```

## Migration from Old CLI to Configuration

**Before (Old CLI version - now removed):**
```bash
python expression-noise-v1.py --model sy_simple --noise-levels 0 0.1 0.2 --samples 2000 --upload-s3
```

**After (Current configuration-based version):**
1. Open `expression-noise-v1.py`
2. Modify configuration:
   ```python
   MODEL_NAME = "sy_simple"
   NOISE_LEVELS = [0, 0.1, 0.2]
   N_SAMPLES = 2000
   UPLOAD_S3 = True
   ```
3. Run:
   ```bash
   python expression-noise-v1.py
   ```

## Next Steps

1. Use the consolidated scripts for all future batch jobs
2. Create experiment-specific configurations by copying and renaming files
3. Store configurations in version control for reproducibility
4. Use the ML loader to generate task lists for batch evaluation

## Support

- All scripts are configuration-based with clear variables at the top
- Test scripts verify functionality
- Shell scripts provide batch execution examples
- Consistent with the `expression-noise-v1.py` pattern you requested

## Recent Changes

- ✅ **Consolidated**: Removed separate CLI and config versions
- ✅ **Simplified**: All scripts now configuration-based only
- ✅ **Organized**: Test files moved to `src/tests/`
- ✅ **Updated**: Documentation reflects new structure
