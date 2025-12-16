## Environment

Python, package and dependency management via `uv`, virtual environment folder is `.venv`
Package management via `uv add [package-name]`
Internal repository - access via `./src/models/` 
## Data Management

S3-Compatiable access endpoint to retrieve key resources, store data generation and configurations. 
.env configuration
```env
S3_ENDPOINT='https://endpoint.endpoint.com:port'
S3_BUCKET_NAME='bio-data'
S3_SECRET_KEY='your-secret-key-here'
S3_ACCESS_KEY='your-access-key-here'
S3_REGION_NAME='us-east-1'
SAVE_RESULT_PATH='new-peak-project/experiments/ch5-paper' # s3 key prefix for files 
CONFIG_PATH='data-and-mechanism-project' 
```
## Versioning and file structure 

Main folder: `ch5-paper`
Subfolders will be labelled as `section-1`, `section-2` etc.  
Notebook located in sub-folders will be named as: `{section-number}_{exp-number}_{version-number}_{experiment-title}`
Subsequent Results from notebook will be saved at key: `{SAVE_RESULT_PATH}/{exp-number}_{version-number}_{experiment-title}/{result-name}.{extension}`

Inside a notebook, the following parameters will have to be defined: 
```python
notebook_name = 'example_name' # this is 'experiment-title'
exp_number = '01' # representing the 1st experiment in a section
section_number = '02' # meaning that this notebook is located in the 'section-2'
version_number = 'v1' # for differentiating different result versions
notebook_path = f'{SAVE_RESULT_PATH}/{exp-number}_{version-number}_{experiment-title}'
### Key notebook parameters 
# ... 
# ... 
# a .yml file will be generated at the end of the parameters and saved at {SAVE_RESULT_PATH}/{notebook_path}/{version_number}_config.yml
```

Version number is changed when parameters which affects the data/results generation for the notebook is changed. 

## Data models and methods

Below are key classes and methods required for the project, which can be found in `./src/models/`

Class Abstractions
`ModelSpec`: controls model structure 
`ModelBuilder`: controls model parameters and initial conditions 
`ODESolver`: controls model simulation, but can extract parameters if needed 

Synthetic data generation, located in `SyntheticGen` - requires refactoring 
`unified_generate_feature_data`
`unified_generate_target_data`
`unified_generate_model_timecourse_data`

External data generation, located in `./src/Scripts`
- `load_feature_data`
- `load_target_data`
- `load_timecourse_data`
- The final data table structure needs to be identical to that of the synthetic dataset generation, so downstream applications can be easily applied 
- There are some quicker pre-generated results ready for analysis for the FGFR4 model, stored at
	- `new-peak-project/matlab-output/exp22_sampled_eval_results.pkl`
	- `new-peak-project/matlab-output/exp23_eval_results.pkl` 
	- These data are stored from past experiments but should be informative for the experiment, they are accessible via s3 endpoints, an exploration of their table structure should be made before analysis
- The notebook files in `py:percent` format are located in, they contain methods on how the raw time course data can be accessed
	- `./src/notebooks/exp/exp20_fgfr4_model_prediction.py`
	- `./src/notebooks/exp/exp21_fgfr4_distort_prediction.py`
	- `./src/notebooks/exp/exp22_fgfr4_sample_size_effect.py`
	- `./src/notebooks/exp/exp23_fgfr4_batch.py`
- Should further primary data needed to be generated, a refactor plan should be considered first 

Dynamic feature processes, located in `Utils` folder, this engineers raw expression data and simulated time course data into a set of 'dynamic features' that has interpretable labels to be trained with machine learning data 
`dynamic_feature_method`
`last_time_method`

Machine learning pipe, with one implementation located in `./src/ml/Workflow.py`
`batch_eval_standard` 

## Tests 

Tests specifically for the project will be located in `./src/notebooks/ch5-paper/tests`. An example test would be to test for a successful s3 server connection.. etc. 

## Analysis

Figures: size should be usually small with large font labels and ticks so it is of high quality when seen on journals

Reports: are generated so that future analyses can directly read from a markdown file rather than the entire Jupyter notebook. The report should preserve: 
- Data table structures: of the loaded results or data 
- Analysis results: ideally with significance and p-values attached for statistical tests

Keep in mind that some files with `.py` format are actually notebooks, their `ipynb` format is auto-generated using `JupyText`. Do not run the those files in `.py` format, they are meant to be executed manually via the `ipynb` format. But you can read the code in `.py` format if needed. `.py` files that are not notebooks can be safely executed. A key difference is that notebook files in `.py` format will have `# %%` or `# %% [markdown]` tags to separate code cells and markdown cells respectively.