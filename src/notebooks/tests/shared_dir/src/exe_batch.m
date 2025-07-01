clear; clc; close all;

%  global variables
% (param names, state names, param values, initial values)
model = 'FGFR4_model_rev2a_mex';
paramnames     = eval(strcat("deblank(",model,"('Parameters'))"));
statenames     = eval(strcat("deblank(",model,"('States'))"));
variable_names  = eval(strcat("deblank(",model,"('VariableNames'))"));
p0  = eval(strcat(model,"('parametervalues')"));
X0  = eval(model);

init_file = 'all_initial_conditions.csv';
param_file = 'true_parameters.csv';
output_dir = 'G:/My Drive/DAWSON PHD PROJECT/Biomarker Data Repository/data/new-peak-project/experiments/matlab_output';
output_dir_file = [output_dir, '/batch_simulation_data_comb.csv'];

%%
batch_run_simulation(init_file, param_file, paramnames, output_dir_file, 'MATCH')