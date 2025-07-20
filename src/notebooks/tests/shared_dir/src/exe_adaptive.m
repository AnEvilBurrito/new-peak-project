clear; clc; close all;

% Load model variables
model = 'FGFR4_model_rev2a_mex';
paramnames      = eval(strcat("deblank(",model,"('Parameters'))"));
statenames      = eval(strcat("deblank(",model,"('States'))"));
variable_names  = eval(strcat("deblank(",model,"('VariableNames'))"));
p0              = eval(strcat(model,"('parametervalues')"));
X0              = eval(model);

% Input/output paths
init_file       = 'all_initial_conditions.csv';
param_file      = 'true_parameters.csv';
output_dir      = 'G:/My Drive/DAWSON PHD PROJECT/Biomarker Data Repository/data/new-peak-project/experiments/matlab_output';

% Sampling settings
distortion_scale = 4;
max_attempts     = 50;
random_seed      = 42;

% Output files
param_outfile       = fullfile(output_dir, 'distorted_params_4_test.csv');
map_outfile         = fullfile(output_dir, 'param_ic_mapping_4_test.csv');
skipped_log_file    = fullfile(output_dir, 'skipped_ics_log_4_test.txt');
simulation_outfile  = fullfile(output_dir, 'adaptive_suboptimal_data_4_test.csv');  % new!

% Run adaptive sampling
[params, mapping, total_tries, skipped, final_tbl] = adaptive_parameter_sampling( ...
    init_file, param_file, distortion_scale, max_attempts, ...
    random_seed, true, skipped_log_file, simulation_outfile);

% Save results
writetable(params, param_outfile);
writetable(mapping, map_outfile);
% final_tbl already written inside the function if path is non-empty

% Console output
fprintf("Adaptive parameter sampling complete.\n");
fprintf("Total successful sets: %d\n", height(params));
fprintf("Total simulation attempts: %d\n", total_tries);
fprintf("Skipped ICs due to baseline failure: %d\n", numel(skipped));
