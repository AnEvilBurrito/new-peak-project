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
% distortion_scale is now interpreted as standard deviation of relative error
% p' = p * (1 + epsilon), epsilon ~ N(0, distortion_scale)
distortion_scale = 0.2;  % 20% standard deviation (reasonable for Gaussian noise)
max_attempts     = 50;
random_seed      = 42;

% Output files
scale_str = strrep(num2str(distortion_scale), '.', '_');
param_outfile       = fullfile(output_dir, sprintf('distorted_params_%s_test.csv', scale_str));
map_outfile         = fullfile(output_dir, sprintf('param_ic_mapping_%s_test.csv', scale_str));
skipped_log_file    = fullfile(output_dir, sprintf('skipped_ics_log_%s_test.txt', scale_str));
simulation_outfile  = fullfile(output_dir, sprintf('adaptive_suboptimal_data_%s_test.csv', scale_str));  % new!

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