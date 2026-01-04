clear; clc; close all;

% Distortion factors to loop over (standard deviation of relative error)
% Updated to match Python Gaussian multiplicative noise approach
% distortion_factor = std dev of epsilon in: p' = p * (1 + epsilon), epsilon ~ N(0, distortion_factor)
distortion_scales = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0];

% Load model and parameter info
model = 'FGFR4_model_rev2a_mex';
paramnames     = eval(strcat("deblank(",model,"('Parameters'))"));
statenames     = eval(strcat("deblank(",model,"('States'))"));
variable_names = eval(strcat("deblank(",model,"('VariableNames'))"));
p0             = eval(strcat(model,"('parametervalues')"));
X0             = eval(model);

% Paths
init_file   = 'all_initial_conditions.csv';
output_dir  = 'G:/My Drive/DAWSON PHD PROJECT/Biomarker Data Repository/data/new-peak-project/experiments/matlab_output';

% Base parameter file
base_param_file = 'true_parameters.csv';

% Settings
max_attempts = 50;
random_seed  = 42;

% Loop over distortion scales
for i = 1:length(distortion_scales)
    scale = distortion_scales(i);
    scale_str = strrep(num2str(scale), '.', '_');
    scale_raw = num2str(scale);

    fprintf("Running distortion scale = %s\n", scale_raw);

    % Output filenames
    param_outfile     = fullfile(output_dir, sprintf('distorted_params_%s.csv', scale_str));
    map_outfile       = fullfile(output_dir, sprintf('param_ic_mapping_%s.csv', scale_str));
    skipped_log_file  = fullfile(output_dir, sprintf('skipped_ics_log_%s.txt', scale_str));
    simulation_outfile = fullfile(output_dir, sprintf('adaptive_suboptimal_data_%s.csv', scale_raw));  % NEW

    % Run adaptive sampling
    [params, mapping, total_tries, skipped, final_tbl] = adaptive_parameter_sampling( ...
        init_file, base_param_file, scale, max_attempts, random_seed, false, ...
        skipped_log_file, simulation_outfile);  % <- output CSV path passed in

    % Write metadata tables
    writetable(params, param_outfile);
    writetable(mapping, map_outfile);

    % Log
    fprintf("Finished scale %s: %d successful sets, %d total tries\n", ...
        scale_raw, height(params), total_tries);
    fprintf("Skipped ICs due to baseline failure: %d\n", numel(skipped));
end