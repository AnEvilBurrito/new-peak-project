clear; clc; close all;

% Distortion scales to loop over
distortion_scales = [1.05, 1.1, 1.2, 1.3, 1.5, 2, 4, 10, 20];

% Static model loading
model = 'FGFR4_model_rev2a_mex';
paramnames     = eval(strcat("deblank(",model,"('Parameters'))"));
statenames     = eval(strcat("deblank(",model,"('States'))"));
variable_names = eval(strcat("deblank(",model,"('VariableNames'))"));
p0             = eval(strcat(model,"('parametervalues')"));
X0             = eval(model);

% Input initial condition file (same across runs)
init_file = 'all_initial_conditions.csv';

% Base output directory
output_dir = 'G:/My Drive/DAWSON PHD PROJECT/Biomarker Data Repository/data/new-peak-project/experiments/matlab_output';

% Loop over distortion scales
for i = 1:length(distortion_scales)
    scale = distortion_scales(i);

    % Construct file names based on current scale
    param_file = sprintf('modified_parameters_distorted_%.2f.csv', scale);
    output_file = sprintf('%s/suboptimal_data_%.2f.csv', output_dir, scale);

    fprintf('Running simulation for distortion = %.2f\n', scale);
    batch_run_simulation(init_file, param_file, paramnames, output_file, 'MATCH');
    fprintf('Finished simulation for distortion = %.2f\n', scale);
end
