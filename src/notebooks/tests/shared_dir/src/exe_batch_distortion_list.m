clear; clc; close all;

% Distortion scales to loop over
distortion_scales = [1.05, 1.1, 1.2, 1.3, 1.5, 2, 4, 10, 20];

% Load model
model = 'FGFR4_model_rev2a_mex';
paramnames     = eval(strcat("deblank(",model,"('Parameters'))"));
statenames     = eval(strcat("deblank(",model,"('States'))"));
variable_names = eval(strcat("deblank(",model,"('VariableNames'))"));
p0             = eval(strcat(model,"('parametervalues')"));
X0             = eval(model);

% Paths
init_file  = 'all_initial_conditions.csv';
output_dir = 'G:/My Drive/DAWSON PHD PROJECT/Biomarker Data Repository/data/new-peak-project/experiments/matlab_output';

% Load all ICs once
init_tbl = readtable(init_file);

% Rename IC column to standard if needed
if ~any(strcmp(init_tbl.Properties.VariableNames, 'IC_ID'))
    init_tbl.Properties.VariableNames{1} = 'IC_ID';
end

total_simulations = 0;

for i = 1:length(distortion_scales)
    scale = distortion_scales(i);
    scale_str = strrep(num2str(scale), '.', '_');
    scale_raw = num2str(scale);

    % Files
    param_file = fullfile(output_dir, sprintf('distorted_params_%s.csv', scale_str));
    mapping_file = fullfile(output_dir, sprintf('param_ic_mapping_%s.csv', scale_str));
    output_file = fullfile(output_dir, sprintf('adaptive_suboptimal_data_%s.csv', scale_raw));
    temp_init_file = fullfile(output_dir, sprintf('filtered_init_%s.csv', scale_str));

    % Load parameter-IC mapping
    if ~isfile(mapping_file)
        warning("Mapping file not found for scale %s, skipping...", scale_raw);
        continue;
    end
    map_tbl = readtable(mapping_file);

    % Rename IC_ID column if needed
    if ~any(strcmp(map_tbl.Properties.VariableNames, 'IC_ID'))
        map_tbl.Properties.VariableNames{2} = 'IC_ID';  % second col should be IC_ID
    end

    % Filter initial conditions based on IC_IDs
    filtered_init = innerjoin(map_tbl(:, "IC_ID"), init_tbl, 'Keys', 'IC_ID');

    % Save filtered init file
    writetable(filtered_init, temp_init_file);

    % Run batch simulation
    fprintf('Running simulation for scale = %s\n', scale_raw);
    batch_run_simulation(temp_init_file, param_file, paramnames, output_file, 'MATCH');

    % Count simulations
    n_sim = height(filtered_init);
    fprintf('--> Simulated %d ICs for distortion scale %s\n', n_sim, scale_raw);
    total_simulations = total_simulations + n_sim;
end

fprintf('\nTotal simulations run across all distortion scales: %d\n', total_simulations);
