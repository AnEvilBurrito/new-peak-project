function distorted_params_tbl = generate_parameters(param_csv, num_samples, scale, seed)
%GENERATE_PARAMETERS Generate distorted parameter sets from base CSV file.
%
% Inputs:
%   param_csv   - Path to CSV with columns 'Parameter' and 'Value'
%   num_samples - Number of distorted samples to generate
%   scale       - Distortion scale (e.g., 2 → distortion range [0.5, 2])
%   seed        - Random seed for reproducibility
%
% Output:
%   distorted_params_tbl - Table with distorted parameter sets and param_set_id

    % Read base parameters
    base_tbl = readtable(param_csv, 'TextType', 'string');
    param_names = base_tbl.Parameter;
    param_values = base_tbl.Value;

    % Distortion range
    min_scale = 1 / scale;
    max_scale = scale;

    % Set random seed
    rng(seed);

    % Preallocate output
    param_mat = zeros(num_samples, numel(param_values));
    param_ids = (0:num_samples-1)';

    for i = 1:num_samples
        scale_factors = min_scale + (max_scale - min_scale) * rand(size(param_values));
        param_mat(i, :) = param_values .* scale_factors;
    end

    % Convert to table
    distorted_params_tbl = array2table(param_mat, 'VariableNames', param_names');
    distorted_params_tbl.param_set_id = param_ids;

    % Move param_set_id to first column
    distorted_params_tbl = movevars(distorted_params_tbl, 'param_set_id', 'Before', 1);
end

