function distorted_params_tbl = generate_parameters(param_csv, num_samples, scale, seed)
%GENERATE_PARAMETERS Generate distorted parameter sets from base CSV file.
%
% Inputs:
%   param_csv   - Path to CSV with columns 'Parameter' and 'Value'
%   num_samples - Number of distorted samples to generate
%   scale       - Distortion scale (standard deviation of relative error)
%                 p' = p * (1 + epsilon), epsilon ~ N(0, scale)
%   seed        - Random seed for reproducibility
%
% Output:
%   distorted_params_tbl - Table with distorted parameter sets and param_set_id

    % Read base parameters
    base_tbl = readtable(param_csv, 'TextType', 'string');
    param_names = base_tbl.Parameter;
    param_values = base_tbl.Value;

    % Set random seed
    rng(seed);

    % Preallocate output
    param_mat = zeros(num_samples, numel(param_values));
    param_ids = (0:num_samples-1)';

    for i = 1:num_samples
        % Generate Gaussian multiplicative noise
        % epsilon ~ N(0, scale)
        epsilon = normrnd(0, scale, size(param_values));
        
        % Apply multiplicative noise: p' = p * (1 + epsilon)
        distorted_vals = param_values .* (1 + epsilon);
        
        % Ensure positivity (clip at 1e-8)
        distorted_vals = max(distorted_vals, 1e-8);
        
        param_mat(i, :) = distorted_vals;
    end

    % Convert to table
    distorted_params_tbl = array2table(param_mat, 'VariableNames', param_names');
    distorted_params_tbl.param_set_id = param_ids;

    % Move param_set_id to first column
    distorted_params_tbl = movevars(distorted_params_tbl, 'param_set_id', 'Before', 1);
    
    % Warning for large distortion factors
    if scale > 1.0
        warning("distortion_scale=%f is large (>1.0 = 100%% std dev). Most parameters will change by more than 100%%.", scale);
    end
end