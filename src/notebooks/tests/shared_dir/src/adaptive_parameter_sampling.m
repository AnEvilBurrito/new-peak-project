function [accepted_params, param_ic_map, total_tries, skipped_ics, final_tbl] = adaptive_parameter_sampling( ...
    init_file, param_file, distortion_scale, max_attempts, seed, verbose, skip_log_file, output_csv_file)

    if nargin < 6, verbose = true; end
    if nargin < 7, skip_log_file = ''; end
    if nargin < 8, output_csv_file = ''; end

    model = 'FGFR4_model_rev2a_mex';
    paramnames = eval(strcat("deblank(",model,"('Parameters'))"));
    rng(seed);

    % Load initial conditions
    init_tbl = readtable(init_file);
    ic_ids = string(init_tbl{:,1});
    x0_list = init_tbl{:,2:end};

    % Load base parameter values
    param_tbl = readtable(param_file);
    param_template_vals = param_tbl{1, 2:end};

    num_ic = size(x0_list, 1);
    num_params = numel(param_template_vals);

    % Gaussian multiplicative noise parameters
    % distortion_scale is now interpreted as standard deviation (std) of relative error
    % p' = p * (1 + epsilon), where epsilon ~ N(0, distortion_scale)
    
    % Warning for large distortion factors (like Python script)
    if distortion_scale > 1.0
        warning("distortion_scale=%f is large (>1.0 = 100%% std dev). Most parameters will change by more than 100%%.", distortion_scale);
    end
    
    % Output containers
    accepted_params = table();
    param_ic_map = [];
    skipped_ics = {};
    total_tries = 0;
    param_set_id = 0;

    for i = 1:num_ic
        ic_id = ic_ids(i);
        x0s = double(x0_list(i, :));

        % Check baseline parameter set first
        try
            [~, ~, ~] = run_simulation(param_template_vals(:)', x0s(:)', paramnames);
        catch
            if verbose
                warning("IC %s: baseline parameter set failed — skipping resampling", ic_id);
            end
            skipped_ics{end+1,1} = ic_id;
            continue;
        end

        % Baseline is valid, proceed with distortion attempts
        attempt = 0;
        success = false;

        while ~success && attempt < max_attempts
            attempt = attempt + 1;
            total_tries = total_tries + 1;

            % Generate Gaussian multiplicative noise
            % epsilon ~ N(0, distortion_scale)
            epsilon = normrnd(0, distortion_scale, 1, num_params);
            
            % Apply multiplicative noise: p' = p * (1 + epsilon)
            distorted_vals = param_template_vals .* (1 + epsilon);
            
            % Ensure positivity (clip at 1e-8 like Python implementation)
            distorted_vals = max(distorted_vals, 1e-8);

            try
                [~, ~, ~] = run_simulation(distorted_vals(:)', x0s(:)', paramnames);
                row = array2table(distorted_vals, 'VariableNames', paramnames);
                row.param_set_id = param_set_id;
                accepted_params = [accepted_params; row];
                param_ic_map = [param_ic_map; {param_set_id, ic_id}];
                param_set_id = param_set_id + 1;
                success = true;
            catch ME
                if verbose
                    warning("IC %s failed on attempt %d: %s", ic_id, attempt, ME.message);
                end
            end
        end

        if ~success && verbose
            warning("IC %s: all %d distortion attempts failed", ic_id, max_attempts);
        end
    end

    if size(accepted_params, 1) ~= size(param_ic_map, 1)
        error("Mismatch between accepted_params and param_ic_map rows!");
    end

    % Convert param_ic_map to table
    param_ic_map = cell2table(param_ic_map, 'VariableNames', {'param_set_id', 'IC_ID'});
    accepted_params = movevars(accepted_params, 'param_set_id', 'Before', 1);

    % Final verification and output generation
    fprintf("Verifying %d accepted parameter sets...\n", height(accepted_params));
    all_tbl = cell(height(accepted_params), 1);

    for k = 1:height(accepted_params)
        row_param = double(accepted_params{k, 2:end})';
        ic_id = param_ic_map.IC_ID(k);
        x0_row_idx = find(string(init_tbl{:,1}) == ic_id, 1);

        if isempty(x0_row_idx)
            error("IC_ID %s not found in initial condition table during verification.", ic_id);
        end

        x0 = double(init_tbl{x0_row_idx, 2:end})';

        try
            [tbl, ~, ~] = run_simulation(row_param, x0, paramnames);
        catch ME
            error("Final validation failed at index %d: param_set_id=%d, IC=%s\nError: %s", ...
                k, accepted_params.param_set_id(k), ic_id, ME.message);
        end

        % Format to match batch_run_simulation output
        tbl.RunID = repmat(k, height(tbl), 1);
        tbl.IC_ID = repmat({convertToChar(ic_id)}, height(tbl), 1);
        tbl.ParamSet_ID = repmat({convertToChar(accepted_params.param_set_id(k))}, height(tbl), 1);
        tbl.Time = str2double(tbl.Properties.RowNames);
        tbl.Properties.RowNames = {};
        all_tbl{k} = tbl;
    end

    fprintf("All accepted parameter sets passed final simulation validation.\n");

    final_tbl = vertcat(all_tbl{:});
    meta_cols = {'RunID', 'IC_ID', 'ParamSet_ID', 'Time'};
    data_cols = setdiff(final_tbl.Properties.VariableNames, meta_cols, 'stable');
    final_tbl = final_tbl(:, [meta_cols, data_cols]);

    % Optionally write simulation output table
    if ~isempty(output_csv_file)
        writetable(final_tbl, output_csv_file);
        fprintf("Simulation data written to: %s\n", output_csv_file);
    end

    % Optionally write skipped ICs
    if ~isempty(skip_log_file)
        fid = fopen(skip_log_file, 'w');
        fprintf(fid, "Skipped ICs due to base param failure:\n");
        for j = 1:length(skipped_ics)
            fprintf(fid, "%s\n", string(skipped_ics{j}));
        end
        fclose(fid);
    end
end

function out = convertToChar(x)
    if iscell(x), x = x{1}; end
    if ischar(x), out = x;
    elseif isstring(x), out = char(x);
    elseif isnumeric(x), out = num2str(x);
    else, error("Unsupported ID type: %s", class(x));
    end
end