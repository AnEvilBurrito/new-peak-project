function batch_run_simulation(init_file, param_file, paramnames, out_file, mode)
% BATCH_RUN_SIMULATION runs (IC x PARAM) combinations in parallel.
%
% Inputs:
%   init_file   - .csv file with initial conditions (first column = ID)
%   param_file  - .csv file with parameter sets (first column = ID)
%   paramnames  - 1 x P cell array of parameter names
%   out_file    - Output filename (.csv)
%   mode        - 'COMBINE' or 'MATCH'

    arguments
        init_file
        param_file
        paramnames
        out_file
        mode {mustBeMember(mode, {'COMBINE', 'MATCH'})}
    end

    % Load initial conditions
    init_tbl = readtable(init_file);
    init_ids = init_tbl{:,1};         % first column = ID
    init_mat = init_tbl{:,2:end};     % remaining = initial conditions

    % Load parameter sets
    param_tbl = readtable(param_file);
    param_ids = param_tbl{:,1};       % first column = ID
    param_mat = param_tbl{:,2:end};   % remaining = parameter values

    x = size(init_mat, 1);  % number of initial conditions
    y = size(param_mat, 1); % number of parameter sets

    switch mode
        case 'COMBINE'
            total_runs = x * y;
            [I_idx, J_idx] = ind2sub([x, y], 1:total_runs);

        case 'MATCH'
            if x ~= y
                error("MATCH mode requires the same number of initial conditions and parameter sets (got %d vs %d).", x, y);
            end
            total_runs = x;
            I_idx = 1:x;
            J_idx = 1:y;
    end

    % Preallocate outputs
    all_tbl = cell(1, total_runs);
    failure_flags = false(1, total_runs);

    parfor run_idx = 1:total_runs
        i = I_idx(run_idx);
        j = J_idx(run_idx);

        x0 = init_mat(i, :);
        paramvals = param_mat(j, :);
        ic_id = init_ids(i);
        param_id = param_ids(j);

        try
            [tbl, ~, ~] = run_simulation(paramvals, x0, paramnames);

            ic_id_str = convertToChar(ic_id);
            param_id_str = convertToChar(param_id);

            % Add metadata
            tbl.RunID = repmat(run_idx, height(tbl), 1);
            tbl.IC_ID = repmat({ic_id_str}, height(tbl), 1);
            tbl.ParamSet_ID = repmat({param_id_str}, height(tbl), 1);
            tbl.Time = str2double(tbl.Properties.RowNames);

            tbl.Properties.RowNames = {};
            all_tbl{run_idx} = tbl;

        catch ME
            failure_flags(run_idx) = true;
            fprintf("Simulation failed: Run %d (IC %s, ParamSet %s): %s\n", ...
                run_idx, convertToChar(ic_id), convertToChar(param_id), ME.message);
        end
    end

    % Assemble successful results
    success_tbls = all_tbl(~failure_flags);
    if isempty(success_tbls)
        error("All simulations failed. Nothing to save.");
    end

    final_tbl = vertcat(success_tbls{:});

    % Reorder columns
    meta_cols = {'RunID', 'IC_ID', 'ParamSet_ID', 'Time'};
    data_cols = setdiff(final_tbl.Properties.VariableNames, meta_cols, 'stable');
    final_tbl = final_tbl(:, [meta_cols, data_cols]);

    % Ensure .csv extension
    if ~endsWith(out_file, '.csv')
        out_file = strcat(out_file, '.csv');
    end

    % Write output table
    writetable(final_tbl, out_file);

    % Final report
    num_failures = sum(failure_flags);
    num_success = total_runs - num_failures;

    fprintf("Simulations completed: %d successful, %d failed.\n", num_success, num_failures);
    fprintf("Results saved to: %s\n", out_file);
end

function out = convertToChar(x)
% CONVERTTOCHAR Safely convert any value (including cell wrappers) to a char vector
    if iscell(x)
        x = x{1};  % unwrap one layer
    end
    if ischar(x)
        out = x;
    elseif isstring(x)
        out = char(x);
    elseif isnumeric(x)
        out = num2str(x);
    else
        error("Unsupported ID type: %s", class(x));
    end
end
