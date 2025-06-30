function batch_run_simulation(init_file, param_file, paramnames, out_file)
% BATCH_RUN_SIMULATION performs a grid of simulations for all initial conditions × parameter sets.
%
% Inputs:
%   init_file   - HDF5 file (.h5) with initial conditions (first column = ID)
%   param_file  - HDF5 file (.h5) with parameter sets (first column = ID)
%   paramnames  - 1 x P cell array of parameter names (must match parameter set columns excluding ID)
%   out_file    - Output .h5 file to store simulation result table
%
% Output:
%   Saves a flat HDF5 table compatible with pandas.read_hdf()

    % Load initial conditions from Python-written HDF5
    init_data = h5read(init_file, '/data/block0_values');
    init_names = h5read(init_file, '/data/block0_items');
    init_tbl = array2table(init_data', 'VariableNames', cellstr(init_names));

    % Load parameter sets
    param_data = h5read(param_file, '/data/block0_values');
    param_names = h5read(param_file, '/data/block0_items');
    param_tbl = array2table(param_data', 'VariableNames', cellstr(param_names));

    % Separate out IDs and numeric matrices
    init_ids = init_tbl{:,1};           % First column = cell line ID
    init_mat = init_tbl{:, 2:end};      % Initial condition values

    param_ids = param_tbl{:,1};         % First column = parameter set ID
    param_mat = param_tbl{:, 2:end};    % Parameter values

    [x, ~] = size(init_mat);
    [y, ~] = size(param_mat);

    % Initialize results
    all_tbl = {};
    run_idx = 1;

    for i = 1:x
        for j = 1:y
            x0 = init_mat(i, :);
            paramvals = param_mat(j, :);
            ic_id = init_ids(i);
            param_id = param_ids(j);

            try
                [tbl, ~, ~] = run_simulation(paramvals, x0, paramnames);

                % Attach metadata columns
                tbl.RunID = repmat(run_idx, height(tbl), 1);
                tbl.IC_ID = repmat(ic_id, height(tbl), 1);
                tbl.ParamSet_ID = repmat(param_id, height(tbl), 1);
                tbl.Time = str2double(tbl.Properties.RowNames);

                % Remove row names to simplify output
                tbl.Properties.RowNames = {};

                all_tbl{end+1} = tbl; %#ok<AGROW>

                fprintf('Run %d completed: IC %s, ParamSet %s\n', run_idx, string(ic_id), string(param_id));

            catch ME
                warning("Simulation failed at IC %s, ParamSet %s: %s", string(ic_id), string(param_id), ME.message);
            end

            run_idx = run_idx + 1;
        end
    end

    % Combine all simulation results
    if isempty(all_tbl)
        error('No simulations completed successfully. Nothing to save.');
    end

    final_tbl = vertcat(all_tbl{:});

    % Reorder columns: metadata first, then readouts
    meta_cols = {'RunID', 'IC_ID', 'ParamSet_ID', 'Time'};
    data_cols = setdiff(final_tbl.Properties.VariableNames, meta_cols, 'stable');
    final_tbl = final_tbl(:, [meta_cols, data_cols]);

    % Enforce .h5 extension
    if ~endsWith(out_file, '.h5')
        warning('Output file does not end with .h5. Forcing .h5 extension.');
        out_file = strcat(out_file, '.h5');
    end

    % Save to HDF5 file as flat table
    writetable(final_tbl, out_file, 'FileType', 'spreadsheet');

    fprintf("All %d simulations completed.\nResults saved to: %s\n", run_idx - 1, out_file);
end
