% Takes in a multiple (x) initial conditions and multiple (y) parameter sets and
% run all of them all, the total number of runs will be x * y

% both parameters and initial conditions should be in HDF5 format with a .h5
% suffix

function batch_run_simulation(init_file, param_file, paramnames, out_file)
% BATCH_RUN_SIMULATION performs a grid of simulations for all initial conditions × parameter sets.
%
% Inputs:
%   init_file   - HDF5 file (.h5) with initial conditions (rows = initial states)
%   param_file  - HDF5 file (.h5) with parameter sets (rows = parameter vectors)
%   paramnames  - 1 x P cell array of parameter names (must match parameter set columns)
%   out_file    - Output .h5 file to store simulation result table
%
% Example:
%   batch_run_simulation('init.h5', 'parameters.h5', paramnames, 'all_simulation_data.h5')
    % Manually load Python-written HDF5 for initial conditions and parameters
    
    % Load initial conditions
    init_data = h5read(init_file, '/data/block0_values');
    init_names = h5read(init_file, '/data/block0_items');
    init_tbl = array2table(init_data', 'VariableNames', cellstr(init_names));
    init_mat = table2array(init_tbl);
    
    % Load parameter sets
    param_data = h5read(param_file, '/data/block0_values');
    param_names = h5read(param_file, '/data/block0_items');
    param_tbl = array2table(param_data', 'VariableNames', cellstr(param_names));
    param_mat = table2array(param_tbl);


    [x, N_states] = size(init_mat);
    [y, N_params] = size(param_mat);

    % Storage containers
    all_tbl = {};
    run_idx = 1;

    % Main loop: all combinations
    for i = 1:x
        for j = 1:y
            x0 = init_mat(i, :);
            paramvals = param_mat(j, :);

            try
                [tbl, ~, ~] = run_simulation(paramvals, x0, paramnames);

                % Attach metadata columns
                tbl.RunID = repmat(run_idx, height(tbl), 1);
                tbl.IC_ID = repmat(i, height(tbl), 1);
                tbl.ParamSet_ID = repmat(j, height(tbl), 1);
                tbl.Time = str2double(tbl.Properties.RowNames);

                % Remove row names to allow HDF5 saving
                tbl.Properties.RowNames = {};

                all_tbl{end+1} = tbl; %#ok<AGROW>

                fprintf('Run %d completed: IC %d, ParamSet %d\n', run_idx, i, j);

            catch ME
                warning("Simulation failed at IC %d, ParamSet %d: %s", i, j, ME.message);
            end

            run_idx = run_idx + 1;
        end
    end

    % Combine all simulation results
    if isempty(all_tbl)
        error('No simulations completed successfully. Nothing to save.');
    end

    final_tbl = vertcat(all_tbl{:});

    % Enforce .h5 extension
    if ~endsWith(out_file, '.h5')
        warning('Output file does not end with .h5. Forcing .h5 extension.');
        out_file = strcat(out_file, '.h5');
    end

    % Save final table to HDF5 format
    writetable(final_tbl, out_file);

    fprintf("\n🎉 All %d simulations completed. \n Results saved to: %s\n", run_idx - 1, out_file);
end

