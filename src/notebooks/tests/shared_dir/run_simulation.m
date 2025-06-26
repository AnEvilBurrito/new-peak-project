function [tbl, statevalues, tspan] = run_simulation(paramvals, x0s, paramnames)
% RUN_SIMULATION runs FGFR4 MEX model with specified parameters and returns key outputs.
%
% Inputs:
%   paramvals    - 1 x P vector of parameter values
%   x0s          - initial state values (1 x N)
%   paramnames   - 1 x P cell array of parameter names
%
% Outputs:
%   tbl          - readout table for drug phase
%   statevalues  - full simulation state matrix
%   tspan        - simulation time vector

    % Validate input
    if length(paramvals) ~= length(paramnames)
        error('paramvals and paramnames must be the same length');
    end

    % Model setup
    model = 'FGFR4_model_rev2a_mex';
    options.maxnumsteps = 2000;
    StimOn = paramvals(strcmp(paramnames, 'FGF_on'));

    % Time setup (as you had)
    drug_response_time = (0:24) * 60;
    Time_starv_section = linspace(0, StimOn, 1000);
    Time_qstim_section = Time_starv_section(end) + linspace(0, StimOn, 1000);
    Time_drug_section  = Time_qstim_section(end) + reshape(drug_response_time, 1, []);
    tspan = sort(unique([Time_starv_section, Time_qstim_section, Time_drug_section]));

    Tindex_starv = ismember(tspan, Time_starv_section);
    Tindex_qstim = ismember(tspan, Time_qstim_section);
    Tindex_drug  = ismember(tspan, Time_drug_section);

    % Drug settings
    drug_names = {'FGFR4i_0', 'PI3Ki_0', 'ERBBi_0', 'AKTi_0', 'MEKi_0'};
    drug_dose  = [100, 0, 0, 0, 0];
    paramvals(strcmp(paramnames, 'inh_on')) = Time_qstim_section(end);

    if ~all(ismember(drug_names, paramnames))
        error('=> Some drug_names are not present in paramnames');
    end
    paramvals(ismember(paramnames, drug_names)) = drug_dose;
    
    missing = setdiff(drug_names, paramnames);
    if ~isempty(missing)
        error('Missing drug names: %s', strjoin(missing, ', '));
    end

    % Run model
    MEX_output = eval([model, '(tspan, x0s, paramvals'', options);']);
    statevalues = MEX_output.statevalues;

    % Readouts
    myreadouts = {
        'pAkt', 'pIGFR', 'pFGFR4', 'pERBB', 'pIRS', 'aPI3K', 'PIP3', ...
        'pFRS2', 'aGrb2', 'aPDK1', 'amTORC1', 'pS6K', 'aSos', 'aShp2', ...
        'aRas', 'aRaf', 'pMEK', 'pERK', 'aGAB1', 'aGAB2', 'SPRY2', ...
        'pSPRY2', 'PTP', 'aCbl', 'FOXO', 'amTORC2'};
    rds_idx = ismember(MEX_output.states, myreadouts);
    readouts = MEX_output.states(rds_idx);
    arry = statevalues(Tindex_drug, rds_idx);

    tbl = array2table(arry, 'VariableNames', readouts, 'RowNames', string(drug_response_time));
end
