function tbl = run_FGFR4_model(init_conditions, bestfit_paramsets, paramsets_id, norm, readouts, timeframe)
    model = 'FGFR4_model_rev2a_mex';
    param_names     = eval(strcat("deblank(",model,"('Parameters'))"));
    state_names     = eval(strcat("deblank(",model,"('States'))"));
    variable_names  = eval(strcat("deblank(",model,"('VariableNames'))"));
    p0  = eval(strcat(model,"('parametervalues')"));
    X0  = eval(model);

    paramnames = param_names;
    statenames = state_names;
    x0s = init_conditions;
    if isstring(init_conditions)
        if init_conditions == "default"
            x0s = X0;
        end
    end 

    % settings for the simulation
    StimOn  = bestfit_paramsets(1,strcmp(paramnames,'FGF_on'));
    options.maxnumsteps = 2000;
    
    % time frames for the simulation
    % 5000 (starvation) --> 5000 (q-stimulation) --> drug response time
    drug_response_time = timeframe;
    if isstring(timeframe)
        if timeframe == "default"
            drug_response_time = [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]*60; % [min]
        end
    end 
    Time_starv_section  = linspace(0,StimOn,1000);
    Time_qstim_section  = Time_starv_section(end) + linspace(0,StimOn,1000)  ;
    Time_drug_section   = Time_qstim_section(end) + reshape(drug_response_time,1,[]);
    tspan               = sort(unique([Time_starv_section Time_qstim_section Time_drug_section]));
    
    Tindex_starv        = ismember(tspan,Time_starv_section);
    Tindex_qstim        = ismember(tspan,Time_qstim_section);
    Tindex_drug         = ismember(tspan,Time_drug_section);
    
    Time_starv          = tspan(Tindex_starv);
    Time_qstim          = tspan(Tindex_qstim) - Time_starv_section(end);
    Time_drug           = tspan(Tindex_drug) - Time_qstim_section(end);
    
    parm_id = paramsets_id; % the top 50 best-fitted parameters
    
    paramvals = bestfit_paramsets(parm_id,:);
    
    % simulation setting for the drug treatment
    drug_names = {'FGFR4i_0';'PI3Ki_0';'ERBBi_0';'AKTi_0';'MEKi_0'};
    drug_dose = [100, 0, 0, 0, 0]; % nM {10nMBLU + 500nMBYL; 100nMBLU}
    % drug treatment (drug 1 and 2)
    paramvals(strcmp(paramnames,'inh_on')) = Time_qstim_section(end);
    if ~all(strcmp(paramnames(ismember(paramnames,drug_names)),drug_names))
        error('=> param names are not matched to given drug names')
    end
    paramvals(ismember(paramnames,drug_names)) = drug_dose;
    
    
    % run ODE solver (MEX)
    % MEX output
    MEX_output=eval(strcat(model,"(tspan,x0s,paramvals',options)"));
    statevalues=MEX_output.statevalues;
    
    % readout variable and state variables
    state_vals_strv     = statevalues(Tindex_starv,:);
    state_vals_qstim    = statevalues(Tindex_qstim,:);
    state_vals     = statevalues(Tindex_drug,:);
    
    myreadouts = readouts;
    if isstring(readouts)
        if readouts == "active"
            myreadouts = {
                'pAkt'
                'pIGFR'
                'pFGFR4'
                'pERBB'
                'pIRS'
                'aPI3K'
                'PIP3'
                'pFRS2'
                'aGrb2'
                'aPDK1'
                'amTORC1'
                'pS6K'
                'aSos'
                'aShp2'
                'aRas'
                'aRaf'
                'pMEK'
                'pERK'
                'aGAB1'
                'aGAB2'
                'SPRY2'
                'pSPRY2'
                'PTP'
                'aCbl'
                'FOXO'
                'amTORC2'};
        end 
    end 
    
    rds_idx = ismember(MEX_output.states,myreadouts);
    readouts = MEX_output.states(ismember(MEX_output.states,myreadouts));
    
    
    arry = state_vals(:,rds_idx);
    if norm
        arry = data_normalization(arry,2); % normalized to t=0;
    end
    tbl = array2table(arry,'VariableNames',readouts,'RowNames',string(drug_response_time));
end 