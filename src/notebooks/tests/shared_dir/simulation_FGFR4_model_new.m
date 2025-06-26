clear; clc; close all;

%  global variables
% (param names, state names, param values, initial values)
model = 'FGFR4_model_rev2a_mex';
param_names     = eval(strcat("deblank(",model,"('Parameters'))"));
state_names     = eval(strcat("deblank(",model,"('States'))"));
variable_names  = eval(strcat("deblank(",model,"('VariableNames'))"));
p0  = eval(strcat(model,"('parametervalues')"));
X0  = eval(model);


%% 1. load the best-fitted parameter sets
bestfit_paramsets = readmatrix('fitted_paramsets_rev2_STEP3.csv');
% note: the first column is the fit score
bestfit_paramsets(:,1) = [];
% the first parameter used for this synthetic study
paramnames = param_names;
statenames = state_names;
x0s = X0;

%% Load the individualised specie values 

init_conditions_table = readtable('median-ccle_protein_expression-fgfr4_model_ccle_match_rules-375x51-initial_conditions.csv.csv','Delimiter',',', 'ReadVariableNames', true');
% ensure that each row is of a 'double' type that is 51x1
%% Test replacing X0s with a row from init conditions table

x0s = transpose(table2array(init_conditions_table(4,2:end)));

%% 2. run the ode model

% settings for the simulation
StimOn  = bestfit_paramsets(1,strcmp(paramnames,'FGF_on'));
options.maxnumsteps = 2000;

% time frames for the simulation
% 5000 (starvation) --> 5000 (q-stimulation) --> drug response time
drug_response_time = [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]*60; % [min]

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




parm_id = 1; % the top 50 best-fitted parameters

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

rds_idx = ismember(MEX_output.states,myreadouts);
readouts = MEX_output.states(ismember(MEX_output.states,myreadouts));


arry = state_vals(:,rds_idx);
% arry = data_normalization(arry,2); % normalized to t=0;
tbl = array2table(arry,'VariableNames',readouts,'RowNames',string(drug_response_time));


%% save the simulation results
fname = strcat('simulation_FGFR4_param_new','.xlsx');
writetable(tbl,fname,'WriteRowNames',true,'Sheet',num2str(parm_id))


%% Running the script as a loop of every cell line
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

dynamic_feature_labels = {'auc', 'median', 'tfc', 'tmax', 'max', 'tmin', 'min', 'ttsv', 'tsv', 'init'};
total_column_size = length(dynamic_feature_labels)*length(myreadouts);

all_dynamic_feature_columns = {};
for specie = myreadouts
    for dyn = dynamic_feature_labels
        dyn_feat = strcat(specie,'_',dyn);
        all_dynamic_feature_columns = [all_dynamic_feature_columns dyn_feat];
    end
end

all_dynamic_feature_columns_reshaped = reshape(transpose(all_dynamic_feature_columns), [1, total_column_size]);

total_output_matrix = [];
cell_lines_successful = [];
every_cellline = height(init_conditions_table);

for indi_row = 1:every_cellline
    % --- This part of the script extracts information from init table
    initial_conditions = table2array(init_conditions_table(indi_row,2:end));
    cellline = init_conditions_table(indi_row,'Var1');
    cellline_string = string(table2cell(cellline));
    column_names = init_conditions_table.Properties.VariableNames(2:end);
    % sets species to individualised cell line values
    x0s = transpose(table2array(init_conditions_table(indi_row,2:end)));
    
    try
        % --- Runs the model simulations
        % settings for the simulation
        StimOn  = bestfit_paramsets(1,strcmp(paramnames,'FGF_on'));
        options.maxnumsteps = 2000;
        
        % time frames for the simulation
        % 5000 (starvation) --> 5000 (q-stimulation) --> drug response time
        drug_response_time = [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]*60; % [min]
        
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
        
        
        
        
        parm_id = 1; % the top 50 best-fitted parameters
        
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
        
        rds_idx = ismember(MEX_output.states,myreadouts);
        readouts = MEX_output.states(ismember(MEX_output.states,myreadouts));
        
        
        arry = state_vals(:,rds_idx);
        % arry = data_normalization(arry,2); % normalized to t=0;
        tbl = array2table(arry,'VariableNames',readouts,'RowNames',string(drug_response_time));
    
        % - saving simulation results
        text = sprintf("simulation_data/%s.csv", cellline_string);
        writetable(tbl, text, 'Delimiter',',', 'WriteRowNames',true);
    
        dynamic_features = cell(length(myreadouts), length(dynamic_feature_labels));
        for i = 1:width(tbl)
            col_name = tbl.Properties.VariableNames{i};
            col_data = tbl.(i);
        
            % dynamic features
            auc = trapz(col_data);
            [max_val, max_time] = max(col_data);
            [min_val, min_time] = min(col_data);
            % max_time = t(max_val);
            % mean_val = mean(col_data);
            median_val = median(col_data);
            % calculation of total fold change (tfc)
            start_val = col_data(1);
            end_val = col_data(end);
            % if end_val >= 1
            %     tfc = end_val;
            % else 
            %     % end value is lower than start value, tfc will be negative 
            %     tfc = -(start_val/end_val);
            % end 
            if start_val == 0
                % tfc hard to define when start val is 0, because end value must be
                % a positive float and leads to infinity 
                tfc = 1000; 
            else
                if end_val - start_val > 0
                    tfc = ((end_val - start_val) / start_val);
                elseif end_val - start_val < 0
                    if end_val == 0
                        tfc = -1000; 
                    else
                        tfc = -((start_val - end_val) / end_val); 
                    end 
                end
            end
        
        
            % calculation of time to stability (tsv)
            tsv = length(col_data);
            abs_change_tolerance = 0.01;
            while tsv > 1
                if abs(col_data(tsv)-col_data(tsv-1)) < abs_change_tolerance
                    tsv = tsv - 1;
                else
                    tsv_value = col_data(tsv);
                    break
                end
            end
            if tsv == 1
                tsv_value = col_data(1);
            end
        
            max_sim_time = length(col_data);
            n_auc = auc / max_sim_time;
            n_max_time = max_time / max_sim_time;
            n_min_time = min_time / max_sim_time;
            n_tsv = tsv / max_sim_time; 
            dynamic_features(i,:) = {n_auc, median_val, tfc, n_max_time, max_val, n_min_time, min_val, n_tsv, tsv_value, start_val};
        end
        
        dynamic_features_reshaped = reshape(transpose(dynamic_features), [1, total_column_size]);
        
        total_output_matrix = [total_output_matrix; dynamic_features_reshaped];
        cell_lines_successful = [cell_lines_successful; cellline_string];
        disp(strcat('Complete run', string(indi_row)));
        
        % USE BELOW FOR TESTING 
        % if indi_row == 2
        %     break 
        % end 

    catch exception
        disp(exception);
    end 
end

%% save results completely
% prepare for some time for processing the file
% - saving dynamic features 
row_names = cell_lines_successful;
col_names = all_dynamic_feature_columns_reshaped;
total_output_table = array2table(total_output_matrix, "RowNames", row_names, "VariableNames",col_names);
writetable(total_output_table, "dyn_features_proteomics.csv", 'Delimiter',',', 'WriteRowNames',true);





