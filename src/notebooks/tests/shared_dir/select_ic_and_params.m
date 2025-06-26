clear; clc; close all;

%  global variables
% (param names, state names, param values, initial values)
model = 'FGFR4_model_rev2a_mex';
param_names     = eval(strcat("deblank(",model,"('Parameters'))"));
state_names     = eval(strcat("deblank(",model,"('States'))"));
variable_names  = eval(strcat("deblank(",model,"('VariableNames'))"));
p0  = eval(strcat(model,"('parametervalues')"));
X0  = eval(model);

%% 1. load the best-fitted parameter sets and init conditions
bestfit_paramsets = readmatrix('fitted_paramsets_rev2_STEP3.csv');
% note: the first column is the fit score
bestfit_paramsets(:,1) = [];
% the first parameter used for this synthetic study
paramnames = param_names;
statenames = state_names;

% Select parameter set
param_id = 1;
paramvals = bestfit_paramsets(param_id, :);
init_id = 4;

init_conditions_table = readtable('median-ccle_protein_expression-fgfr4_model_ccle_match_rules-375x51-initial_conditions.csv.csv','Delimiter',',', 'ReadVariableNames', true');
x0s = transpose(table2array(init_conditions_table(init_id,2:end)));

%% Set-up initial_conditions and parameters variables

init_conditions = [statenames, num2cell(x0s)]; 
parameters = [paramnames, num2cell(paramvals.')];

%% Export 

output = [{'State', 'Initial Value'}; init_conditions];
writecell(output, 'init_conditions.csv')

output = [{'Parameter', 'Value'}; parameters];
writecell(output, 'parameters.csv')