% test_simulation.m

% Load or define the inputs
model = 'FGFR4_model_rev2a_mex';

% Load model parameter names and initial state
paramnames = eval(strcat("deblank(",model,"('Parameters'))"));
x0s        = eval(model);
p0         = eval(strcat(model,"('parametervalues')"));

% Directory where CSVs are saved
data_dir = 'C:\Github\new-peak-project\src\notebooks\tests\shared_dir\src';  % change this to your actual path

% Load parameters
param_data = readcell(fullfile(data_dir, 'parameters.csv'));  % cell array
paramnames = param_data(2:end, 1);                             % exclude header
paramvals  = cell2mat(param_data(2:end, 2));                   % convert values to numeric

% Load initial conditions
state_data = readcell(fullfile(data_dir, 'init_conditions.csv'));
statenames = state_data(2:end, 1);  % cell array of names
x0s        = cell2mat(state_data(2:end, 2));

%% Call the function
[tbl, statevalues, tspan] = run_simulation(paramvals, x0s, paramnames);

%% export tbl to h5



%% Show results
disp(tbl);
state_names = eval(strcat(model, "('States')"));
pERK_index = strcmp(deblank(state_names), 'pERK');
plot(tspan, statevalues(:, pERK_index));
xlabel('Time (min)');
ylabel('pERK level');
title('pERK Time Course');

