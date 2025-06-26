filename = "G:\My Drive\DAWSON PHD PROJECT\Biomarker Data Repository\data\new-peak-project\experiments\test6_4_50\result.h5";

% Read numeric data
data = h5read(filename, '/data/block0_values')';  % Transpose to get rows x cols

% Read column names
raw_names = h5read(filename, '/data/block0_items');

% Convert fixed-length character array to cell array of strings
if ischar(raw_names)
    column_names = cellstr(raw_names');  % Transpose first, then convert
elseif isstring(raw_names) || iscellstr(raw_names)
    column_names = raw_names;
else
    error('Unknown format for column names.');
end

% Construct table
T = array2table(data, 'VariableNames', column_names);
