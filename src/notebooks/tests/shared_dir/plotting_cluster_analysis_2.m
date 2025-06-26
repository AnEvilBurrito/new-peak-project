% Loading data first

%% load the best-fitted parameter sets
bestfit_paramsets = readmatrix('fitted_paramsets_rev2_STEP3.csv');
% note: the first column is the fit score
bestfit_paramsets(:,1) = [];

% Load the individualised specie values 

init_conditions_table = readtable('initial_conditions.csv','Delimiter',',', 'ReadVariableNames', true');

% initial set-up variables
all_species = {
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
seed = 1;
plot_id = string(seed); 

%% dynamics generation 

% simulate first N cell lines and store each outputs in a table 
sim_data = table();
total = height(init_conditions_table);
rng(seed, "twister");
random_set = randperm(total,200);
for i = 1:length(random_set)
    k = random_set(i);
    x0s = transpose(table2array(init_conditions_table(k,2:end)));
    cellline = string(init_conditions_table.Var1(k));
    % select a paramset id 
    param_id = 1;
    try
        tbl = run_FGFR4_model(x0s, bestfit_paramsets, param_id, false, "active", "default");
        sim_data = [sim_data; {cellline, tbl}];
    catch exception
        disp(cellline);
        disp(exception);
    end
end 

sim_data.Properties.VariableNames = {'cellline', 'data'};

%% plotting all species for a given cell-line

clf; 
reset(gcf);
cellline_index = 1;

for i=1:length(all_species)
    specie = all_species{i};
    cell_data = sim_data(cellline_index, "data").data;
    tbl = cell_data{1}; 
    cellline = string(sim_data(cellline_index, "cellline").cellline);
    col_data = table2array(tbl(:,specie));
    col_data = data_normalization(col_data, 2);
    plot(col_data, "LineWidth", 1)
    xlabel('Time (Hours)')
    ylabel('Expression')
    hold on 
end

grid on 
title('All Specie Time Course of the FGFR4 Model', 'FontSize', 16)
subtitle(cellline)
xlabel('Time (Hours)', 'FontSize', 14)
ylabel('Expression', 'FontSize', 14)
hold off

f = gcf; 
% Requires R2020a or later
text = sprintf("figures/%s_cellline_%d_all_species.png", plot_id, cellline_index);
exportgraphics(f,text,'Resolution',300)

%% plotting all species on subplots, for all celllines 

clf; 
reset(gcf);

all_species = {
            'pAkt'};

n = length(all_species);
cols = 4;

for i=1:n
    specie = all_species{i};
    for cell_id = 1:200 
        cell_data = sim_data(cell_id, "data").data;
        tbl = cell_data{1}; 
        % cellline = string(sim_data(cellline_index, "cellline").cellline);
        col_data = table2array(tbl(:,specie));
        col_data = data_normalization(col_data, 2);
        subplot(ceil(n/4),4,i)
        plot(col_data, "LineWidth", 1)
        hold on 
        title(specie)
    end 
end

sgtitle('All specie dynamics for all simulated cell lines (n=200)')
hold off 
set(gcf,'position',[0,0,1600,400*n])
f = gcf; 
% Requires R2020a or later
text = sprintf("figures/%s_all_species_all_celllines.png", plot_id);
exportgraphics(f,text,'Resolution',300)

%% CLUSTER ANALYSIS OF A SET OF SPECIES

species_set = {'PTP', 'aCbl'};
species_color = {'r.', 'b.'};
plot_name = 'PTPvsaCbl';

%% Generate Dynamic Features 

all_dynamic_features = []; 
dynamic_feats_specie_idx = {};
all_celllines = {};

for i = 1:length(random_set)
    for si = 1:length(species_set)
        specie = species_set{si};
        cell_data = sim_data(i, "data").data;
        tbl = cell_data{1}; 
        col_data = table2array(tbl(:,specie));
        col_data = data_normalization(col_data, 2);
        cellline = string(sim_data(i, "cellline").cellline);
        dynamic_features = get_dynamic_features(col_data);
        % append new data 
        all_celllines = [all_celllines; cellline];
        all_dynamic_features = [all_dynamic_features; dynamic_features];
        dynamic_feats_specie_idx = [dynamic_feats_specie_idx; si];
    end 
end

all_dynamic_features = cell2mat(all_dynamic_features);

%% TSNE Dimensionality reduction 

rng(seed, "twister");
Y = tsne(all_dynamic_features);
idx = dynamic_feats_specie_idx;
idx = cell2mat(idx);
figure;
for i = 1:length(species_set)
    sname = species_set{i};
    color = species_color{i};
    plot(Y(idx==i,1), Y(idx==i,2), color, 'MarkerSize', 12) 
    hold on 
end 

legend(species_set);
title('TSNE Plot of the dynamic features', 'FontSize', 14)
subtitle('FGFR4 Model (n=200)')
% legend on

f = gcf; 
% Requires R2020a or later
text = sprintf("figures/SSET_%s_%s_tsne.png",plot_name, plot_id);
exportgraphics(f,text,'Resolution',300)

%% K means clustering 


%% plotting original

cellline_legend = {};
for i = 1:200
    cell_data = sim_data(i, "data").data;
    tbl = cell_data{1}; 
    cellline = string(sim_data(i, "cellline").cellline);
    col_data = tbl.pIGFR;
    plot(col_data, 'LineWidth',1)
    cellline_legend = [cellline_legend; cellline];
    hold on 
end

% legend(cellline_legend, "Visible","off");
grid on 
title('pIGFR Time Course of the FGFR4 Model', 'FontSize', 16)
xlabel('Time (Hours)', 'FontSize', 14)
ylabel('Expression', 'FontSize', 14)
hold off


f = gcf; 
% Requires R2020a or later
text = sprintf("figures/%s_dynamics_plot.png", plot_id);
exportgraphics(f,text,'Resolution',300)

%% generating dynamic features

all_celllines = {};
all_dynamic_features = [];

for i = 1:length(random_set)
    cell_data = sim_data(i, "data").data;
    tbl = cell_data{1}; 
    col_data = tbl.pIGFR;
    cellline = string(sim_data(i, "cellline").cellline);
    dynamic_features = get_dynamic_features(col_data);

    % append new data 
    all_celllines = [all_celllines; cellline];
    all_dynamic_features = [all_dynamic_features; dynamic_features];
end

all_dynamic_features = cell2mat(all_dynamic_features);

%% tsne analysis and plot

% apply tsne on dynamic features 

rng default
Y = tsne(all_dynamic_features);
gscatter(Y(:,1),Y(:,2),all_celllines)
title('TSNE Plot of the dynamic features', 'FontSize', 14)
subtitle('FGFR4 Model (n=200)')
legend off

f = gcf; 
% Requires R2020a or later
text = sprintf("figures/%s_tsne.png", plot_id);
exportgraphics(f,text,'Resolution',300)

%% kmeans cluster

opts = statset('Display','final');
[idx,C] = kmeans(Y,2, Options=opts);

%% plotting cluster on tsne plot 

X = Y;
figure;
plot(X(idx==1,1),X(idx==1,2),'r.','MarkerSize',12)
hold on
plot(X(idx==2,1),X(idx==2,2),'b.','MarkerSize',12)
plot(C(:,1),C(:,2),'kx',...
     'MarkerSize',15,'LineWidth',3) 
legend('Cluster 1','Cluster 2','Centroids',...
       'Location','NW')
title('Cluster Assignments and Centroids', 'FontSize', 14)
hold off

f = gcf; 
% Requires R2020a or later
text = sprintf("figures/%s_tsne_cluster.png", plot_id);
exportgraphics(f,text,'Resolution',300)

%% plotting cluster on dynamics plot 

cellline_legend = {};
for i = 1:200
    cell_data = sim_data(i, "data").data;
    tbl = cell_data{1}; 
    cellline = string(sim_data(i, "cellline").cellline);
    col_data = tbl.pIGFR;
    if idx(i) == 1
        subplot(1,2,1);
        plot(col_data, "Color", [1, 0, 0, 0.6], 'LineWidth', 1);
    elseif idx(i) == 2
        subplot(1,2,2);
        plot(col_data, "Color", [0, 0.5, 1, 0.6], 'LineWidth', 1);
    end 
    cellline_legend = [cellline_legend; cellline];
    hold on 
end

% legend(cellline_legend, "Visible","off"); 
subplot(1,2,1);
grid on;
title('cluster 1', 'FontSize', 16)
subtitle('Specie pIFGR of the FGFR4 Model (n=200)')
xlabel('Time (Hours)')
ylabel('Expression')
subplot(1,2,2);
grid on;
title('cluster 2', 'FontSize', 16)
subtitle('Specie pIFGR of the FGFR4 Model (n=200)')
xlabel('Time (Hours)')
ylabel('Expression')
hold off

sgtitle('Kmeans Clusters of Dynamic Features in time course', 'FontSize', 16)


set(gcf,'position',[0,0,800,400])
f = gcf; 

% Requires R2020a or later
text = sprintf("figures/%s_dynamics_plot_cluster.png", plot_id);
exportgraphics(f,text,'Resolution',300)


