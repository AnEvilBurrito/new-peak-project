% function which obtains dynamic features from a time series array 

% Current dynamic features: 
% 'auc', 'median', 'tfc', 'tmax', 'max', 'tmin', 'min', 'ttsv', 'tsv', 'init'

function dynamic_features = get_dynamic_features(col_data)
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
        dynamic_features = {n_auc, median_val, tfc, n_max_time, max_val, n_min_time, min_val, n_tsv, tsv_value, start_val};
end 