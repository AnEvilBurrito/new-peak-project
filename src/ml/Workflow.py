from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from tqdm import tqdm
import pandas as pd
import numpy as np

def build_pipeline(model, scale=False):
    steps = [('imputer', SimpleImputer(strategy='mean'))]
    if scale:
        steps.append(('scaler', StandardScaler()))
    steps.append(('model', model))
    return Pipeline(steps)



def evaluate_model(model, model_name, feature_data, feature_data_name, target_data, test_size=0.2, random_state=4):
    # Align rows between X and y
    common_idx = feature_data.index.intersection(target_data.index)
    X = feature_data.loc[common_idx]
    y = target_data.loc[common_idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        'Model': model_name, 
        'Feature Data': feature_data_name,
        'Mean Squared Error': mean_squared_error(y_test, y_pred),
        'R2 Score': r2_score(y_test, y_pred),
        'Pearson Correlation': pearsonr(y_test, y_pred)[0],
        'Pearson P-Value': pearsonr(y_test, y_pred)[1]
    }


def batch_eval_standard(feature_data_list, feature_data_names, target_data, target_name, 
                     num_repeats=10,
                     test_size=0.2,
                     o_random_seed=42, 
                     n_jobs=-1):
    """
    Evaluate multiple models on multiple feature datasets against one target dataset.
    
    Parameters:
    - feature_data_list: List of DataFrames containing feature data.
    - feature_data_names: List of names corresponding to each feature dataset.
    - target_data: DataFrame containing the target variable.
    - target_name: Name of the target variable column in target_data.
    - o_random_seed: Random seed for reproducibility.
    
    Returns:
    - DataFrame containing evaluation metrics for each model and feature dataset.
    """
    
    if not isinstance(feature_data_list, list) or not isinstance(feature_data_names, list):
        raise ValueError("feature_data_list and feature_data_names must be lists.")
    
    if len(feature_data_list) != len(feature_data_names):
        raise ValueError("feature_data_list and feature_data_names must have the same length.")
    
    if target_name not in target_data.columns:
        raise ValueError(f"Target name '{target_name}' not found in target_data columns.")
    
    all_models = [
        build_pipeline(LinearRegression()),
        build_pipeline(RandomForestRegressor(n_estimators=100, random_state=o_random_seed)),
        build_pipeline(GradientBoostingRegressor(n_estimators=100, random_state=o_random_seed)),
        build_pipeline(SVR(max_iter=10000), scale=True),
        build_pipeline(MLPRegressor(hidden_layer_sizes=(20,), max_iter=10000, random_state=o_random_seed), scale=True)
    ]

    all_models_desc = ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'Support Vector Machine', 'Neural Network']
    zipped_model_data = list(zip(all_models, all_models_desc))
    all_features = feature_data_list
    all_features_desc = feature_data_names
    zipped_feature_data = list(zip(all_features, all_features_desc))

    # random states are rand ints between 0 and 10000, for n values 
    np.random.seed(o_random_seed)
    n_random = num_repeats
    all_random_states = np.random.randint(0, 10000, n_random)
    metric_data = []
    if n_jobs == 1:          
        for (feature_data, feature_data_name) in tqdm(zipped_feature_data):
            # print('Feature Data:', feature_data_name)
            # print('Feature Data Shape:', feature_data.shape)
            for (model, model_name) in zipped_model_data:
                # print('Model:', model_name)
                for rand in all_random_states:
                    metrics = evaluate_model(model, model_name, feature_data, feature_data_name, target_data[target_name], random_state=rand, test_size=test_size)
                    metric_data.append(metrics)
                    
    else:        
        # parallelise the model evaluation process using joblib
        from joblib import Parallel, delayed

        metric_data = Parallel(n_jobs=-1)(delayed(evaluate_model)(model, model_name, feature_data, feature_data_name, target_data[target_name], random_state=rand, test_size=test_size) 
                                        for (feature_data, feature_data_name) in zipped_feature_data
                                        for (model, model_name) in zipped_model_data
                                        for rand in all_random_states)

    # make a dataframe of the metric data
    metric_df = pd.DataFrame(metric_data)
    return metric_df