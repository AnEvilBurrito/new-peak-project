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
from sklearn.base import BaseEstimator, TransformerMixin


class ClippingTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that clips extreme values to prevent float32 overflow in sklearn.
    Useful for datasets with extreme values from parameter distortion experiments.
    """
    def __init__(self, threshold=1e9):
        self.threshold = threshold
        
    def fit(self, X, y=None):
        # No fitting needed
        return self
        
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_clipped = X.copy()
            for col in X.columns:
                if pd.api.types.is_numeric_dtype(X[col]):
                    X_clipped[col] = X_clipped[col].clip(lower=-self.threshold, upper=self.threshold)
            return X_clipped
        else:
            # numpy array
            return np.clip(X, -self.threshold, self.threshold)
            
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def build_pipeline(model, scale=False, clip_threshold=None):
    """
    Build a scikit-learn pipeline with optional clipping, imputation, scaling, and model.
    
    Args:
        model: The estimator to use as the final step
        scale: Whether to include StandardScaler
        clip_threshold: If provided, clip values to ±threshold to prevent float32 overflow
    """
    steps = []
    
    # Add clipping if threshold provided
    if clip_threshold is not None:
        steps.append(('clipper', ClippingTransformer(threshold=clip_threshold)))
    
    steps.append(('imputer', SimpleImputer(strategy='mean')))
    
    if scale:
        steps.append(('scaler', StandardScaler()))
    
    steps.append(('model', model))
    return Pipeline(steps)


def build_pipeline_with_clipping(model, scale=False, threshold=1e9):
    """Convenience function for building pipeline with clipping"""
    return build_pipeline(model, scale=scale, clip_threshold=threshold)



def evaluate_model(model, model_name, feature_data, feature_data_name, target_data, test_size=0.2, random_state=42):
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


def batch_eval(feature_data_list, feature_data_names, target_data, target_name, 
               all_models, all_models_desc,
               num_repeats=10,
               test_size=0.2,
               o_random_seed=42, 
               n_jobs=-1):
    """
    Evaluate multiple custom models on multiple feature datasets against one target dataset.
    
    Parameters:
    - feature_data_list: List of DataFrames containing feature data.
    - feature_data_names: List of names corresponding to each feature dataset.
    - target_data: DataFrame containing the target variable.
    - target_name: Name of the target variable column in target_data.
    - all_models: List of model/pipeline objects to evaluate.
    - all_models_desc: List of descriptive names corresponding to each model.
    - num_repeats: Number of random train/test splits to evaluate (default: 10).
    - test_size: Proportion of data to use for testing (default: 0.2).
    - o_random_seed: Random seed for reproducibility of random states (default: 42).
    - n_jobs: Number of parallel jobs to run (-1 for all available cores, 1 for serial).
    
    Returns:
    - DataFrame containing evaluation metrics for each model and feature dataset.
    """
    
    if not isinstance(feature_data_list, list) or not isinstance(feature_data_names, list):
        raise ValueError("feature_data_list and feature_data_names must be lists.")
    
    if len(feature_data_list) != len(feature_data_names):
        raise ValueError("feature_data_list and feature_data_names must have the same length.")
    
    if target_name not in target_data.columns:
        raise ValueError(f"Target name '{target_name}' not found in target_data columns.")
    
    if not isinstance(all_models, list) or not isinstance(all_models_desc, list):
        raise ValueError("all_models and all_models_desc must be lists.")
    
    if len(all_models) != len(all_models_desc):
        raise ValueError("all_models and all_models_desc must have the same length.")
    
    if len(all_models) == 0:
        raise ValueError("all_models list cannot be empty.")
    
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
            for (model, model_name) in zipped_model_data:
                for rand in all_random_states:
                    metrics = evaluate_model(model, model_name, feature_data, feature_data_name, target_data[target_name], random_state=rand, test_size=test_size)
                    metric_data.append(metrics)
                    
    else:        
        # parallelise the model evaluation process using joblib
        from joblib import Parallel, delayed

        metric_data = Parallel(n_jobs=n_jobs)(delayed(evaluate_model)(model, model_name, feature_data, feature_data_name, target_data[target_name], random_state=rand, test_size=test_size) 
                                        for (feature_data, feature_data_name) in zipped_feature_data
                                        for (model, model_name) in zipped_model_data
                                        for rand in all_random_states)

    # make a dataframe of the metric data
    metric_df = pd.DataFrame(metric_data)
    return metric_df


def _validate_paired_inputs(feature_data_list, feature_data_names,
                           target_data_list, target_name_list):
    """
    Validate 1:1 mapping between features and targets for paired evaluation.
    
    Parameters:
    - feature_data_list: List of feature DataFrames
    - feature_data_names: List of feature names
    - target_data_list: List of target DataFrames
    - target_name_list: List of target column names
    
    Raises:
    - ValueError if inputs are invalid
    """
    if not isinstance(feature_data_list, list) or not isinstance(feature_data_names, list):
        raise ValueError("feature_data_list and feature_data_names must be lists.")
    
    if not isinstance(target_data_list, list) or not isinstance(target_name_list, list):
        raise ValueError("target_data_list and target_name_list must be lists.")
    
    if len(feature_data_list) != len(feature_data_names):
        raise ValueError("feature_data_list and feature_data_names must have the same length.")
    
    if len(target_data_list) != len(target_name_list):
        raise ValueError("target_data_list and target_name_list must have the same length.")
    
    if len(feature_data_list) != len(target_data_list):
        raise ValueError("feature_data_list and target_data_list must have the same length for 1:1 pairing.")
    
    # Validate each target column exists in corresponding target DataFrame
    for i, (target_df, target_name) in enumerate(zip(target_data_list, target_name_list)):
        if target_name not in target_df.columns:
            raise ValueError(f"Target name '{target_name}' not found in target_data_list[{i}] columns: {list(target_df.columns)}")


def batch_eval_pairs(feature_data_list, feature_data_names,
                    target_data_list, target_name_list,
                    all_models, all_models_desc,
                    num_repeats=10, test_size=0.2,
                    o_random_seed=42, n_jobs=-1):
    """
    Evaluate multiple custom models on paired feature-target datasets (1:1 mapping).
    
    Each feature dataset is evaluated against its corresponding target dataset.
    
    Parameters:
    - feature_data_list: List of DataFrames containing feature data
    - feature_data_names: List of names for each feature dataset
    - target_data_list: List of DataFrames containing target data (1:1 with features)
    - target_name_list: List of target column names (1:1 with features)
    - all_models: List of model/pipeline objects to evaluate
    - all_models_desc: List of descriptive names for each model
    - num_repeats: Number of random train/test splits
    - test_size: Proportion for testing
    - o_random_seed: Random seed for reproducibility
    - n_jobs: Number of parallel jobs (-1 for all cores, 1 for serial)
    
    Returns:
    - DataFrame with evaluation metrics for each (model, feature, target) combination
    """
    
    # Validate paired inputs
    _validate_paired_inputs(feature_data_list, feature_data_names,
                           target_data_list, target_name_list)
    
    if not isinstance(all_models, list) or not isinstance(all_models_desc, list):
        raise ValueError("all_models and all_models_desc must be lists.")
    
    if len(all_models) != len(all_models_desc):
        raise ValueError("all_models and all_models_desc must have the same length.")
    
    if len(all_models) == 0:
        raise ValueError("all_models list cannot be empty.")
    
    zipped_model_data = list(zip(all_models, all_models_desc))
    
    # random states are rand ints between 0 and 10000, for n values 
    np.random.seed(o_random_seed)
    n_random = num_repeats
    all_random_states = np.random.randint(0, 10000, n_random)
    metric_data = []
    
    # Create paired feature-target tuples
    paired_data = list(zip(feature_data_list, feature_data_names, target_data_list, target_name_list))
    
    if n_jobs == 1:          
        for feature_data, feature_name, target_data, target_name in tqdm(paired_data):
            for model, model_name in zipped_model_data:
                for rand in all_random_states:
                    metrics = evaluate_model(model, model_name, feature_data, feature_name, 
                                           target_data[target_name], random_state=rand, test_size=test_size)
                    metric_data.append(metrics)
                    
    else:        
        # parallelise the model evaluation process using joblib
        from joblib import Parallel, delayed
        
        # Define function for parallel evaluation of a single pair
        def evaluate_pair(feature_data, feature_name, target_data, target_name, model, model_name, rand):
            return evaluate_model(model, model_name, feature_data, feature_name, 
                                target_data[target_name], random_state=rand, test_size=test_size)
        
        metric_data = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_pair)(feature_data, feature_name, target_data, target_name, model, model_name, rand)
            for feature_data, feature_name, target_data, target_name in paired_data
            for model, model_name in zipped_model_data
            for rand in all_random_states
        )

    # make a dataframe of the metric data
    metric_df = pd.DataFrame(metric_data)
    return metric_df


def batch_eval_standard(feature_data_list, feature_data_names, target_data, target_name, 
                     num_repeats=10,
                     test_size=0.2,
                     o_random_seed=42, 
                     n_jobs=-1,
                     clip_threshold=1e9):
    """
    Evaluate multiple standard models on multiple feature datasets against one target dataset.
    
    Parameters:
    - feature_data_list: List of DataFrames containing feature data.
    - feature_data_names: List of names corresponding to each feature dataset.
    - target_data: DataFrame containing the target variable.
    - target_name: Name of the target variable column in target_data.
    - num_repeats: Number of random train/test splits to evaluate (default: 10).
    - test_size: Proportion of data to use for testing (default: 0.2).
    - o_random_seed: Random seed for reproducibility (default: 42).
    - n_jobs: Number of parallel jobs to run (-1 for all available cores, 1 for serial).
    - clip_threshold: Clip feature values to ±threshold to prevent float32 overflow (default: 1e9).
    
    Returns:
    - DataFrame containing evaluation metrics for each model and feature dataset.
    """
    
    # Define the standard set of models with clipping to prevent float32 overflow
    all_models = [
        build_pipeline(LinearRegression(), clip_threshold=clip_threshold),
        build_pipeline(RandomForestRegressor(n_estimators=100, random_state=o_random_seed), clip_threshold=clip_threshold),
        build_pipeline(GradientBoostingRegressor(n_estimators=100, random_state=o_random_seed), clip_threshold=clip_threshold),
        build_pipeline(SVR(max_iter=10000), scale=True, clip_threshold=clip_threshold),
        build_pipeline(MLPRegressor(hidden_layer_sizes=(20,), max_iter=10000, random_state=o_random_seed), scale=True, clip_threshold=clip_threshold)
    ]

    all_models_desc = ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'Support Vector Machine', 'Neural Network']
    
    # Use the new batch_eval function with standard models
    return batch_eval(
        feature_data_list=feature_data_list,
        feature_data_names=feature_data_names,
        target_data=target_data,
        target_name=target_name,
        all_models=all_models,
        all_models_desc=all_models_desc,
        num_repeats=num_repeats,
        test_size=test_size,
        o_random_seed=o_random_seed,
        n_jobs=n_jobs
    )


def batch_eval_standard_pairs(feature_data_list, feature_data_names,
                             target_data_list, target_name_list,
                             num_repeats=10, test_size=0.2,
                             o_random_seed=42, n_jobs=-1,
                             clip_threshold=1e9):
    """
    Evaluate standard models on paired feature-target datasets.
    Uses the standard set of models (Linear Regression, Random Forest, etc.)
    
    Parameters:
    - feature_data_list: List of DataFrames containing feature data
    - feature_data_names: List of names for each feature dataset
    - target_data_list: List of DataFrames containing target data (1:1 with features)
    - target_name_list: List of target column names (1:1 with features)
    - num_repeats: Number of random train/test splits
    - test_size: Proportion for testing
    - o_random_seed: Random seed for reproducibility
    - n_jobs: Number of parallel jobs (-1 for all cores, 1 for serial)
    - clip_threshold: Clip feature values to ±threshold to prevent float32 overflow
    
    Returns:
    - DataFrame with evaluation metrics for each (model, feature, target) combination
    """
    
    # Define the standard set of models with clipping to prevent float32 overflow
    all_models = [
        build_pipeline(LinearRegression(), clip_threshold=clip_threshold),
        build_pipeline(RandomForestRegressor(n_estimators=100, random_state=o_random_seed), clip_threshold=clip_threshold),
        build_pipeline(GradientBoostingRegressor(n_estimators=100, random_state=o_random_seed), clip_threshold=clip_threshold),
        build_pipeline(SVR(max_iter=10000), scale=True, clip_threshold=clip_threshold),
        build_pipeline(MLPRegressor(hidden_layer_sizes=(20,), max_iter=10000, random_state=o_random_seed), scale=True, clip_threshold=clip_threshold)
    ]

    all_models_desc = ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'Support Vector Machine', 'Neural Network']
    
    # Use the batch_eval_pairs function with standard models
    return batch_eval_pairs(
        feature_data_list=feature_data_list,
        feature_data_names=feature_data_names,
        target_data_list=target_data_list,
        target_name_list=target_name_list,
        all_models=all_models,
        all_models_desc=all_models_desc,
        num_repeats=num_repeats,
        test_size=test_size,
        o_random_seed=o_random_seed,
        n_jobs=n_jobs
    )
