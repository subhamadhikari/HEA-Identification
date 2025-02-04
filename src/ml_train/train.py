import os

import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


random_state = 1

# Define machine learning models using Pipelines
pipe = {
    "RF": Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=100,
                    bootstrap=False,
                    max_features=1 / 3,
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    ),
    "XGB": Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                xgb.XGBRegressor(
                    n_estimators=500,
                    learning_rate=0.4,
                    reg_lambda=0.01,
                    reg_alpha=0.1,
                    colsample_bytree=0.5,
                    colsample_bylevel=0.7,
                    num_parallel_tree=6,
                    tree_method="gpu_hist",
                    gpu_id=0,
                ),
            ),
        ]
    ),
}


def train_interpolation(
    X_all, y_all, train_idx, test_idx, scope_name, csv_dir, overwrite=False
):
    """
    Trains interpolation models (Random Forest and XGBoost) and evaluates their performance.

    Parameters:
    X_all (pd.DataFrame): The complete feature set.
    y_all (pd.Series): The complete target variable.
    train_idx (array-like): Indices for training data selection.
    test_idx (array-like): Indices for test data selection.
    scope_name (str): Identifier for the dataset scope (e.g., experiment name).
    csv_dir (str): Directory where CSV output files will be saved.
    overwrite (bool): If True, overwrite existing CSV files with the same name. Default is False.

    Returns:
    dict: A dictionary containing the performance metrics for each model.
    """
    X_pool, X_test, y_pool, y_test = train_test_split(
        X_all.loc[train_idx],
        y_all.loc[train_idx],
        test_size=0.2,
        random_state=random_state,
    )

    # save the divided datasets
    # train_csv = pd.concat([X_test,y_test], axis=1)
    # train_csv.to_csv('interpolate_test.csv', index=False)
    # print(train_csv)
    # return

    metrics = {}

    # Iterate over each model defined in the pipeline
    for model_name in pipe.keys():
        csv_out = f"{csv_dir}/size_effect_rand_split_{scope_name}_{model_name}.csv"

        # Check if the metrics CSV already exists and should not be overwritten
        if os.path.exists(csv_out) and not overwrite:
            # Read the existing results
            metrics[model_name] = pd.read_csv(csv_out, index_col=0)
            mad = get_mad_std(y_test)
            metrics[model_name]["mae/mad"] = metrics[model_name]["mae"] / mad
            metrics[model_name]["mae/mad_std"] = metrics[model_name]["mae_std"] / mad
            continue

        metrics[model_name] = perf_vs_size(
            pipe[model_name], X_pool, y_pool, X_test, y_test, csv_out, overwrite
        )

        # Print performance of the full model
        mae = metrics[model_name].iloc[-1]["mae"]
        r2 = metrics[model_name].iloc[-1]["r2"]
        print(f"{scope_name} {model_name} {mae} {r2}")

    return metrics

def train_extrapolation(X_all, y_all, train_idx, test_idx, scope_name, csv_dir, overwrite=False):
    """
    Train models for extrapolation on the provided dataset and evaluate their performance.

    This function takes in feature and target datasets, splits them into training and testing sets,
    and then trains specified models to predict the target variable based on the features.
    It evaluates the performance using Mean Absolute Deviation (MAD) and stores the metrics in a CSV file.

    Parameters:
    - X_all (pd.DataFrame): The complete feature dataset.
    - y_all (pd.Series): The complete target variable dataset.
    - train_idx (array-like): Indices of the training set.
    - test_idx (array-like): Indices of the testing set.
    - scope_name (str): Name of the scope used for reporting and file naming.
    - csv_dir (str): Directory to save the performance metrics CSV files.
    - overwrite (bool): If True, overwrite existing CSV files. Default is False.

    Returns:
    - dict: A dictionary containing performance metrics for each model.
    """

    # Split the data into training and testing sets based on the provided indices
    X_pool, y_pool = X_all.loc[train_idx], y_all.loc[train_idx]
    X_test, y_test = X_all.loc[test_idx], y_all.loc[test_idx]


    # Save the test set to a CSV file for reference
    # train_csv = pd.concat([X_test,y_test], axis=1)
    # train_csv.to_csv('extrapolate_test.csv', index=False)

    mad = get_mad_std(y_test)
    print(f'{scope_name} MAD of the test set: {mad}')

    metrics = {}
    for model_name in pipe.keys():
        csv_out = f'{csv_dir}/size_effect_{scope_name}_{model_name}.csv'
        if os.path.exists(csv_out) and not overwrite:
            print("skipped!")
            continue  # Skip if the file exists

        metrics[model_name] = perf_vs_size(pipe[model_name], X_pool, y_pool, X_test, y_test, csv_out, overwrite)

        # Print performance of the full model
        mae = metrics[model_name].iloc[-1]['mae']
        r2 = metrics[model_name].iloc[-1]['r2']
        print(f'{scope_name} {model_name} {mae} {r2}')

    return metrics