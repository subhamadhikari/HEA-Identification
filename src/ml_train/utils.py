import time

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_predict(model, X_train, y_train, X_test, y_test,print_metrics=True):
    """
    Train the given model on the training dataset and predict the target variable on the test dataset.

    This function measures the time taken to fit the model, predicts the target variable for the test set,
    and computes evaluation metrics including RMSE, MAE, and R-squared. It can optionally print the metrics to the console.

    Parameters:
    - model: The machine learning model to be trained and evaluated.
    - X_train (pd.DataFrame): The training feature dataset.
    - y_train (pd.Series): The training target variable dataset.
    - X_test (pd.DataFrame): The testing feature dataset.
    - y_test (pd.Series): The testing target variable dataset.
    - print_metrics (bool): If True, print the computed metrics. Default is True.

    Returns:
    - tuple: A tuple containing the trained model, the predicted values for the test set, and a dictionary of metrics.
        - model: The trained model after fitting.
        - y_pred (pd.DataFrame): Predicted values for the test set.
        - metrics (dict): A dictionary containing RMSE, MAE, and R-squared values.
    """
    # record the time
    time_init = time.time()

    # fit the model on the training set
    model.fit(X_train, y_train)
    # predict the test set
    y_pred = pd.DataFrame(model.predict(X_test), index=y_test.index)

    # record the time
    time_elapsed = time.time() - time_init

    # Metrics
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    if print_metrics == True:
        print(f'rmse={rmse:.3f}, mae={mae:.3f}, r2={r2:.3f}, time={time_elapsed:.1f} s')
    metrics = {}
    metrics['rmse'] = rmse
    metrics['mae'] = mae
    metrics['r2'] = r2

    return model, y_pred, metrics