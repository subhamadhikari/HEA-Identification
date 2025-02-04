import os
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def train_predict(model, X_train, y_train, X_test, y_test, print_metrics=True):
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
        print(f"rmse={rmse:.3f}, mae={mae:.3f}, r2={r2:.3f}, time={time_elapsed:.1f} s")
    metrics = {}
    metrics["rmse"] = rmse
    metrics["mae"] = mae
    metrics["r2"] = r2

    return model, y_pred, metrics

def perf_vs_size(model, X_pool, y_pool, X_test, y_test, csv_out,
                 overwrite=False,frac_list=None,n_frac = 15, n_run_factor=1):
    """
    Evaluate the performance of a machine learning model against varying training set sizes.

    This function measures how the model's performance (RMSE, MAE, and R-squared) changes 
    as the training dataset size is varied. It can output results to a CSV file and 
    can skip previously computed metrics to save computation time.

    Parameters:
    - model: A machine learning model from scikit-learn to be evaluated.
    - X_pool (pd.DataFrame): The feature dataset to be used for training.
    - y_pool (pd.Series): The target variable corresponding to the training dataset.
    - X_test (pd.DataFrame): The feature dataset to be used for testing.
    - y_test (pd.Series): The target variable corresponding to the testing dataset.
    - csv_out (str): The path to the CSV file where metrics will be saved.
    - overwrite (bool): If True, overwrite the existing CSV file. Default is False.
    - frac_list (list): A list of fractions representing the sizes of the training set as a fraction of the total. Default is None.
    - n_frac (int): The number of different training set sizes to evaluate. Default is 15.
    - n_run_factor (int): A factor to scale the number of runs for each training size. Default is 1.

    Returns:
    - pd.DataFrame: A DataFrame containing the computed metrics (RMSE, MAE, R-squared) 
      along with their standard deviations, indexed by the fraction of the training set size.
    """


    if os.path.exists(csv_out) and not overwrite:
        df = pd.read_csv(csv_out,index_col=0)

    else:
        df = pd.DataFrame(columns=['rmse','mae','r2','rmse_std','mae_std','r2_std'])

    if frac_list is None:
        frac_min = np.log10(100/X_pool.shape[0])
        frac_list = np.logspace(frac_min,0,n_frac)


    for frac in frac_list:
        skip = False
        # skip if frac is close to an existing frac
        for frac_ in df.index:
            if abs(frac - frac_)/frac_ < 0.25:
                skip = True
        if skip:
            continue

        if frac * X_pool.shape[0] < 80:
            continue

        # determine the number of runs based on frac
        if frac < 0.01:
            n_run = 20
        elif frac < 0.05:
            n_run = 10 # 20
        elif frac >= 0.05 and frac < 0.5:
            n_run = 6 #10
        elif frac >= 0.5 and frac < 1:
            n_run = 4 #5
        else:
            n_run = 1

        n_run = max(1, int(n_run * n_run_factor))

        print(f'frac={frac:.3f}, n_run={n_run}')

        metrics_ = {}
        for random_state_ in range(n_run):
            if frac == 1:
                X_train, y_train = X_pool, y_pool
            else:
                X_train, _, y_train, _ = train_test_split(X_pool, y_pool, train_size=frac,
                                                            random_state=random_state_  )
                
            _, _, metrics_[random_state_] = train_predict(model, X_train, y_train, X_test, y_test)

        metrics_ = pd.DataFrame(metrics_).transpose()[['rmse','mae','r2']]
        means = metrics_.mean(axis=0)
        std = metrics_.std(axis=0)
        std.index = [f'{col}_std' for col in std.index]

        # add metrics_.mean(axis=1) and metrics_.std(axis=1) to metrics[model_name]
        df.loc[frac] = pd.concat([means,std])
        print(df.loc[frac])
        # save the metrics
        df.sort_index().to_csv(csv_out, index_label='frac')

    return df

def plot_metrics_vs_size(metrics, metrics_name,id_train,
                         xlims=None, ylims=None,
                         figsize=(4,4),
                         ax_in=None,
                         fig_ax = None,
                         markers=None
                         ):
    """
    Plot the performance metrics of different models against the size of the training set.

    This function creates a plot displaying how a specified performance metric (e.g., RMSE, MAE, R-squared)
    varies as the size of the training set changes. It can handle multiple models and allows customization
    of the axes, markers, and figure size.

    Parameters:
    - metrics (dict): A dictionary where keys are model names and values are DataFrames containing metrics.
                      Each DataFrame should have the metrics name and its corresponding standard deviation.
    - metrics_name (str): The name of the metric to plot (e.g., 'rmse', 'mae', 'r2').
    - id_train (list): The list of training data identifiers, used to determine the size of the training set.
    - xlims (list): Optional; limits for the x-axis (training set fraction).
    - ylims (list): Optional; limits for the y-axis (metric value).
    - figsize (tuple): Size of the figure (width, height) in inches. Default is (4, 4).
    - ax_in (matplotlib.axes.Axes): Optional; if provided, use this Axes object for plotting instead of creating a new one.
    - fig_ax (tuple): Optional; a tuple of (figure, Axes) to use for plotting.
    - markers (dict): Optional; a dictionary mapping model names to marker styles for the plot. If None, default markers are used.

    Returns:
    - tuple: A tuple containing the figure and Axes objects used for the plot.
    """
    if fig_ax is not None:
        fig, ax = fig_ax
    elif ax_in is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax_in.get_figure()
        ax = ax_in.twinx()


    if markers is None:
        markers = {'RF':'o',
                   'XGB':'s'}


    for model_name in metrics.keys():
        ax.errorbar(metrics[model_name].index, metrics[model_name][metrics_name],
                    yerr=metrics[model_name][f'{metrics_name}_std'],
                    fmt=f'-{markers[model_name]}',markersize=5,label=model_name, capsize=3)

    if xlims is None:
        xlims = [100/len(id_train)*0.9, 1]
    ax.set_xlim(xlims)

    if ylims is None:
        ylims = [0.4, 1]
    ax.set_ylim(ylims)



    ax.set_xscale('log')
    ax.set_xlabel('Fraction of the full training set')

    # add the upper x axis for the number of training data
    ax2 = ax.twiny()
    xlims = ax.get_xlim()
    ax2.set_xlim([xlims[0]*len(id_train), xlims[1]*len(id_train)])
    ax2.set_xscale('log')
    ax2.set_xlabel('Training set size')
    ax2.tick_params(axis='x', which='major', pad=0)

    if metrics_name == 'r2':
        ax.set_ylabel('$R^2$')
    else:
        ax.set_ylabel(f'{metrics_name.upper()} (eV/atom)')

    if ax_in is None:
        ax.legend(loc='upper center')
        ax.grid(linewidth=0.1)
    return fig, ax