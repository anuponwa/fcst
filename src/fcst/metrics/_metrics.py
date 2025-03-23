import numpy as np
import pandas as pd


def mae(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
):
    """Returns mean absolute error (MAE)"""

    abs_error = np.fabs(y_true - y_pred)

    return abs_error.mean()


def mape(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
):
    """Returns mean absolute percentage error (MAPE)"""

    abs_error = np.fabs(y_true - y_pred)

    arr_metric = abs_error / y_true
    arr_metric[(y_true == 0) & (y_pred != 0)] = 1
    arr_metric[(y_true == 0) & (y_pred == 0)] = 0

    return arr_metric.mean()


def smape(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
):
    """Returns symmetric mean absolute percentage error (SMAPE)"""

    abs_error = np.fabs(y_true - y_pred)

    arr_metric = (2 * abs_error) / (y_true + y_pred)
    arr_metric[(y_true == 0) | (y_pred == 0)] = 1
    arr_metric[(y_true == 0) & (y_pred == 0)] = 0

    return arr_metric.mean()


def mse(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
):
    """Returns mean squared error (MSE)"""

    error_sq = np.square(y_true - y_pred)

    return np.mean(error_sq)


def rmse(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
):
    """Returns root mean squared error (RMSE)"""

    return np.sqrt(mse(y_true, y_pred))


# Row-wise

def mae_row(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
):
    """Returns row-wise absolute error"""

    abs_error = np.fabs(y_true - y_pred)

    return abs_error


def mape_row(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
):
    """Returns row-wise mean absolute percentage error (MAPE)"""

    abs_error = np.fabs(y_true - y_pred)

    arr_metric = abs_error / y_true
    arr_metric[(y_true == 0) & (y_pred != 0)] = 1
    arr_metric[(y_true == 0) & (y_pred == 0)] = 0

    return arr_metric


def smape_row(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
):
    """Returns row-wise symmetric mean absolute percentage error (SMAPE)"""

    abs_error = np.fabs(y_true - y_pred)

    arr_metric = (2 * abs_error) / (y_true + y_pred)
    arr_metric[(y_true == 0) | (y_pred == 0)] = 1
    arr_metric[(y_true == 0) & (y_pred == 0)] = 0

    return arr_metric