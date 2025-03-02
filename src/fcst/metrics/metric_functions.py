import numpy as np
import pandas as pd


def mape(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    symmetric: bool = False,
):
    """Returns mean absolute percentage error (MAPE) or symmetric version"""

    abs_error = np.fabs(y_true - y_pred)

    if symmetric:
        arr_metric = (2 * abs_error) / (y_true + y_pred)
        arr_metric[(y_true == 0) | (y_pred == 0)] = 1
        arr_metric[(y_true == 0) & (y_pred == 0)] = 0

    else:
        arr_metric = abs_error / y_true
        arr_metric[(y_true == 0) & (y_pred != 0)] = 1
        arr_metric[(y_true == 0) & (y_pred == 0)] = 0

    return arr_metric.mean()
