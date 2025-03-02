from collections.abc import Iterable

import pandas as pd

from ..forecasting.forecasting import forecast
from fcst.metrics import mape
from ..common.types import ModelDict, ModelResults


def get_backtest_periods(
    series: pd.Series, backtest_periods: int
) -> Iterable[pd.Series]:
    """Generates each time series for back-test"""

    data_end_date = series.index.max()

    for i in range(1, backtest_periods + 1):
        yield series.loc[series.index <= data_end_date - i]


def backtest_evaluate(
    series: pd.Series,
    models: ModelDict,
    backtest_periods: int = 5,
    eval_periods: int = 2,
    min_data_points: int = 8,
) -> ModelResults:
    """Rolling back-test the series with multiple BaseForecaster models

    Parameters
    ----------
    series (pd.Series): Pandas Series of floats
        Preprocessed, sorted, and filtered time series.
        It's assumed that the series has all the months,
        and ends with the `data_date` you want to train.
        This Series should come from the preprocessing step.

    models (ModelDict): Model dictionary
        The keys are model names and
        the values are the forecaster models from `sktime`.

    backtest_periods (int): Number of periods to back-test (Default is 3)

    eval_periods (int): Number of periods to evaluate in each rolling back-test (Default is 2)

    min_data_points (int): Minimum data points in the series to perform back-testing

    Returns
    -------
    ModelResults: A dictionary reporting the average error of each model
    """

    if len(series) == 0:
        raise ValueError("`series` must have more than 0 length for back-testing.")
    
    models = models.copy()

    if len(series) < min_data_points:
        return {"MeanDefault": 1.}

    true_series = series.copy()
    actual_col = "actual"
    fcst_col = "forecast"

    true_series = true_series.rename(actual_col)
    model_results = {}

    for model_name, model in models.items():
        eval_results = []
        try:  # Try backtesting for the model
            for backtest_series in get_backtest_periods(series, backtest_periods):
                test_output = forecast(model, backtest_series, periods=eval_periods)
                df_eval = pd.concat([test_output, true_series], axis=1, join="inner")
                eval_results.append(df_eval)

            df_eval = pd.concat(eval_results)

            model_results[model_name] = mape(
                df_eval[actual_col], df_eval[fcst_col], symmetric=True
            )
        except Exception:  # Skip the failed model
            continue

    # In case, it skips all the models, use "Mean" as the fallback
    if not model_results:
        return {"MeanDefault": 1.}

    return dict(sorted(model_results.items(), key=lambda x: x[1]))
