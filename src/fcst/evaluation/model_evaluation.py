from collections.abc import Iterable
from typing import Literal, Tuple, overload

import pandas as pd

from fcst.metrics import mape

from ..common.types import ModelDict, ModelResults
from ..forecasting.forecasting import forecast


def get_backtest_periods(
    series: pd.Series, backtest_periods: int
) -> Iterable[pd.Series]:
    """Generates each time series for back-test"""

    data_end_date = series.index.max()

    for i in range(1, backtest_periods + 1):
        yield series.loc[series.index <= data_end_date - i]


@overload
def backtest_evaluate(
    series: pd.Series,
    models: ModelDict,
    backtest_periods: int = 5,
    eval_periods: int = 2,
    min_data_points: int = 8,
    return_results: bool = False,
    eval_method: Literal["rolling", "one-time"] = "rolling",
) -> ModelResults: ...


@overload
def backtest_evaluate(
    series: pd.Series,
    models: ModelDict,
    backtest_periods: int = 5,
    eval_periods: int = 2,
    min_data_points: int = 8,
    return_results: bool = True,
    eval_method: Literal["rolling", "one-time"] = "rolling",
) -> Tuple[ModelResults, pd.DataFrame]: ...


def backtest_evaluate(
    series: pd.Series,
    models: ModelDict,
    backtest_periods: int = 5,
    eval_periods: int = 2,
    min_data_points: int = 8,
    return_results: bool = False,
    eval_method: Literal["rolling", "one-time"] = "rolling",
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
            If `eval_method`=="one-time", this argument is ignored, and `backtest_periods` will be used instead.

        min_data_points (int): Minimum data points in the series to perform back-testing

        return_results (bool): Whether or not to return the back-testing raw results (Default is False)

        eval_method ("rolling" or "one-time"): The method to evaluate back-testing (Default="rolling")

    Returns
    -------
        ModelResults: A dictionary reporting the average error of each model (when `return_results` = False)

        Tuple[ModelResults, pd.DataFrame]: The results dictionary along with the rolling back-test in each period of each model (when `return_results` = True)
    """

    if len(series) == 0:
        raise ValueError("`series` must have more than 0 length for back-testing.")

    if eval_method not in ("one-time", "rolling"):
        raise ValueError("`eval_method` must be either 'one-time' or 'rolling'.")

    models = models.copy()

    if len(series) < min_data_points:
        model_results = {"MeanDefault": 1.0}

        if return_results:
            return model_results, pd.DataFrame()

        return model_results

    true_series = series.copy()
    actual_col = "actual"
    fcst_col = "forecast"

    true_series = true_series.rename(actual_col)
    model_results = {}

    all_eval = []

    def _get_eval_df(
        backtest_series: pd.Series,
        true_series: pd.Series,
        model,
        model_name,
        eval_periods: int,
    ):
        backtest_data_date = backtest_series.index.max()
        test_output = forecast(model, backtest_series, periods=eval_periods)
        df_eval = pd.concat([test_output, true_series], axis=1, join="inner")
        df_eval["backtest_data_date"] = backtest_data_date
        df_eval["model_name"] = model_name
        return df_eval

    for model_name, model in models.items():
        eval_results = []
        try:  # Try backtesting for the model
            if eval_method == "rolling":
                for backtest_series in get_backtest_periods(series, backtest_periods):
                    # backtest_data_date = backtest_series.index.max()
                    # test_output = forecast(model, backtest_series, periods=eval_periods)
                    # df_eval = pd.concat(
                    #     [test_output, true_series], axis=1, join="inner"
                    # )
                    # df_eval["backtest_data_date"] = backtest_data_date
                    # df_eval["model_name"] = model_name

                    df_eval = _get_eval_df(
                        backtest_series=backtest_series,
                        true_series=true_series,
                        model=model,
                        model_name=model_name,
                        eval_periods=eval_periods,
                    )
                    eval_results.append(df_eval)

            elif eval_method == "one-time":
                _data_end_date = series.index.max()
                backtest_series = series.loc[
                    series.index <= _data_end_date - backtest_periods
                ]
                
                df_eval = _get_eval_df(
                    backtest_series=backtest_series,
                    true_series=true_series,
                    model=model,
                    model_name=model_name,
                    eval_periods=backtest_periods,  # Use backtest periods instead
                )
                
                eval_results.append(df_eval)

        except Exception:  # Skip the failed model
            continue

        df_eval = pd.concat(eval_results)

        # Append results
        model_results[model_name] = mape(
            df_eval[actual_col], df_eval[fcst_col], symmetric=True
        )

        if return_results:
            all_eval.append(df_eval)

    # In case, it skips all the models, use "Mean" as the fallback
    if not model_results:
        model_results = {"MeanDefault": 1.0}

    model_results = dict(sorted(model_results.items(), key=lambda x: x[1]))

    if return_results:
        if all_eval:
            df_eval_results = pd.concat(all_eval)
        else:
            df_eval_results = pd.DataFrame()

        return model_results, df_eval_results

    return model_results
