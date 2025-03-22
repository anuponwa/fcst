from collections.abc import Iterable
from typing import Tuple, overload

import pandas as pd

from fcst.metrics import smape

from ..common.types import ModelDict, ModelResults
from ..forecasting.forecasting import forecast


def get_backtest_periods(
    series: pd.Series,
    backtest_periods: int,
    eval_periods: int,
    keep_eval_fixed: bool = False,
) -> Iterable[pd.Series]:
    """Generates each time series for back-test training splits

    Parameters
    ----------
        series (pd.Series): Time-series with PeriodIndex as index
            The series must be resampled and missing values have been filled

        backtest_periods (int): The number of rolling back-test periods

        eval_periods (int): The number of evaluation periods in each rolling

        keep_eval_fixed (bool): Whether or not to keep eval_periods fixed (Default = False)

    Returns
    -------
        Iterable[pd.Series]: Iterable of training time-series for rolling back-testing
    """

    data_end_date = series.index.max()

    # Determine the first training end date
    # If eval_periods > backtest: use eval_periods as the test periods
    # otherwise, use the backtest_periods

    # If keep_eval_fixed, minus further to take into account the backtest and eval periods
    if keep_eval_fixed:
        buffer_periods = eval_periods + (backtest_periods - 1)
    else:
        buffer_periods = (
            eval_periods if eval_periods > backtest_periods else backtest_periods
        )

    first_train_end = data_end_date - buffer_periods

    # Number of rolling times
    for i in range(backtest_periods):
        train_end = first_train_end + i

        if train_end <= series.index.min():
            break  # Stop if not enough training data

        yield series.loc[series.index <= train_end]


@overload
def backtest_evaluate(
    series: pd.Series | pd.DataFrame,
    models: ModelDict,
    backtest_periods: int = 5,
    eval_periods: int = 2,
    keep_eval_fixed: bool = False,
    min_data_points: int = 8,
    fcst_col_index: int = 0,
    return_results: bool = False,
) -> ModelResults: ...


@overload
def backtest_evaluate(
    series: pd.Series | pd.DataFrame,
    models: ModelDict,
    backtest_periods: int = 5,
    eval_periods: int = 2,
    keep_eval_fixed: bool = False,
    min_data_points: int = 8,
    fcst_col_index: int = 0,
    return_results: bool = True,
) -> Tuple[ModelResults, pd.DataFrame]: ...


def backtest_evaluate(
    series: pd.Series | pd.DataFrame,
    models: ModelDict,
    backtest_periods: int = 5,
    eval_periods: int = 2,
    keep_eval_fixed: bool = False,
    min_data_points: int = 8,
    fcst_col_index: int = 0,
    return_results: bool = False,
) -> ModelResults:
    """Rolling back-test the series with multiple BaseForecaster models

    Parameters
    ----------
        series (pd.Series | pd.DataFrame): Pandas Series of floats
            Preprocessed, sorted, and filtered time series.
            It's assumed that the series has all the months,
            and ends with the `data_date` you want to train.
            This Series should come from the preprocessing step.

        models (ModelDict): Model dictionary
            The keys are model names and
            the values are the forecaster models from `sktime`.

        backtest_periods (int): The number of rolling back-test periods (Default is 5)

        eval_periods (int): The number of evaluation periods in each rolling (Default is 2)

        keep_eval_fixed (bool): Whether or not to keep eval_periods fixed (Default = False)

        min_data_points (int): Minimum data points in the series to perform back-testing

        fcst_col_index (int): Column index used for back-testing (in the case of pd.DataFrame)

        return_results (bool): Whether or not to return the back-testing raw results (Default is False)

    Returns
    -------
        ModelResults: A dictionary reporting the average error of each model (when `return_results` = False)

        Tuple[ModelResults, pd.DataFrame]: The results dictionary along with the rolling back-test in each period of each model (when `return_results` = True)
    """

    if len(series) == 0:
        raise ValueError("`series` must have more than 0 length for back-testing.")

    models = models.copy()

    if len(series) < min_data_points:
        model_results = {"MeanDefault": 1.0}

        if return_results:
            return model_results, pd.DataFrame()

        return model_results

    if isinstance(series, pd.Series):
        true_series = series.copy()
    elif isinstance(series, pd.DataFrame):
        col_of_interest = series.columns[fcst_col_index]
        true_series = series[col_of_interest].copy()

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
            for backtest_series in get_backtest_periods(
                series, backtest_periods, eval_periods, keep_eval_fixed
            ):
                df_eval = _get_eval_df(
                    backtest_series=backtest_series,
                    true_series=true_series,
                    model=model,
                    model_name=model_name,
                    eval_periods=eval_periods,
                )
                eval_results.append(df_eval)

        except Exception as e:  # Skip the failed model
            # print(model_name, e)  # Print error log
            continue

        df_eval = pd.concat(eval_results)

        # Append results
        model_results[model_name] = smape(df_eval[actual_col], df_eval[fcst_col])

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
