import warnings
from collections.abc import Iterable
from typing import Literal, Tuple, overload

import pandas as pd
from joblib import Parallel, delayed

from ..common.types import ModelDict
from ..evaluation.model_evaluation import backtest_evaluate
from ..evaluation.model_selection import select_best_models
from ..forecasting.ensemble import ensemble_forecast
from ..models.model_list import base_models
from ..preprocessing import prepare_timeseries


@overload
def _forecasting_pipeline(
    series: pd.Series,
    backtest_periods: int,
    eval_periods: int,
    top_n: int,
    forecasting_periods: int,
    models: ModelDict = base_models,
    return_backtest_results: bool = False,
    eval_method: Literal["rolling", "one-time"] = "rolling",
) -> pd.DataFrame: ...


@overload
def _forecasting_pipeline(
    series: pd.Series,
    backtest_periods: int,
    eval_periods: int,
    top_n: int,
    forecasting_periods: int,
    models: ModelDict = base_models,
    return_backtest_results: bool = True,
    eval_method: Literal["rolling", "one-time"] = "rolling",
) -> Tuple[pd.DataFrame, pd.DataFrame]: ...


def _forecasting_pipeline(
    series: pd.Series,
    backtest_periods: int,
    eval_periods: int,
    top_n: int,
    forecasting_periods: int,
    models: ModelDict = base_models,
    return_backtest_results: bool = False,
    eval_method: Literal["rolling", "one-time"] = "rolling",
) -> Tuple[str, pd.DataFrame]:
    """Performs model selection and ensemble forecast for a single time-series

    Parameters
    ----------
        series (pd.Series): Time series to forecast
            The series must be preprocessed. The index is time period index.
            The missing dates must be filled.
            The easiest way is to use `extract_timeseries()` function from `preprocessing`.

        backtest_periods (int): Number of periods to back-test

        eval_periods (int): Number of periods to evaluate in each rolling back-test
            If `eval_method`=="one-time", this argument is ignored, and `backtest_periods` will be used instead.

        top_n (int): Top N models to return

        forecasting_periods (int): Forecasting periods

        models: (ModelDict): A dictionary of models to use in forecasting (Default = base_models)

        return_backtest_results (bool): Whether or not to return the back-testing raw results (Default is False)

        eval_method ("rolling" or "one-time"): The method to evaluate back-testing (Default="rolling")

    Returns
    -------
        Tuple[str, pd.Series]: ID and the resulting forecasted series (when return_backtest_results = False)

        Tuple[pd.Series, pd.DataFrame]: ID and the resulting forecasted series with the back-testing raw results (when return_backtest_results = True)
    """
    with warnings.catch_warnings():
        # Suppress all warnings from inside this function
        warnings.simplefilter("ignore")

        try:
            model_results = backtest_evaluate(
                series,
                models,
                backtest_periods=backtest_periods,
                eval_periods=eval_periods,
                return_results=return_backtest_results,
                eval_method=eval_method,
            )

            if return_backtest_results:
                model_results, df_backtest_results = model_results[0], model_results[1]

            models_list = select_best_models(model_results=model_results, top_n=top_n)

            forecast_results = ensemble_forecast(
                models=models,
                model_names=models_list,
                series=series,
                periods=forecasting_periods,
            )

            df_forecast_results = pd.DataFrame(forecast_results)
            df_forecast_results["selected_models"] = "|".join(models_list)

            if return_backtest_results:
                return df_forecast_results, df_backtest_results

            return df_forecast_results

        except Exception as e:
            print("Unexpected error occurred for ID:", e)


@overload
def run_forecasting_automation(
    df_raw: pd.DataFrame,
    date_col: str,
    value_col: str,
    data_period_date: pd.Period,
    backtest_periods: int,
    eval_periods: int,
    top_n: int,
    forecasting_periods: int,
    id_cols: list[str] | None = None,
    id_join_char: str = "_",
    min_cap: int | None = 0,
    freq: str = "M",
    models: ModelDict = base_models,
    eval_method: Literal["rolling", "one-time"] = "rolling",
    return_backtest_results: bool = False,
    parallel: bool = True,
    n_jobs: int = -1,
) -> pd.DataFrame: ...


@overload
def run_forecasting_automation(
    df_raw: pd.DataFrame,
    date_col: str,
    value_col: str,
    data_period_date: pd.Period,
    backtest_periods: int,
    eval_periods: int,
    top_n: int,
    forecasting_periods: int,
    id_cols: list[str] | None = None,
    id_join_char: str = "_",
    min_cap: int | None = 0,
    freq: str = "M",
    models: ModelDict = base_models,
    eval_method: Literal["rolling", "one-time"] = "rolling",
    return_backtest_results: bool = True,
    parallel: bool = True,
    n_jobs: int = -1,
) -> Tuple[pd.DataFrame, pd.DataFrame]: ...


def run_forecasting_automation(
    df_raw: pd.DataFrame,
    date_col: str,
    value_col: str,
    data_period_date: pd.Period,
    backtest_periods: int,
    eval_periods: int,
    top_n: int,
    forecasting_periods: int,
    id_cols: list[str] | None = None,
    id_join_char: str = "_",
    min_cap: int | None = 0,
    freq: str = "M",
    models: ModelDict = base_models,
    eval_method: Literal["rolling", "one-time"] = "rolling",
    return_backtest_results: bool = False,
    parallel: bool = True,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Runs and returns forecast results for each ID

    This automatically runs the pipeline.
    The process assumes you already have the `df_forecasting`
    The index must be datetime or period index, use `prepare_forecasting_df()` function.
    The dataframe must have an `id_col` to distinguish different time-series.

    For each ID, the steps consist of:
    1. Tries to rolling back-test
    2. Select the best model(s) for a particular time-series ID
    3. Ensemble forecast using the best model(s)

    Parameters
    ----------
        df_raw (pd.DataFrame): Raw DF that has a date column, other info, and the value to forecast

        date_col (str): The date column to use in forecasting

        value_col (str): Column to forecast

        date_period_date (pd.Period): Ending date for data to use for training

        backtest_periods (int): Number of periods to back-test

        eval_periods (int): Number of periods to evaluate in each rolling back-test
            If `eval_method`=="one-time", this argument is ignored, and `backtest_periods` will be used instead.

        top_n (int): Top N models to return

        forecasting_periods (int): Forecasting periods

        id_cols (list[str] | None): A list containing the column names to create a unique time-series ID (Default is None)
            If None, the whole dataframe is treated as a single time-series
            If a list of columns is passed in, a new "id" index will be created

        id_join_char (str): A character to join multiple ID columns (Default = "_")

        min_cap (int | None): Minimum value to cap before forecast
            If set, the value is used to set the minimum.
            For example, you might want to set 0 for sales.
            If None, use the existing values.

        freq (str): Frequency to resample and forecast (Default = "M")

        models: (ModelDict): A dictionary of models to use in forecasting (Default = base_models)

        eval_method ("rolling" or "one-time"): The method to evaluate back-testing (Default="rolling")

        return_backtest_results (bool): Whether or not to return the back-testing raw results (Default is False),

        parallel (bool): Whether or not to utilise parallisation (Default is True)

        n_jobs (int): For parallel only, the number of jobs (Default = -1)

    Returns
    -------
        pd.DataFrame: The ensemble forecast DataFrame (when `return_backtest_results` = False)

        Tuple[pd.DataFrame, pd.DataFrame]: The ensemble forecast DataFrame and the back-testing raw results (when `return_backtest_results` = True)
    """

    models = models.copy()

    def _fcst(series):  # Internal function for simplicity
        with warnings.catch_warnings():
            # Suppress all warnings from inside this function
            warnings.simplefilter("ignore")
            return _forecasting_pipeline(
                series=series,
                backtest_periods=backtest_periods,  # Constant
                eval_periods=eval_periods,  # Constant
                top_n=top_n,  # Constant
                forecasting_periods=forecasting_periods,  # Constant
                models=models,  # Constant
                return_backtest_results=return_backtest_results,  # Constant
                eval_method=eval_method,  # Constant
            )

    timeseries: pd.Series | Iterable[Tuple[str, pd.Series]] = prepare_timeseries(
        df_raw=df_raw,
        date_col=date_col,
        value_col=value_col,
        data_period_date=data_period_date,
        id_cols=id_cols,
        min_cap=min_cap,
        freq=freq,
        id_join_char=id_join_char,
    )

    if not id_cols:
        return _fcst(series=timeseries)

    # Check if run in parallel
    if parallel:
        timeseries_list = list(timeseries)
        results_list = Parallel(n_jobs=n_jobs)(
            (delayed(_fcst)(series)) for _, series in timeseries_list
        )

        results = [
            (id_, result) for (id_, _), result in zip(timeseries_list, results_list)
        ]
    else:
        results = [(id_, _fcst(series)) for id_, series in timeseries]

    def _filter_none_results(results_list: list[Tuple[str, pd.Series]]):
        return list(filter(lambda x: x[1] is not None, results_list))

    def _get_df_forecasting_from_each_result(
        result: Tuple[str, pd.DataFrame | Tuple[pd.Series, pd.DataFrame]],
    ):
        id_ = result[0]

        if not return_backtest_results:
            df_results = result[1]
        else:
            df_results = result[1][0]

        df_results["id"] = id_

        return df_results

    def _get_df_backtest_from_each_result(
        result: Tuple[str, Tuple[pd.DataFrame, pd.DataFrame]],
    ):
        id_ = result[0]
        df_raw = result[1][1]

        df_raw["id"] = id_

        return df_raw

    results_filtered = _filter_none_results(results)
    df_forecast_results = pd.concat(
        map(_get_df_forecasting_from_each_result, results_filtered)
    )

    if return_backtest_results:
        df_backtest_results = pd.concat(
            map(_get_df_backtest_from_each_result, results_filtered)
        )
        return df_forecast_results, df_backtest_results

    return df_forecast_results
