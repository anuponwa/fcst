import warnings
from typing import Tuple, overload

import pandas as pd
from joblib import Parallel, delayed

from ..common.types import ModelDict
from ..evaluation.model_evaluation import backtest_evaluate
from ..evaluation.model_selection import select_best_models
from ..forecasting.ensemble import ensemble_forecast
from ..models import base_models
from ..preprocessing.timeseries import extract_timeseries


@overload
def run_forecasting_pipeline(
    series: pd.Series,
    backtest_periods: int,
    eval_periods: int,
    top_n: int,
    forecasting_periods: int,
    models: ModelDict = base_models,
    id_: str = "",
    return_backtest_results: bool = False,
) -> Tuple[str, pd.DataFrame]: ...


@overload
def run_forecasting_pipeline(
    series: pd.Series,
    backtest_periods: int,
    eval_periods: int,
    top_n: int,
    forecasting_periods: int,
    models: ModelDict = base_models,
    id_: str = "",
    return_backtest_results: bool = True,
) -> Tuple[str, pd.DataFrame, pd.DataFrame]: ...


def run_forecasting_pipeline(
    series: pd.Series,
    backtest_periods: int,
    eval_periods: int,
    top_n: int,
    forecasting_periods: int,
    models: ModelDict = base_models,
    id_: str = "",
    return_backtest_results: bool = False,
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

        top_n (int): Top N models to return

        forecasting_periods (int): Forecasting periods

        models: (ModelDict): A dictionary of models to use in forecasting (Default = base_models)

        id_ (str): ID identifying the series (Default = "")

        return_backtest_results (bool): Whether or not to return the back-testing raw results (Default is False),

    Returns
    -------
        Tuple[str, pd.Series]: ID and the resulting forecasted series (when return_backtest_results = False)

        Tuple[str, pd.Series, pd.DataFrame]: ID and the resulting forecasted series with the back-testing raw results (when return_backtest_results = True)
    """

    try:
        model_results = backtest_evaluate(
            series,
            models,
            backtest_periods=backtest_periods,
            eval_periods=eval_periods,
            return_results=return_backtest_results,
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
            return id_, df_forecast_results, df_backtest_results

        return id_, df_forecast_results

    except Exception as e:
        print("Unexpected error occurred for ID:", id_, e)


# df_raw: pd.DataFrame,
# date_col: str,
# value_col: str,
# data_period_date: pd.Period,
# id_cols: list[str] | None = None,
# min_cap: int | None = 0,
# freq: str = "M",
# id_join_char: str = "_",

# TODO: Change forecasting automation to run from df_raw
# TODO: Remove forecasting pipeline as df_raw can be series


@overload
def run_forecasting_automation(
    df_forecasting: pd.DataFrame,
    value_col: str,
    data_period_date: pd.Period,
    backtest_periods: int,
    eval_periods: int,
    top_n: int,
    forecasting_periods: int,
    id_col: str = "id",
    models: ModelDict = base_models,
    return_backtest_results: bool = False,
    parallel: bool = True,
    n_jobs: int = -1,
) -> pd.DataFrame: ...


@overload
def run_forecasting_automation(
    df_forecasting: pd.DataFrame,
    value_col: str,
    data_period_date: pd.Period,
    backtest_periods: int,
    eval_periods: int,
    top_n: int,
    forecasting_periods: int,
    id_col: str = "id",
    models: ModelDict = base_models,
    return_backtest_results: bool = False,
    parallel: bool = True,
    n_jobs: int = -1,
) -> Tuple[pd.DataFrame, pd.DataFrame]: ...


def run_forecasting_automation(
    df_forecasting: pd.DataFrame,
    value_col: str,
    data_period_date: pd.Period,
    backtest_periods: int,
    eval_periods: int,
    top_n: int,
    forecasting_periods: int,
    id_col: str = "id",
    models: ModelDict = base_models,
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
        df_forecasting (pd.DataFrame): Preprocessed DF for forecasting
            Where the index is the pd.PeriodIndex,
            and the columns are id and value.
            The values are resampled to the specified `freq`.

        value_col (str): Column to forecast

        date_period_date (pd.Period): Ending date for data to use for training

        backtest_periods (int): Number of periods to back-test

        eval_periods (int): Number of periods to evaluate in each rolling back-test

        top_n (int): Top N models to return

        forecasting_periods (int): Forecasting periods

        id_col (str): ID column name to distinguish between series

        models: (ModelDict): A dictionary of models to use in forecasting (Default = base_models)

        return_backtest_results (bool): Whether or not to return the back-testing raw results (Default is False),

        parallel (bool): Whether or not to utilise parallisation (Default is True)

        n_jobs (int): For parallel only, the number of jobs (Default = -1)

    Returns
    -------
        pd.DataFrame: The ensemble forecast DataFrame (when `return_backtest_results` = False)

        Tuple[pd.DataFrame, pd.DataFrame]: The ensemble forecast DataFrame and the back-testing raw results (when `return_backtest_results` = True)
    """

    models = models.copy()

    def _fcst(id_, series):  # Internal function for delayed, parallel
        with warnings.catch_warnings():
            # Suppress all warnings from inside this function
            warnings.simplefilter("ignore")
            return run_forecasting_pipeline(
                series=series,
                backtest_periods=backtest_periods,  # Constant
                eval_periods=eval_periods,  # Constant
                top_n=top_n,  # Constant
                forecasting_periods=forecasting_periods,  # Constant
                models=models,  # Constant
                id_=id_,
                return_backtest_results=return_backtest_results,
            )

    each_series = extract_timeseries(
        df_forecasting,
        value_col=value_col,
        data_period_date=data_period_date,
        id_col=id_col,
    )

    # Run in parallel
    if parallel:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_fcst)(id_, series) for id_, series in each_series
        )
    else:
        results = [_fcst(id_, series) for id_, series in each_series]

    def _filter_none_results(results_list: list[Tuple[str, pd.Series]]):
        return list(filter(lambda x: x is not None, results_list))

    def _get_df_forecasting_from_each_result(result: Tuple[str, pd.Series]):
        id_ = result[0]
        df_results = result[1]

        df_results["id"] = id_

        return df_results

    def _get_df_backtest_from_each_result(result: Tuple[str, pd.Series, pd.DataFrame]):
        id_ = result[0]
        df_raw = result[2]

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
