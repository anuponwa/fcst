import pandas as pd
from joblib import Parallel, delayed
from typing import Tuple
import warnings

from ..preprocessing.timeseries import get_each_timeseries
from ..forecasting.ensemble import ensemble_forecast
from ..evaluation.model_evaluation import backtest_evaluate
from ..evaluation.model_selection import select_best_models
from ..common.types import ModelDict
from ..models import base_models


def _forecasting_pipeline(
    id_: str,
    series: pd.Series,
    models: ModelDict,
    backtest_periods: int,
    eval_periods: int,
    top_n: int,
    forecasting_periods: int,
) -> Tuple[str, pd.Series]:
    """Performs model selection and ensemble forecast for a single time-series

    Parameters
    ----------
        id_ (str): ID identifying the series

        series (pd.Series): Time series to forecast

        models: (ModelDict): A dictionary of models to use in forecasting

        backtest_periods (int): Number of periods to back-test

        eval_periods (int): Number of periods to evaluate in each rolling back-test

        top_n (int): Top N models to return

        forecasting_periods (int): Forecasting periods

    Returns
    -------
        Tuple[str, pd.Series]: ID and the resulting forecasted series
    """

    try:
        model_results = backtest_evaluate(
            series,
            models,
            backtest_periods=backtest_periods,
            eval_periods=eval_periods,
        )

        models_list = select_best_models(model_results=model_results, top_n=top_n)

        return id_, ensemble_forecast(
            models=models,
            model_names=models_list,
            series=series,
            periods=forecasting_periods,
        )

    except Exception as e:
        print("Unexpected error occurred for ID:", id_, e)


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
    parallel: bool = True,
    n_jobs: int = -1,
) -> list[Tuple[str, pd.Series]]:
    """Runs and returns forecast results for each ID

    This automatically runs the pipeline.
    The process assumes you already have the `df_forecasting`
    The index must be datetime or period index, use `prepare_forecasting_df()` function.
    The dataframe must have an `id_col` to distinguish different time-series.
    
    The steps consist of:
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

        models: (ModelDict): A dictionary of models to use in forecasting

        backtest_periods (int): Number of periods to back-test

        eval_periods (int): Number of periods to evaluate in each rolling back-test

        top_n (int): Top N models to return

        forecasting_periods (int): Forecasting periods

        id_col (str): ID column name to distinguish between series

        parallel (bool): Whether or not to utilise parallisation (Default is True)

        n_jobs (int): For parallel only, the number of jobs (Default = -1)

    Returns
    -------
        list[str, pd.Series]: A list containing the ID and the ensemble forecast results
    """

    models = models.copy()

    def _fcst(id_, series):  # Internal function for delayed, parallel
        with warnings.catch_warnings():
            # Suppress all warnings from inside this function
            warnings.simplefilter("ignore")
            return _forecasting_pipeline(
                id_,
                series,
                models,  # Constant
                backtest_periods,  # Constant
                eval_periods,  # Constant
                top_n,  # Constant
                forecasting_periods,  # Constant
            )

    each_series = get_each_timeseries(
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
    

    def _get_df_from_each_result(result: Tuple[str, pd.Series]):
        id_ = result[0]
        res_series = result[1]

        df_results = pd.DataFrame(res_series)
        df_results["id"] = id_

        return df_results
    
    df_forecasting_results_filtered = _filter_none_results(results)
    df_forecast_results = pd.concat(
        map(_get_df_from_each_result, df_forecasting_results_filtered)
    )

    return df_forecast_results
