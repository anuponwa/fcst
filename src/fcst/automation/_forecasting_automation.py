import warnings
from typing import Literal, Tuple, overload

import pandas as pd
from joblib import Parallel, delayed

from ..common.types import ModelDict
from ..evaluation.backtesting import backtest_evaluate
from ..evaluation.model_selection import select_best_models
from ..forecasting.ensemble import _ensemble_forecast_X, ensemble_forecast
from ..models.model_list import fast_models, multivar_models
from ..preprocessing import prepare_multivar_timeseries, prepare_timeseries


@overload
def _forecasting_pipeline(
    series: pd.Series,
    backtest_periods: int,
    eval_periods: int,
    top_n: int,
    forecasting_periods: int,
    models: ModelDict = fast_models,
    df_y_X: pd.DataFrame = None,
    multivar_models: ModelDict = multivar_models,
    return_backtest_results: bool = False,
    keep_eval_fixed: bool = False,
) -> pd.DataFrame: ...


@overload
def _forecasting_pipeline(
    series: pd.Series,
    backtest_periods: int,
    eval_periods: int,
    top_n: int,
    forecasting_periods: int,
    models: ModelDict = fast_models,
    df_y_X: pd.DataFrame = None,
    multivar_models: ModelDict = multivar_models,
    return_backtest_results: bool = True,
    keep_eval_fixed: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]: ...


def _forecasting_pipeline(
    series: pd.Series,
    backtest_periods: int,
    eval_periods: int,
    top_n: int,
    forecasting_periods: int,
    models: ModelDict = fast_models,
    df_y_X: pd.DataFrame = None,
    multivar_models: ModelDict = multivar_models,
    return_backtest_results: bool = False,
    keep_eval_fixed: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
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

        models (ModelDict): A dictionary of models to use in forecasting (Default = fast_models)

        df_y_X (pd.DataFrame): A dataframe for multivariate forecast (Default = None)
            The dataframe must be preprocessed. The index is time period index.
            The missing dates must be filled.
            The first column is the column of interest, other columns are features.
            Use prepare or extract timeseries from `preprocessing` module.

        multivar_models (ModelDict): A dictionary of multivariate models to use in forecasting (Default = multivar_models)

        return_backtest_results (bool): Whether or not to return the back-testing raw results (Default is False)

        keep_eval_fixed (bool): Whether or not to keep eval_periods fixed (Default = False)

    Returns
    -------
        Tuple[str, pd.Series]: ID and the resulting forecasted series (when return_backtest_results = False)

        Tuple[pd.Series, pd.DataFrame]: ID and the resulting forecasted series with the back-testing raw results (when return_backtest_results = True)
    """

    with warnings.catch_warnings():
        # Suppress all warnings from inside this function
        warnings.simplefilter("ignore")

        run_multivar = False
        if isinstance(df_y_X, pd.DataFrame) and isinstance(multivar_models, dict):
            run_multivar = True

        try:
            model_results = backtest_evaluate(
                series,
                models,
                backtest_periods=backtest_periods,
                eval_periods=eval_periods,
                keep_eval_fixed=keep_eval_fixed,
                return_results=return_backtest_results,
            )

            if run_multivar:  # Run multi-variate models also
                model_results_multi = backtest_evaluate(
                    df_y_X,
                    multivar_models,
                    backtest_periods=backtest_periods,
                    eval_periods=eval_periods,
                    keep_eval_fixed=keep_eval_fixed,
                    return_results=return_backtest_results,
                )

            # Back-test results
            if return_backtest_results:
                model_results, df_backtest_results = model_results[0], model_results[1]
                if run_multivar:
                    model_results_multi, df_backtest_results_multi = (
                        model_results_multi[0],
                        model_results_multi[1],
                    )
                    df_backtest_results = pd.concat(
                        [df_backtest_results, df_backtest_results_multi]
                    )

            # Gather the results from multivar models for model selection
            if run_multivar:
                # Append tags so we know which models are uni or multivariate
                model_results = {k: (v, "1_uni") for k, v in model_results.items()}
                model_results_multi = {
                    k: (v, "2_multi") for k, v in model_results_multi.items()
                }

                model_results = {**model_results, **model_results_multi}

                print(model_results)

            models_list = select_best_models(model_results=model_results, top_n=top_n)

            multivar_models_exists = True

            if run_multivar:
                model_types = [m[1] for m in models_list]
                model_names_list = [m[0] for m in models_list]

                # Check if we need to run multivariate models
                if "2_multi" not in model_types:
                    multivar_models_exists = False

            if not run_multivar:
                forecast_results = ensemble_forecast(
                    models=models,
                    model_names=models_list,
                    series=series,
                    periods=forecasting_periods,
                )
            else:
                if multivar_models_exists:
                    forecast_results = _ensemble_forecast_X(
                        models=models,
                        multivar_models=multivar_models,
                        model_names=models_list,
                        series=series,
                        df_y_X=df_y_X,
                        periods=forecasting_periods,
                    )
                else:
                    forecast_results = ensemble_forecast(
                        models=models,
                        model_names=model_names_list,
                        series=series,
                        periods=forecasting_periods,
                    )

            df_forecast_results = pd.DataFrame(forecast_results)

            if not run_multivar:
                df_forecast_results["selected_models"] = "|".join(models_list)
            else:
                df_forecast_results["selected_models"] = "|".join(model_names_list)

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
    agg_method: Literal["sum", "mean"] = "sum",
    fillna: Literal["bfill", "ffill"] | int | float = 0,
    models: ModelDict = fast_models,
    df_X_raw: pd.DataFrame = None,
    feature_cols: list[str] | None = None,
    min_caps_X: float | int | dict[str, float | int] | None = 0,
    agg_methods_X: Literal["sum", "mean"] | dict[str, Literal["sum", "mean"]] = "sum",
    fillna_X: Literal["bfill", "ffill"] | int | float = 0,
    multivar_models: ModelDict = multivar_models,
    keep_eval_fixed: bool = False,
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
    agg_method: Literal["sum", "mean"] = "sum",
    fillna: Literal["bfill", "ffill"] | int | float = 0,
    models: ModelDict = fast_models,
    df_X_raw: pd.DataFrame = None,
    feature_cols: list[str] | None = None,
    min_caps_X: float | int | dict[str, float | int] | None = 0,
    agg_methods_X: Literal["sum", "mean"] | dict[str, Literal["sum", "mean"]] = "sum",
    fillna_X: Literal["bfill", "ffill"] | int | float = 0,
    multivar_models: ModelDict = multivar_models,
    keep_eval_fixed: bool = False,
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
    min_cap: int | None = 0,
    freq: str = "M",
    agg_method: Literal["sum", "mean"] = "sum",
    fillna: Literal["bfill", "ffill"] | int | float = 0,
    models: ModelDict = fast_models,
    df_X_raw: pd.DataFrame = None,
    feature_cols: list[str] | None = None,
    min_caps_X: float | int | dict[str, float | int] | None = 0,
    agg_methods_X: Literal["sum", "mean"] | dict[str, Literal["sum", "mean"]] = "sum",
    fillna_X: Literal["bfill", "ffill"] | int | float = 0,
    multivar_models: ModelDict = multivar_models,
    keep_eval_fixed: bool = False,
    return_backtest_results: bool = False,
    parallel: bool = True,
    n_jobs: int = -1,
    id_join_char: str = "_",
) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
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

        min_cap (int | None): Minimum value to cap before forecast
            If set, the value is used to set the minimum.
            For example, you might want to set 0 for sales.
            If None, use the existing values.

        freq (str): Frequency to resample and forecast (Default = "M")

        agg_methods (Literal["sum", "mean"]): String specifying aggregation method to value column (Default = "sum")

        fillna (Literal["bfill", "ffill"] | int | float): Method or number to fill missing values (Default = 0)
            Method to fill missing values:
            - int/float: Fill with a specific number (e.g., 0).
            - "ffill": Forward fill.
            - "bfill": Backward fill.

        models: (ModelDict): A dictionary of models to use in forecasting (Default = fast_models)

        df_X_raw (pd.DataFrame): Raw DF for external features that has a date column, other info, and the values to forecast (Default = None)

        feature_cols (list[str]): List of feature columns (Default = None)

        min_caps_X (float | int | dict[str, float | int] | None): Minimum value to cap before forecast
            If set, the value is used to set the minimum.
            For example, you might want to set 0 for sales.
            If None, use the existing values.
            It can also be a dictionary, e.g.,
                min_caps = {"feature_1": 0}

        agg_methods_X (Literal["sum", "mean"]): String specifying aggregation method to value column (Default = "sum")

        fillna_X (Literal["bfill", "ffill"] | int | float): Method or number to fill missing values (Default = 0)
            Method to fill missing values:
            - int/float: Fill with a specific number (e.g., 0).
            - "ffill": Forward fill.
            - "bfill": Backward fill.

        multivar_models (ModelDict): A dictionary of multivariate models to use in forecasting (Default = multivar_models)

        keep_eval_fixed (bool): Whether or not to keep eval_periods fixed (Default = False)

        return_backtest_results (bool): Whether or not to return the back-testing raw results (Default is False)

        parallel (bool): Whether or not to utilise parallisation (Default is True)

        n_jobs (int): For parallel only, the number of jobs (Default = -1)

        id_join_char (str): A character to join multiple ID columns (Default = "_")

    Returns
    -------
        pd.DataFrame: The ensemble forecast DataFrame (when `return_backtest_results` = False)

        Tuple[pd.DataFrame, pd.DataFrame]: The ensemble forecast DataFrame and the back-testing raw results (when `return_backtest_results` = True)
    """

    models = models.copy()

    # If X is provided, features and multivar models must also be provided
    X_cond = df_X_raw is not None
    feat_cond = feature_cols is not None
    multivar_models_cond = multivar_models is not None

    # Check run multivariate
    if X_cond:
        if not feat_cond or not multivar_models_cond:
            raise ValueError(
                "`df_X_raw`, `feature_cols`, and `multivar_models` must all be provided for multivariate forecasting."
            )

    run_multivar = False

    if X_cond and feat_cond and multivar_models_cond:
        run_multivar = True

    if run_multivar:
        multivar_models = multivar_models.copy()

    def _fcst(series, df_y_X=None):  # Internal function for simplicity
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
                df_y_X=df_y_X,
                multivar_models=multivar_models,  # Constant
                return_backtest_results=return_backtest_results,  # Constant
                keep_eval_fixed=keep_eval_fixed,  # Constant
            )

    timeseries: dict[str, pd.Series] = prepare_timeseries(
        df_raw=df_raw,
        date_col=date_col,
        value_col=value_col,
        data_period_date=data_period_date,
        id_cols=id_cols,
        min_cap=min_cap,
        freq=freq,
        agg_method=agg_method,
        fillna=fillna,
        id_join_char=id_join_char,
    )

    if run_multivar:
        df_multivar: dict[str, pd.DataFrame] = prepare_multivar_timeseries(
            df_raw=df_raw,
            df_X_raw=df_X_raw,
            date_col=date_col,
            value_col=value_col,
            feature_cols=feature_cols,
            data_period_date=data_period_date,
            id_cols=id_cols,
            min_cap=min_cap,
            min_caps_X=min_caps_X,
            freq=freq,
            agg_method=agg_method,
            agg_methods_X=agg_methods_X,
            fillna=fillna,
            fillna_X=fillna_X,
            id_join_char=id_join_char,
        )

    ids_list = [k for k in timeseries.keys()]

    if not run_multivar:
        timeseries_list = [(v, None) for v in timeseries.values()]

    else:
        timeseries_list = [(v, df_multivar.get(k, None)) for k, v in timeseries.items()]

    # Check if run in parallel
    if parallel:
        results_list = Parallel(n_jobs=n_jobs)(
            (delayed(_fcst)(series, df_y_X)) for series, df_y_X in timeseries_list
        )

        results = [(id_, result) for id_, result in zip(ids_list, results_list)]
    else:
        results = [
            (id_, _fcst(series, df_y_X))
            for id_, (series, df_y_X) in zip(ids_list, timeseries_list)
        ]

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
