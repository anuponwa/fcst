from typing import Tuple

import numpy as np
import pandas as pd

from ..common.types import Forecaster, ModelDict
from ..models._models import MeanDefaultForecaster
from .forecasting import forecast


def ensemble_forecast(
    models: ModelDict,
    model_names: list[str],
    series: pd.Series,
    periods: int,
    forecast_col: str = "forecast",
    min_data_points: int = 3,
    fallback_model: Forecaster = MeanDefaultForecaster(window=3),
    min_forecast: float | int | None = 0,
    max_forecast_factor: float | int | None = 2.5,
) -> pd.Series:
    """Forecasts the series using an ensemble method

    Parameters
    ----------
        models (ModelDict): Model dictionary
            The keys are model names and
            the values are the forecaster models from `sktime` or other compatible models.

        model_names (list[str]): Models to use for ensembling

        series (pd.Series): Pandas Series of floats
            Preprocessed, sorted, and filtered time series.
            It's assumed that the series has all the months,
            and ends with the `data_date` you want to train.
            This Series should come from the preprocessing step.

        periods (int): Forecasting periods

        forecast_col (str): The column name for the output forecast (Default is "forecast")

        min_data_points (int): Minimum data points the series must have to forecast using the model (Default is 3)

        fallback_model (Forecaster): A model used as a fall-back when the number of data points is too low (Default to Mean)

        min_forecast (float | int | None): Set a minimum forecast (Default = 0)

        max_forecast_factor (float | int | None): The factor of maximum forecast relative to the maximum history (Default = 2.5)

    Returns
    -------
        pd.Series[float]: Future time horizon depending on the series' end date and `periods`
    """

    models = models.copy()

    set_diff = set(model_names).difference(set(models.keys()))

    if len(set_diff) > 0:
        raise ValueError(
            f"`model_names` must exist in `models` keys. Key(s) error: {set_diff}."
        )

    forecast_results = []

    for model in model_names:
        model_output = forecast(
            model=models[model],
            series=series,
            periods=periods,
            forecast_col=forecast_col,
            min_data_points=min_data_points,
            fallback_model=fallback_model,
            min_forecast=min_forecast,
            max_forecast_factor=max_forecast_factor,
        )
        forecast_results.append(model_output)

    predictions = pd.concat(forecast_results, join="inner", axis=1).apply(
        np.mean, axis=1
    )
    predictions = predictions.rename(forecast_col)

    return predictions


def _ensemble_forecast_X(
    models: ModelDict,
    multivar_models: ModelDict,
    model_names: list[str | Tuple[str, str]],
    series: pd.Series,
    df_y_X: pd.DataFrame,
    periods: int,
    min_data_points: int = 3,
    fallback_model: Forecaster = MeanDefaultForecaster(window=3),
    min_forecast: float | int | None = 0,
    max_forecast_factor: float | int | None = 2.5,
    forecast_col: str = "forecast",
    fcst_col_index: int = 0,
) -> pd.Series:
    """Forecasts the series using an ensemble method

    Parameters
    ----------
        models (ModelDict): Model dictionary
            The keys are model names and
            the values are the forecaster models from `sktime` or other compatible models.

        multivar_models (ModelDict): Model dictionary
            A dictionary of multivariate models to use in forecasting

        model_names (list[str]): Models to use for ensembling

        series (pd.Series): Pandas Series of floats
            Preprocessed, sorted, and filtered time series.
            It's assumed that the series has all the months,
            and ends with the `data_date` you want to train.
            This Series should come from the preprocessing step.

        df_y_X (pd.DataFrame): A dataframe for multivariate forecast (Default = None)
            The dataframe must be preprocessed. The index is time period index.
            The missing dates must be filled.
            The first column is the column of interest, other columns are features.
            Use prepare or extract timeseries from `preprocessing` module.

        periods (int): Forecasting periods

        min_data_points (int): Minimum data points the series must have to forecast using the model (Default is 3)

        fallback_model (Forecaster): A model used as a fall-back when the number of data points is too low (Default to Mean)

        min_forecast (float | int | None): Set a minimum forecast (Default = 0)

        max_forecast_factor (float | int | None): The factor of maximum forecast relative to the maximum history (Default = 2.5)

        forecast_col (str): The column name for the output forecast (Default is "forecast")

        fcst_col_index (int): Index of the column of interest in foreasting (only applies if `series` is pd.DataFrame) (Default = 0 (first column))

    Returns
    -------
        pd.Series[float]: Future time horizon depending on the series' end date and `periods`
    """

    models = models.copy()
    multivar_models = multivar_models.copy()

    uni_model_names = [m[0] for m in model_names if m[1] == "1_uni"]
    multi_model_names = [m[0] for m in model_names if m[1] == "2_multi"]

    set_diff_uni = set(uni_model_names).difference(set(models.keys()))
    set_diff_multi = set(multi_model_names).difference(set(multivar_models.keys()))

    if len(set_diff_uni) > 0:
        raise ValueError(
            f"Univariate models must exist in `models` keys. Key(s) error: {set_diff_uni}."
        )

    if len(set_diff_multi) > 0:
        raise ValueError(
            f"Multivariate models must exist in `multivar_models` keys. Key(s) error: {set_diff_multi}."
        )

    forecast_results = []

    for model in uni_model_names:
        model_output = forecast(
            model=models[model],
            series=series,
            periods=periods,
            min_data_points=min_data_points,
            fallback_model=fallback_model,
            min_forecast=min_forecast,
            max_forecast_factor=max_forecast_factor,
        )
        forecast_results.append(model_output)

    for model in multi_model_names:
        model_output = forecast(
            model=multivar_models[model],
            series=df_y_X,
            periods=periods,
            min_forecast=min_forecast,
            max_forecast_factor=max_forecast_factor,
            fcst_col_index=fcst_col_index,
        )
        forecast_results.append(model_output)

    predictions = pd.concat(forecast_results, join="inner", axis=1).apply(
        np.mean, axis=1
    )
    predictions = predictions.rename(forecast_col)

    return predictions
