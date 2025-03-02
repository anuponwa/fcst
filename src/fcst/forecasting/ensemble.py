import numpy as np
import pandas as pd

from .forecasting import forecast
from ..common.types import ModelDict


def ensemble_forecast(
    models: ModelDict,
    model_names: list[str],
    series: pd.Series,
    periods: int,
    forecast_col: str = "forecast",
) -> pd.Series:
    """Forecasts the series using an ensemble method

    Parameters
    ----------
    models (ModelDict): Model dictionary
        The keys are model names and
        the values are the forecaster models from `sktime`.

    model_names (list[str]): Models to use for ensembling

    series (pd.Series): Pandas Series of floats
        Preprocessed, sorted, and filtered time series.
        It's assumed that the series has all the months,
        and ends with the `data_date` you want to train.
        This Series should come from the preprocessing step.

    periods (int): Forecasting periods

    forecast_col (str): The column name for the output forecast (Default is "forecast")

    Returns
    -------
    pd.Series[float]: Future time horizon depending on the series' end date and `periods`
    """

    models = models.copy()

    set_diff = set(model_names).difference(set(models.keys()))

    if len(set_diff) > 0:
        raise ValueError(f"`model_names` must exist in `models` keys. Key(s) error: {set_diff}.")

    forecast_results = []

    for model in model_names:
        model_output = forecast(model=models[model], series=series, periods=periods)
        forecast_results.append(model_output)

    predictions = pd.concat(forecast_results, join="inner", axis=1).apply(np.mean, axis=1)
    predictions = predictions.rename(forecast_col)

    return predictions
