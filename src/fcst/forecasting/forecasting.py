import pandas as pd

from fcst.horizon import get_future_forecast_horizon
from ..models._models import MeanDefaultForecaster
from ..common.types import Forecaster


def forecast(
    model: Forecaster,
    series: pd.Series,
    periods: int,
    forecast_col: str = "forecast",
    min_data_points: int = 3,
    fallback_model: Forecaster = MeanDefaultForecaster(window=3),
) -> pd.Series:
    """Forecasts the series using a given model from the data date

    Parameters
    ----------
    model (BaseForecaster): `sktime` forecaster model

    series (pd.Series): Pandas Series of floats
        Preprocessed, sorted, and filtered time series.
        It's assumed that the series has all the months,
        and ends with the `data_date` you want to train.
        This Series should come from the preprocessing step.

    periods (int): Forecasting periods

    forecast_col (str): The column name for the output forecast (Default is "forecast")

    min_data_points (int): Minimum data points the series must have to forecast using the model (Default is 3)

    fallback_model (Forecaster): A model used as a fall-back when the number of data points is too low (Default to Mean)

    Returns
    -------
    pd.Series[float]: Future time horizon depending on the series' end date and `periods`
    """

    if len(series) == 0:
        raise ValueError("`series` must have more than 0 length for forecasting.")

    # Set cap
    min_forecast = 0
    max_forecast = series.max() * 2.5

    data_end_date = series.index.max()  # Get the latest date from the series

    if len(series) < min_data_points:  # if N data points too low, use mean
        model = fallback_model

    fh = get_future_forecast_horizon(data_end_date, periods)
    model.fit(series, fh=fh)
    predictions = model.predict()
    # Rename the series name
    predictions = predictions.rename(forecast_col)

    # Cap values
    predictions.loc[predictions < min_forecast] = min_forecast
    predictions.loc[predictions > max_forecast] = max_forecast

    return predictions
