# -*- coding: utf-8 -*-
"""
horizon sub-package
~~~~
Provides periods and horizons utilities.
"""

import pandas as pd
from sktime.forecasting.base import ForecastingHorizon


def get_future_periods(
    data_period_date: pd.Period,
    periods: int = 5,
) -> pd.PeriodIndex:
    """Returns future periods from the current data_period_date"""

    return pd.period_range(data_period_date + 1, periods=periods)


def get_future_forecast_horizon(
    data_period_date: pd.Period,
    periods: int = 5,
) -> ForecastingHorizon:
    """Returns future forecasting horizon to use with models"""

    future_period_index = get_future_periods(data_period_date, periods)
    return ForecastingHorizon(future_period_index, is_relative=False)
