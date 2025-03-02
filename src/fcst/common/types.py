from typing import Protocol

import pandas as pd
from sktime.forecasting.base import ForecastingHorizon


class Forecaster(Protocol):
    def fit(self, y: pd.Series, X=None, fh: ForecastingHorizon = None):
        """Trains the forecaster"""

        ...

    def predict(self, fh: ForecastingHorizon = None, X=None):
        """Make predictions for the given forecasting horizon"""
        ...


ModelDict = dict[str, Forecaster]
ModelResults = dict[str, float]
