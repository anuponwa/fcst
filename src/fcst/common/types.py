from typing import Protocol, Tuple

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
# Either error measure or (error, model_type)
ModelResults = dict[str, float | Tuple[float, str]]
