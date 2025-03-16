import pandas as pd
from sktime.forecasting.base import ForecastingHorizon

from ..common.types import Forecaster


class MeanDefaultForecaster(Forecaster):
    """Averages the latest `window` values"""

    def __init__(self, window: int = 3):
        self.window = window
        self.mean_val = 0
        self.fh = None

    def fit(self, y: pd.Series, X=None, fh: ForecastingHorizon = None):
        self.mean_val = y.iloc[-self.window :].mean()
        if fh is not None:
            self.fh = fh

        self.cutoff = y.index.max()

        return self

    def predict(self, fh: ForecastingHorizon = None, X=None):
        if self.fh is None and fh is None:
            raise ValueError("`fh` must be passed in either in `fit()` or `predict()`")

        if fh is not None:
            self.fh = fh

        return pd.Series(
            self.mean_val, index=self.fh.to_absolute_index(cutoff=self.cutoff)
        )


class ZeroForecaster(Forecaster):
    """Always predicts 0"""

    def __init__(self):
        self.pred_val = 0
        self.fh = None

    def fit(self, y: pd.Series, X=None, fh: ForecastingHorizon = None):
        if fh is not None:
            self.fh = fh

        self.cutoff = y.index.max()

        return self

    def predict(self, fh: ForecastingHorizon = None, X=None):
        if self.fh is None and fh is None:
            raise ValueError("`fh` must be passed in either in `fit()` or `predict()`")

        if fh is not None:
            self.fh = fh

        return pd.Series(
            self.pred_val, index=self.fh.to_absolute_index(cutoff=self.cutoff)
        )
