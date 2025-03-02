# -*- coding: utf-8 -*-
"""
models sub-package
~~~~
Provides base models.
"""

import pandas as pd
from sktime.forecasting.auto_reg import AutoREG
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.theta import ThetaForecaster

from ..common.types import Forecaster, ModelDict


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


base_models: ModelDict = {
    "AutoETS": AutoETS(auto=True),
    "Theta": ThetaForecaster(deseasonalize=False),
    "AutoREG": AutoREG(lags=3, trend="n"),
    "AutoREGTrend": AutoREG(lags=3, trend="c"),
    "Naive": NaiveForecaster(strategy="last"),
    "Naive3mths": NaiveForecaster(strategy="mean", window_length=3),
    "Naive6mths": NaiveForecaster(strategy="mean", window_length=6),
    "Mean": NaiveForecaster(strategy="mean"),
    "MeanDefault": MeanDefaultForecaster(window=3),
    "Zero": ZeroForecaster(),
}
