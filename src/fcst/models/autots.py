import pandas as pd
from sktime.forecasting.base import ForecastingHorizon


class AutoTSWrapper:
    def __init__(self, model_cls_instance):
        """Generic wrapper for AutoTS models.

        Parameters
        ----------
            model_cls_instance (class instance): AutoTS model instance
        """

        self.model = model_cls_instance
        self.forecast_length = None
        self.forecast_length = (
            model_cls_instance.forecast_length
            if hasattr(model_cls_instance, "forecast_length")
            else None
        )

        self.fh = None
        self.col = None
        self.freq = None
        self.start = None

    def fit(self, y: pd.Series, X=None, fh: ForecastingHorizon = None):
        """Fits the model to the time series."""

        self.col = y.name
        self.freq = y.index.freq
        df_ts = y.to_frame()
        df_ts.index = df_ts.index.to_timestamp()

        if fh is not None:
            self.fh = fh

        self.start = y.index.max()
        self.model.fit(df_ts)
        return self

    def predict(self, fh: ForecastingHorizon = None, X=None):
        """Predicts future values based on the forecasting horizon."""

        if self.fh is None and fh is None:
            raise ValueError("`fh` must be passed in either in `fit()` or `predict()`")

        if fh is not None:
            self.fh = fh

        forecast_length = self.fh.to_absolute_int(start=self.start).max()
        length = self.forecast_length if self.forecast_length else forecast_length
        df_predict = self.model.predict(length).forecast
        ts_predict = df_predict[self.col]

        # Convert back to PeriodIndex
        period_index = pd.PeriodIndex(ts_predict.index, freq=self.freq)
        ts_predict.index = period_index

        return ts_predict[:forecast_length]
