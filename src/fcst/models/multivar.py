import pandas as pd
from sktime.forecasting.base import ForecastingHorizon


class MultivariateModel:
    def __init__(self, model_cls_instance, val_col: str | None = None):
        """Generic wrapper for multi-variate models.

        Parameters
        ----------
            model_cls_instance (class instance): Multi-variate model instance
        """

        self.model = model_cls_instance
        self.val_col = val_col

    def fit(self, y: pd.DataFrame, X=None, fh: ForecastingHorizon = None):
        """Fits the model to the time series."""

        if fh is not None:
            self.fh = fh

        self.y = y
        self.cutoff = y.index.max()

        self.model.fit(y=self.y, X=X, fh=self.fh)

        return self

    def predict(
        self, fh: ForecastingHorizon = None, X=None
    ) -> pd.Series | pd.DataFrame:
        if self.fh is None and fh is None:
            raise ValueError("`fh` must be passed in either in `fit()` or `predict()`")

        if fh is not None:
            self.fh = fh

        self.df_pred = self.model.predict(fh=fh)

        if self.val_col is not None:
            return self.df_pred[self.val_col]

        return self.df_pred
