import pandas as pd
from sktime.forecasting.base import ForecastingHorizon
from ..common.types import Forecaster


class MultivariateModelWrapper(Forecaster):
    def __init__(self, model_cls_instance, val_col: int | str | None = 0):
        """Generic wrapper for multi-variate models.

        Parameters
        ----------
            model_cls_instance (class instance): Multi-variate model instance

            val_col (int | str | None): Value column of interest to return prediction as series (Default = 0)
                If None, the result will return the whole dataframe.
                If int is provided, it will return the predictions for that specific column index.
                If string is provided, it will use that column name for the predictions.
                In our convention, we use `y` in the first column, and other features are from [1:].
        """

        self.model = model_cls_instance
        if val_col is not None:
            if not isinstance(val_col, int) and not isinstance(val_col, str):
                raise ValueError("`val_col` must be either int or str.")

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
            if isinstance(self.val_col, int):
                col = self.df_pred.columns[self.val_col]
            if isinstance(self.val_col, str):
                col = self.val_col

            return self.df_pred[col]

        return self.df_pred


class MeanDefaultMultiVar(Forecaster):
    """Averages the latest `window` values"""

    def __init__(self, window: int = 3, val_col: int | str | None = 0):
        self.window = window
        self.val_col = val_col
        self.mean_val = 0
        self.fh = None

    def fit(self, y: pd.DataFrame, X=None, fh: ForecastingHorizon = None):
        self.mean_val = y.iloc[-self.window :].mean()
        if fh is not None:
            self.fh = fh

        self.cutoff = y.index.max()

        return self

    def predict(
        self, fh: ForecastingHorizon = None, X=None
    ) -> pd.Series | pd.DataFrame:
        if self.fh is None and fh is None:
            raise ValueError("`fh` must be passed in either in `fit()` or `predict()`")

        if fh is not None:
            self.fh = fh

        index = self.fh.to_absolute_index(cutoff=self.cutoff)

        if self.val_col is not None:
            if isinstance(self.val_col, int):
                col = self.mean_val.columns[self.val_col]
            if isinstance(self.val_col, str):
                col = self.val_col

            return pd.Series(self.mean_val[col], index=index)

        return pd.DataFrame(
            {col: self.mean_val[col] for col in self.mean_val.index}, index=index
        )
