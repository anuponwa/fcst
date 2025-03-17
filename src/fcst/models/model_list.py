from autots.models.basics import (
    FFT,
    BallTreeMultivariateMotif,
    BallTreeRegressionMotif,
    MetricMotif,
    SeasonalityMotif,
    SectionalMotif,
)
from autots.models.cassandra import Cassandra
from autots.models.sklearn import (
    MultivariateRegression,
    UnivariateRegression,
    WindowRegression,
)
from autots.models.statsmodels import ARDL, ETS, GLS
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sktime.forecasting.auto_reg import AutoREG
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.statsforecast import StatsForecastAutoARIMA
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.var_reduce import VARReduce

from ..common.types import ModelDict
from ._models import MeanDefaultForecaster, ZeroForecaster
from .autots import AutoTSWrapper

base_models: ModelDict = {
    "AutoETS": AutoETS(auto=True),
    "Theta": ThetaForecaster(deseasonalize=False),
    "ThetaSeason": ThetaForecaster(sp=12),
    "AutoREG": AutoREG(lags=3, trend="n"),
    "AutoREGTrend": AutoREG(lags=3, trend="c"),
    "AutoRegL12": AutoREG(lags=12, trend="n"),
    "AutoREGTrendL12": AutoREG(lags=12),
    # "VARn": VAR(trend="n"),
    # "VARc": VAR(),
    # "VARct": VAR(trend="ct"),
    "VARReduceL1": VARReduce(lags=1),
    "VARReduceL3": VARReduce(lags=3),
    "VARReduceL6": VARReduce(lags=6),
    "VARReduceL12": VARReduce(lags=12),
    "Naive": NaiveForecaster(strategy="last"),
    "Naive3mths": NaiveForecaster(strategy="mean", window_length=3),
    "Naive6mths": NaiveForecaster(strategy="mean", window_length=6),
    "RidgeReduceL1": make_reduction(Ridge(), strategy="recursive", window_length=1),
    "RidgeReduceL3": make_reduction(Ridge(), strategy="recursive", window_length=3),
    "RidgeReduceL6": make_reduction(Ridge(), strategy="recursive", window_length=6),
    "RidgeReduceL9": make_reduction(Ridge(), strategy="recursive", window_length=9),
    "GradientBoostingReduceL1": make_reduction(
        GradientBoostingRegressor(), strategy="recursive", window_length=1
    ),
    "GradientBoostingReduceL3": make_reduction(
        GradientBoostingRegressor(), strategy="recursive", window_length=3
    ),
    "GradientBoostingReduceL6": make_reduction(
        GradientBoostingRegressor(), strategy="recursive", window_length=6
    ),
    "GradientBoostingReduceL9": make_reduction(
        GradientBoostingRegressor(), strategy="recursive", window_length=9
    ),
    "Mean": NaiveForecaster(strategy="mean"),
    "MeanDefault": MeanDefaultForecaster(window=3),
    "Zero": ZeroForecaster(),
}

slow_models = {
    "AutoArima": StatsForecastAutoARIMA(sp=12),
    "Prophet": Prophet(),
}


autots_models = {
    "BallTreeRegressionMotif": AutoTSWrapper(BallTreeRegressionMotif()),
    "BallTreeMultivariateMotif": AutoTSWrapper(BallTreeMultivariateMotif()),
    "Cassandra": AutoTSWrapper(Cassandra()),
    "MetricMotif": AutoTSWrapper(MetricMotif()),
    "SeasonalityMotif": AutoTSWrapper(SeasonalityMotif()),
    "SectionalMotif": AutoTSWrapper(SectionalMotif()),
    "FFT": AutoTSWrapper(FFT()),
    "UnivariateRegression": AutoTSWrapper(UnivariateRegression()),
    "MultivariateRegression": AutoTSWrapper(MultivariateRegression()),
    "WindowRegression": AutoTSWrapper(WindowRegression(forecast_length=7)),
    "GLS": AutoTSWrapper(GLS()),
    "ARDL": AutoTSWrapper(ARDL()),
    "ARDL_L3_Tn": AutoTSWrapper(ARDL(lags=3, trend="n")),
    "ARDL_L3_Tc": AutoTSWrapper(ARDL(lags=3, trend="c")),
    "ARDL_L3_Tt": AutoTSWrapper(ARDL(lags=3, trend="t")),
    "ARDL_L3_Tct": AutoTSWrapper(ARDL(lags=3, trend="ct")),
    "ARDL_L6_Tn": AutoTSWrapper(ARDL(lags=6, trend="n")),
    "ARDL_L6_Tc": AutoTSWrapper(ARDL(lags=6, trend="c")),
    "ARDL_L6_Tt": AutoTSWrapper(ARDL(lags=6, trend="t")),
    "ARDL_L6_Tct": AutoTSWrapper(ARDL(lags=6, trend="ct")),
    "ARDL_L12_Tn": AutoTSWrapper(ARDL(lags=12, trend="n")),
    "ARDL_L12_Tc": AutoTSWrapper(ARDL(lags=12, trend="c")),
    "ARDL_L12_Tt": AutoTSWrapper(ARDL(lags=12, trend="t")),
    "ARDL_L12_Tct": AutoTSWrapper(ARDL(lags=12, trend="ct")),
    "ETS_a": AutoTSWrapper(
        ETS(
            trend="additive",
            seasonal=False,
            damped_trend=False,
        )
    ),
    "ETS_m": AutoTSWrapper(
        ETS(
            trend="multiplicative",
            seasonal=False,
            damped_trend=False,
        )
    ),
    "ETS_ad": AutoTSWrapper(
        ETS(
            trend="additive",
            seasonal=False,
            damped_trend=True,
        )
    ),
    "ETS_md": AutoTSWrapper(
        ETS(
            trend="multiplicative",
            seasonal=False,
            damped_trend=True,
        )
    ),
    "ETS_as": AutoTSWrapper(
        ETS(
            trend="additive",
            seasonal=True,
            seasonal_periods=12,
            damped_trend=False,
        )
    ),
    "ETS_ms": AutoTSWrapper(
        ETS(
            trend="multiplicative",
            seasonal=True,
            seasonal_periods=12,
            damped_trend=False,
        )
    ),
    "ETS_asd": AutoTSWrapper(
        ETS(
            trend="additive",
            seasonal=True,
            seasonal_periods=12,
            damped_trend=True,
        )
    ),
    "ETS_msd": AutoTSWrapper(
        ETS(
            trend="multiplicative",
            seasonal=True,
            seasonal_periods=12,
            damped_trend=True,
        )
    ),
}

fast_models = {**base_models, **autots_models}
all_models = {**base_models, **autots_models, **slow_models}
