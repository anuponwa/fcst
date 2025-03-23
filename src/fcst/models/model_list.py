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
from sktime.forecasting.ardl import ARDL as ARDL_Sk
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.statsforecast import StatsForecastAutoARIMA
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.var import VAR
from sktime.forecasting.var_reduce import VARReduce
from sktime.forecasting.vecm import VECM

from ..common.types import ModelDict
from ._models import EMA, MeanDefaultForecaster, ZeroForecaster
from .autots import AutoTSWrapper
from .multivar import MultivariateModelWrapper, MeanDefaultMultiVar

base_models: ModelDict = {
    "AutoETS": AutoETS(auto=True),
    "Theta": ThetaForecaster(deseasonalize=False),
    "ThetaSeason": ThetaForecaster(sp=12),
    "AutoREG": AutoREG(lags=3, trend="n"),
    "AutoREGTrend": AutoREG(lags=3, trend="c"),
    "AutoREGTrendL12": AutoREG(lags=12, trend="c"),
    "AutoRegL12": AutoREG(lags=12, trend="n"),
    # VAR Reduce
    "VARReduceL1": VARReduce(lags=1),
    "VARReduceL3": VARReduce(lags=3),
    "VARReduceL6": VARReduce(lags=6),
    "VARReduceL12": VARReduce(lags=12),
    # Naive
    "Naive": NaiveForecaster(strategy="last"),
    "Naive3mths": NaiveForecaster(strategy="mean", window_length=3),
    "Naive6mths": NaiveForecaster(strategy="mean", window_length=6),
    # Reduced/recursive ML models
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
    # Our fallback models
    "Mean": NaiveForecaster(strategy="mean"),
    "MeanDefault": MeanDefaultForecaster(window=3),
    # EMA
    "EMA_3": EMA(span=3),
    "EMA_6": EMA(span=6),
    "EMA_9": EMA(span=9),
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
    # ARDL
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
    # ETS
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

multivar_models = {
    # VAR
    "MultiX_VARn": MultivariateModelWrapper(VAR(trend="n"), val_col=0),
    "MultiX_VARc": MultivariateModelWrapper(VAR(trend="c"), val_col=0),
    "MultiX_VARct": MultivariateModelWrapper(VAR(trend="ct"), val_col=0),
    "MultiX_VARReduceL1": MultivariateModelWrapper(VARReduce(lags=1), val_col=0),
    "MultiX_VARReduceL3": MultivariateModelWrapper(VARReduce(lags=3), val_col=0),
    "MultiX_VARReduceL6": MultivariateModelWrapper(VARReduce(lags=6), val_col=0),
    "MultiX_VARReduceL12": MultivariateModelWrapper(VARReduce(lags=12), val_col=0),
    # VECM
    "MultiX_VECM_n": MultivariateModelWrapper(VECM(deterministic="n"), val_col=0),
    "MultiX_VECM_c": MultivariateModelWrapper(VECM(deterministic="c"), val_col=0),
    "MultiX_VECM_co": MultivariateModelWrapper(VECM(deterministic="co"), val_col=0),
    "MultiX_VECM_ci": MultivariateModelWrapper(VECM(deterministic="ci"), val_col=0),
    "MultiX_VECM_lo": MultivariateModelWrapper(VECM(deterministic="lo"), val_col=0),
    "MultiX_VECM_li": MultivariateModelWrapper(VECM(deterministic="li"), val_col=0),
    "MultiX_VECM_cili": MultivariateModelWrapper(VECM(deterministic="cili"), val_col=0),
    "MultiX_VECM_colo": MultivariateModelWrapper(VECM(deterministic="colo"), val_col=0),
    # ARDL no seasonal
    "MultiX_ARDL_L1_n": MultivariateModelWrapper(
        ARDL_Sk(lags=1, trend="n", seasonal=False, auto_ardl=False), val_col=0
    ),
    "MultiX_ARDL_L3_n": MultivariateModelWrapper(
        ARDL_Sk(lags=3, trend="n", seasonal=False, auto_ardl=False), val_col=0
    ),
    "MultiX_ARDL_L6_n": MultivariateModelWrapper(
        ARDL_Sk(lags=6, trend="n", seasonal=False, auto_ardl=False), val_col=0
    ),
    "MultiX_ARDL_L12_n": MultivariateModelWrapper(
        ARDL_Sk(lags=12, trend="n", seasonal=False, auto_ardl=False), val_col=0
    ),
    "MultiX_ARDL_L1_c": MultivariateModelWrapper(
        ARDL_Sk(lags=1, trend="c", seasonal=False, auto_ardl=False), val_col=0
    ),
    "MultiX_ARDL_L3_c": MultivariateModelWrapper(
        ARDL_Sk(lags=3, trend="c", seasonal=False, auto_ardl=False), val_col=0
    ),
    "MultiX_ARDL_L6_c": MultivariateModelWrapper(
        ARDL_Sk(lags=6, trend="c", seasonal=False, auto_ardl=False), val_col=0
    ),
    "MultiX_ARDL_L12_c": MultivariateModelWrapper(
        ARDL_Sk(lags=12, trend="c", seasonal=False, auto_ardl=False), val_col=0
    ),
    "MultiX_ARDL_L1_ct": MultivariateModelWrapper(
        ARDL_Sk(lags=1, trend="ct", seasonal=False, auto_ardl=False), val_col=0
    ),
    "MultiX_ARDL_L3_ct": MultivariateModelWrapper(
        ARDL_Sk(lags=3, trend="ct", seasonal=False, auto_ardl=False), val_col=0
    ),
    "MultiX_ARDL_L6_ct": MultivariateModelWrapper(
        ARDL_Sk(lags=6, trend="ct", seasonal=False, auto_ardl=False), val_col=0
    ),
    # ARDL seasonal
    "MultiX_ARDL_L1_n_s": MultivariateModelWrapper(
        ARDL_Sk(lags=1, trend="n", seasonal=True, auto_ardl=False), val_col=0
    ),
    "MultiX_ARDL_L3_n_s": MultivariateModelWrapper(
        ARDL_Sk(lags=3, trend="n", seasonal=True, auto_ardl=False), val_col=0
    ),
    "MultiX_ARDL_L6_n_s": MultivariateModelWrapper(
        ARDL_Sk(lags=6, trend="n", seasonal=True, auto_ardl=False), val_col=0
    ),
    "MultiX_ARDL_L1_c_s": MultivariateModelWrapper(
        ARDL_Sk(lags=1, trend="c", seasonal=True, auto_ardl=False), val_col=0
    ),
    "MultiX_ARDL_L3_c_s": MultivariateModelWrapper(
        ARDL_Sk(lags=3, trend="c", seasonal=True, auto_ardl=False), val_col=0
    ),
    "MultiX_ARDL_L6_c_s": MultivariateModelWrapper(
        ARDL_Sk(lags=6, trend="c", seasonal=True, auto_ardl=False), val_col=0
    ),
    "MultiX_ARDL_L1_ct_s": MultivariateModelWrapper(
        ARDL_Sk(lags=1, trend="ct", seasonal=True, auto_ardl=False), val_col=0
    ),
    "MultiX_ARDL_L3_ct_s": MultivariateModelWrapper(
        ARDL_Sk(lags=3, trend="ct", seasonal=True, auto_ardl=False), val_col=0
    ),
    "MultiX_ARDL_L6_ct_s": MultivariateModelWrapper(
        ARDL_Sk(lags=6, trend="ct", seasonal=True, auto_ardl=False), val_col=0
    ),
    "MeanDefault": MeanDefaultMultiVar(window=3, val_col=0),
}
