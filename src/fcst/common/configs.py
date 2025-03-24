from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from ..models._models import MeanDefaultForecaster
from ..models.model_list import fast_models, multivar_models
from .types import ModelDict, Forecaster


@dataclass
class DataProcessingConfig:
    freq: str = field(
        default="M", metadata={"help": " Frequency to resample and forecast"}
    )

    min_cap: int | None = field(
        default=0, metadata={"help": "Minimum value to cap before forecast"}
    )

    agg_method: Literal["sum", "mean"] = field(
        default="sum",
        metadata={"help": "String specifying aggregation method to value column"},
    )

    fillna: Literal["bfill", "ffill"] | int | float = field(
        default=0, metadata={"help": "Method or number to fill missing values"}
    )


@dataclass
class BacktestingConfig:
    """Config for back-testing in automation process"""

    backtest_periods: int = field(
        default=1, metadata={"help": "Number of periods to back-test"}
    )

    eval_periods: int = field(
        default=6,
        metadata={"help": "Number of periods to evaluate in each rolling back-test"},
    )

    keep_eval_fixed: bool = field(
        default=False, metadata={"help": "Whether or not to keep eval_periods fixed"}
    )


@dataclass
class ForecastingConfig:
    """Config for forecasting"""

    min_forecast: float | int | None = field(
        default=0, metadata={"help": "Minimum value to cap for forecast values"}
    )

    max_forecast_factor: float | int | None = field(
        default=2.5,
        metadata={
            "help": "The factor of maximum forecast relative to the maximum history"
        },
    )

    top_n: int = field(
        default=3, metadata={"help": "Top N models to use in ensemble forecast"}
    )

    models: ModelDict = field(
        default_factory=lambda: fast_models,
        metadata={"help": "A dictionary of models to use in forecasting"},
    )

    fallback_model: Forecaster = field(
        default=MeanDefaultForecaster(window=3),
        metadata={
            "help": "A model used as a fall-back when the number of data points is too low"
        },
    )

    fcst_col_index: int = field(
        default=0,
        metadata={
            "help": "Index of the column of interest in foreasting (only applies if `series` is pd.DataFrame) (Default = 0 (first column))"
        },
    )


@dataclass
class MultiVarConfig:
    df_X_raw: pd.DataFrame = field(
        metadata={
            "help": "Raw DF for external features that has a date column, other info, and the values to forecast"
        },
    )

    feature_cols: list[str] | None = field(metadata={"help": "List of feature columns"})

    min_caps_X: float | int | dict[str, float | int] | None = field(
        default=0, metadata={"help": "Minimum value to cap before forecast"}
    )

    agg_methods_X: Literal["sum", "mean"] | dict[str, Literal["sum", "mean"]] = field(
        default="sum",
        metadata={"help": "String specifying aggregation method to value column"},
    )

    fillna_X: Literal["bfill", "ffill"] | int | float = field(
        default=0, metadata={"help": "Method or number to fill missing values"}
    )

    multivar_models: ModelDict = field(
        default_factory=lambda: multivar_models,
        metadata={"help": "A dictionary of models to use in forecasting"},
    )


@dataclass
class MultiProcessingConfig:
    parallel: bool = field(
        default=True, metadata={"help": " Whether or not to utilise parallisation"}
    )
    n_jobs: int = field(
        default=-1, metadata={"help": "Number of jobs in parallelisation"}
    )


df_raw: pd.DataFrame
date_col: str
value_col: str
data_period_date: pd.Period
forecasting_periods: int
id_cols: list[str] | None = None
id_join_char: str = "_"
return_backtest_results: bool = False
