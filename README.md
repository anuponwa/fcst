# fcst
![Publish Tag to PyPI](https://github.com/anuponwa/fcst/actions/workflows/publish-tag-to-pypi.yml/badge.svg)

Package repo on PyPI: [fcst - PyPI](https://pypi.org/project/fcst/)

## Installation
```bash
uv add fcst
```

## Features
This package provides you with these sub-modules
1. automation
    This automatically runs back-test, select the best models, and forecast for you.
    You can customise whether or not to run in parallel, how many top models to select, etc.
2. forecasting
    This provides you with the basic functionality of `fit()` and `predict()`, given that you pass in the model.
3. evaluation
    This provides you with back-test and model selection functionalities.
4. preprocessing
    This allows you to prepare your dataframes, preprocess the time-series data, fill in the missing dates automatically.
5. horizon
    This is an API for dealing with future horizon from `sktime`. But in some modules, it will also do this automatically.
6. models
    Gives you the base models for you to work with. Provides you with the basic models, default (fallback) and zero predictor.
7. metrics
    Our own implementation of forecasting performance metrics.
8. common
    Other common functionalities, e.g., types.

## Usage

**Examples**
```python
from fcst.automation import run_forecasting_automation
from fcst.preprocessing.dataframe import prepare_forecasting_df
from fcst.models import base_models
import pandas as pd


df_input = pd.read_csv("path-to-your/file.csv")

data_period_date = pd.Period("2025-02", freq="M")

df_forecasting = prepare_forecasting_df(
    df_raw=df_input,
    min_cap=0,  # Cap the value to not go under 0
    freq="M",
)

df_forecasting_results = run_forecasting_automation(
    df_forecasting,
    value_col=net_amount,
    data_period_date=data_period_date,
    models=base_models,
    backtest_periods=5,
    eval_periods=2,
    top_n=2,
    forecasting_periods=3,
    parallel=True,
)

# Do something with the results
def format_and_upload_results(df_results):
    ...
```