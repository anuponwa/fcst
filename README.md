# fcst
![Publish Tag to PyPI](https://github.com/anuponwa/fcst/actions/workflows/publish-tag-to-pypi.yml/badge.svg)

Package repo on PyPI: [fcst - PyPI](https://pypi.org/project/fcst/)

## Installation
```bash
uv add fcst
```

## Features
This package provides you with these sub-modules
1. **automation**

    This automatically runs back-test, select the best models, and forecast for you.
    You can customise whether or not to run in parallel, how many top models to select, etc.

2. **forecasting**

    This provides you with the basic functionality of `fit()` and `predict()`, given that you pass in the model.

3. **evaluation**

    This provides you with back-test and model selection functionalities.

4. **preprocessing**

    This allows you to prepare your dataframes, preprocess the time-series data, fill in the missing dates automatically.

5. **horizon**

    This is an API for dealing with future horizon from `sktime`. But in some modules, it will also do this automatically.

6. **models**

    Gives you the base models for you to work with. Provides you with the basic models, default (fallback) and zero predictor.

7. **metrics**

    Our own implementation of forecasting performance metrics.

8. **common**

    Other common functionalities, e.g., types.


## Usage

**Examples**

In case you want to automate the whole process...

```python
from fcst.automation import run_forecasting_automation
import pandas as pd


df_input = pd.read_csv("path-to-your/file.csv")

data_period_date = pd.Period("2025-02", freq="M")

results = run_forecasting_automation(
    df_raw=df_input,
    date_col="date",
    value_col="net_amount",
    data_period_date=data_period_date,
    backtest_periods=3,
    eval_periods=2,
    top_n = 2,
    forecasting_periods=2,
    id_cols=["customer_code", "product_code"],  # In case the unit of analysis is: customer, product
    min_cap=0,  # clean the value for some rows that have < 0
    freq="M",  # Monthly forecast
    parallel=True,
)

# Do something with the results
def format_and_upload_results(df_results):
    ...
```

```python
from fcst.automation import run_forecasting_automation
import pandas as pd


df_input = pd.read_csv("path-to-your/file.csv")

data_period_date = pd.Period("2025-02", freq="M")

results = run_forecasting_automation(
    df_raw=df_input,
    date_col="date",
    value_col="net_amount",
    data_period_date=data_period_date,
    backtest_periods=3,
    eval_periods=2,
    top_n = 2,
    forecasting_periods=2,
    id_cols=None,  # In case you want to predict the whold dataframe
    min_cap=0,  # clean the value for some rows that have < 0
    freq="M",  # Monthly forecast
    parallel=True,
)

# Do something with the results
def format_and_upload_results(df_results):
    ...
```

Other utilities for time-series

```python
from fcst.preprocessing import prepare_timeseries


# Group time-series based on customer-product, then yields a generater of (id_, pd.Series)
timeseries = prepare_timeseries(
    df_raw=df_input,
    date_col="date",
    value_col="net_amount",
    data_period_date=data_period_date,
    id_cols=["customer_code", "product_code"],
)


# Returns the whole DataFrame as a time-series
timeseries = prepare_timeseries(
    df_raw=df_input,
    date_col="date",
    value_col="net_amount",
    data_period_date=data_period_date,
    id_cols=None,
)
```

There are also these two functions from sub-modules, if you want to explore:
* `from fcst.preprocessing.dataframe import prepare_forecasting_df`
* `from fcst.preprocessing.timeseries import extract_timeseries`


## More detailed usage

### Automation

This `automation` sub-module runs cleaning, fill-in the missing dates, evaluation, model selection, forecasting and ensemble, everything automatically.


### Preprocessing

This provides time-series preparation functions. The most complete function and does most of the heavy lifting is `prepare_timeseries()` function.
You can import it from `from fcst.preprocessing import prepare_timeseries`.
You can specify ID columns, date column, value column to forecast, the frequency, and much more.
This function returns either a single time-series or a generator, depending on the `id_cols` parameter.

If you want to go into details or want full control in each step, you can use these two functions:
* `from fcst.preprocessing.dataframe import prepare_forecasting_df`
* `from fcst.preprocessing.timeseries import extract_timeseries`


### Models

By default the `run_forecasting_automation()` uses `base_models` from `models`.
But you can define your own model(s) with `fit()` and `predict()` methods.
You can get the `base_model` and put your own model(s) to the dictionary.


### Metrics

By default, we use `mape()` for measuring accuracy (error) of forecasting models.
We define our own as it handles when the forecast or actual values are 0.
Our `mape()` function also has `symmetric` flag to calculate the sMAPE.


### Horizon

If you want to utilise the forecast horizons or get some future dates, this sub-module provides the basic functionalities.
And this is the base of `forecasting` and `evaluation` functions.


### Forecasting

In `forecasting` sub-module, you can use `forecast` or `ensemble` to forecast using one or more models.


### Evaluation

The `evaluation` sub-module provides the back-testing and model selection functions.
You can pass in a model dictionary to evaluate which models are suitable for each time-series.

