# Change log

## 0.8.4
* Fix bug in `MeanDefaultMultiVar`

## 0.8.3
* Change multi-variate preprocessing to *left join* the external features to the main series and fill missing values using `fillna_X`

## 0.8.2
* Adjust the `model_selection` behavior to filter out `NaN` results before sorting

## 0.8.1
* Remove debugging prints
* Add row-wise metrics
* Add `MeanDefault` model in multi-variate forecasting
* Fix `MeanDefault` bug in multi-variate forecasting

## 0.8.0
* Re-design the library
* Automation config revamp
* Add parameters for more custimisation
* Fully support multi-variate forecasting along side with univariate
* Add more metrics

**Breaking changes**
* Everything, from module, names, functions, and arguments

## 0.7.0
* Re-design the back-test functions

**Breaking changes**
* Backtesting related functions and automation function is changed to support a new and more flexible way of back-testing

## 0.6.1
* Add `EMA` in `model_list`

## 0.6.0
* Add backtest evaluation method: `eval_method` parameter in backtest and automation functions

**Breaking changes**
* Add/re-arrange function arguments

## 0.5.4
* Add reduce recursive models using `Ridge` and `GradientBoosting` from `sklearn` in `model_list`

## 0.5.3
* Add models in `model_list`

## 0.5.2
* Fix import error in `automation` module. `base_models` has been moved to `model_list`

## 0.5.1
* Fix error in `forecasting` module where it couldn't import internal default `MeanDefaultForecaster`

## 0.5.0
* Add AutoTS model wrapper, in case you want to use pre-defined models from AutoTS package
* Add models in `models`. Now you can import `all_models`, `fast_models`, `base_models`

**Breaking changes**

* Dependencies
* Model imports

## 0.4.1
* Bug fix in `run_forecasting_automation()` when `return_backtest_results=True`

## 0.4.0
* Change the `run_forecasting_automation()` function behaviour
* Now this function runs the raw DataFrame and handles data preparation automatically
* It also aligns with the `prepare_timeseries()` function and handles time-series generator well
* Add parameters to the function
* Change function name to be internal use only

**Breaking changes**

* Function name changes
* Re-arrage/add/remove parameters in functions
* Return values are changed

## 0.3.0
* Add preprocessing functionalities
* Add `prepare_timeseries()` function to automate time-series preparation
* `id_cols`, `id_col` can be `None` to treat the whole DataFrame as a single series

**Breaking changes**

* Function names change
* Rearrange parameters

## 0.2.0
* Add functionalities for returning back the back-testing raw results
* Add selected models in the forecasting automation and pipeline
* Add `run_forecasting_pipeline()` function for running the pipeline just for 1 series

**Breaking change**

* Change function names
* Re-arrange some of the parameters in the affected functions

## 0.1.0
* Add forecasting automation
* Add docs

**Breaking change**

* Re-structure some of the codes and folders

## 0.0.1
* First dev version
* Give out most of the functionalities
* Code - semi-structured