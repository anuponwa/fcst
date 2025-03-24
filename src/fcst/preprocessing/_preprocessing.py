from collections.abc import Iterable
from typing import Literal, Tuple

import pandas as pd


def fill_missing_dates(
    series: pd.Series | pd.DataFrame,
    data_period_date: pd.Period,
    fillna: Literal["bfill", "ffill"] | int | float = 0,
) -> pd.Series | pd.DataFrame:
    """Fills in missing dates with value 0 in the provided series

    Parameters
    ----------
        series (pd.Series | pd.DataFrame): Time-series with PeriodIndex as index

        data_period_date (pd.Period): Ending date for data to use for training

        fillna (Literal["bfill", "ffill"] | int | float, optional (Default = 0))
            Method to fill missing values:
            - int/float: Fill with a specific number (e.g., 0).
            - "ffill": Forward fill.
            - "bfill": Backward fill.

    Returns
    -------
        pd.Series | pd.DataFrame
    """

    # Find min date and create the full date range
    min_period = series.index.min()
    full_period_idx = pd.period_range(min_period, data_period_date, freq="M")

    # Re-index the dataframe with the full indices
    series = series.reindex(full_period_idx)

    if fillna in ["bfill", "ffill"]:
        if fillna == "bfill":
            series = series.bfill()
        elif fillna == "ffill":
            series = series.ffill()
    else:
        series = series.fillna(fillna)

    return series


def prepare_forecasting_df(
    df_raw: pd.DataFrame,
    date_col: str,
    value_col: str,
    data_period_date: pd.Period,
    id_cols: list[str] | None = None,
    min_cap: float | int | None = 0,
    freq: str = "M",
    agg_method: Literal["sum", "mean"] = "sum",
    fillna: Literal["bfill", "ffill"] | int | float = 0,
    join_char: str = "_",
) -> pd.DataFrame:
    """Process and prepares DF for forecasting

    Parameters
    ----------
        df_raw (pd.DataFrame): Raw DF that has a date column, other info, and the value to forecast

        date_col (str): The date column to use in forecasting

        value_col (str): The value column to forecast

        date_period_date (pd.Period): Ending date for data to use for training

        id_cols (list[str] | None): A list containing the column names to create a unique time-series ID (Default is None)
            If None, the whole dataframe is treated as a single time-series
            If a list of columns is passed in, a new "id" index will be created

        min_cap (float | int | None): Minimum value to cap before forecast
            If set, the value is used to set the minimum.
            For example, you might want to set 0 for sales.
            If None, use the existing values.

        freq (str): Frequency to resample and forecast (Default = "M")

        agg_methods (Literal["sum", "mean"]): String specifying aggregation method to value column (Default = "sum")

        fillna (Literal["bfill", "ffill"] | int | float): Method or number to fill missing values (Default = 0)
            Method to fill missing values:
            - int/float: Fill with a specific number (e.g., 0).
            - "ffill": Forward fill.
            - "bfill": Backward fill.

        join_char (str): A character to join multiple ID columns (Default = "_")

    Returns
    -------
        pd.DataFrame:
            Where the index is the pd.PeriodIndex,
            and the columns are id and value.
            The values are resampled to the specified `freq`.
    """

    df_raw = df_raw.copy()

    # Prepare columns
    df_raw[date_col] = pd.PeriodIndex(df_raw[date_col], freq=freq)
    df_raw[value_col] = df_raw[value_col].astype(float)

    # Filter data
    df_raw = df_raw[df_raw[date_col] <= data_period_date]

    if id_cols:
        df_raw["id"] = df_raw[id_cols].astype(str).agg(join_char.join, axis=1)
    else:
        df_raw["id"] = "global"

    columns = ["id", date_col, value_col]
    groupby_cols = ["id", date_col]

    df_raw = df_raw[columns]

    # Clean min_cap
    if min_cap is not None:
        df_raw.loc[df_raw[value_col] < min_cap, value_col] = min_cap

    # Aggregate as needed
    if agg_method == "mean":
        df_forecasting = df_raw.groupby(groupby_cols).mean(value_col)
    else:
        df_forecasting = df_raw.groupby(groupby_cols).sum(value_col)

    y_prep = []
    unique_ids = df_forecasting.index.get_level_values(0).unique()
    for id_ in unique_ids:
        df_y_i = df_forecasting.loc[(id_,), [value_col]]
        df_y_i = fill_missing_dates(df_y_i, data_period_date, fillna)
        df_y_i["id"] = id_
        df_y_i = df_y_i.set_index("id", append=True)
        df_y_i.index = df_y_i.index.swaplevel()
        y_prep.append(df_y_i)

    df_forecasting = pd.concat(y_prep)
    df_forecasting = df_forecasting.sort_index()

    return df_forecasting


def prepare_timeseries(
    df_raw: pd.DataFrame,
    date_col: str,
    value_col: str,
    data_period_date: pd.Period,
    id_cols: list[str] | None = None,
    min_cap: float | int | None = 0,
    freq: str = "M",
    agg_method: Literal["sum", "mean"] = "sum",
    fillna: Literal["bfill", "ffill"] | int | float = 0,
    id_join_char: str = "_",
) -> dict[str, pd.Series]:
    """Prepares time-series from Raw DataFrame

    Parameters
    ----------
        df_raw (pd.DataFrame): Raw DF that has a date column, other info, and the value to forecast

        date_col (str): The date column to use in forecasting

        value_col (str): The value column to forecast

        date_period_date (pd.Period): Ending date for data to use for training

        id_cols (list[str] | None): A list containing the column names to create a unique time-series ID (Default is None)
            If None, the whole dataframe is treated as a single time-series
            If a list of columns is passed in, a new "id" index will be created

        min_cap (float | int | None): Minimum value to cap before forecast
            If set, the value is used to set the minimum.
            For example, you might want to set 0 for sales.
            If None, use the existing values.

        freq (str): Frequency to resample and forecast (Default = "M")

        agg_methods (Literal["sum", "mean"]): String specifying aggregation method to value column (Default = "sum")

        fillna (Literal["bfill", "ffill"] | int | float): Method or number to fill missing values (Default = 0)
            Method to fill missing values:
            - int/float: Fill with a specific number (e.g., 0).
            - "ffill": Forward fill.
            - "bfill": Backward fill.

        id_join_char (str): A character to join multiple ID columns (Default = "_")

    Returns
    -------
        Iterable[Tuple[str, pd.Series]]: A tuple of ID name and its time-series (when `id_col` is passed in)

        pd.Series: A single series is returned (when `id_col` is None)
    """

    df_forecasting = prepare_forecasting_df(
        df_raw=df_raw,
        date_col=date_col,
        value_col=value_col,
        data_period_date=data_period_date,
        id_cols=id_cols,
        min_cap=min_cap,
        freq=freq,
        agg_method=agg_method,
        fillna=fillna,
        join_char=id_join_char,
    )

    ret_dict = {}

    unique_ids = df_forecasting.index.get_level_values(0).unique()
    for id_ in unique_ids:
        series = df_forecasting.loc[(id_,), value_col]

        # Gracefully handle the missing sales after filtering
        if len(series) == 0:
            continue

        ret_dict[id_] = series

    return ret_dict


def prepare_X_df(
    df_raw: pd.DataFrame,
    date_col: str,
    feature_cols: list[str],
    data_period_date: pd.Period,
    id_cols: list[str] | None = None,
    min_caps: float | int | dict[str, float | int] | None = 0,
    freq: str = "M",
    agg_methods: Literal["sum", "mean"] | dict[str, Literal["sum", "mean"]] = "sum",
    fillna: Literal["bfill", "ffill"] | int | float = 0,
    join_char: str = "_",
) -> pd.DataFrame:
    """Processes and prepares feature DataFrame X for forecasting.

    Parameters
    ----------
        df_raw (pd.DataFrame): Raw DF that has a date column, other info, and the values to forecast

        date_col (str): The date column to use in forecasting

        feature_cols (list[str]): List of feature columns

        date_period_date (pd.Period): Ending date for data to use for training

        id_cols (list[str] | None): A list containing the column names to create a unique time-series ID (Default is None)
            If None, the whole dataframe is treated as a single time-series
            If a list of columns is passed in, a new "id" index will be created

        min_caps (float | int | dict[str, float | int] | None): Minimum value to cap before forecast
            If set, the value is used to set the minimum.
            For example, you might want to set 0 for sales.
            If None, use the existing values.
            It can also be a dictionary, e.g.,
                min_caps = {"feature_1": 0}

        freq (str): Frequency to resample and forecast (Default = "M")

        agg_methods (Literal["sum", "mean"]): String specifying aggregation method to value column (Default = "sum")

        fillna (Literal["bfill", "ffill"] | int | float): Method or number to fill missing values (Default = 0)
            Method to fill missing values:
            - int/float: Fill with a specific number (e.g., 0).
            - "ffill": Forward fill.
            - "bfill": Backward fill.

        join_char (str): A character to join multiple ID columns (Default = "_")

    Returns
    -------
        pd.DataFrame
            DataFrame with `final_date_col`, `final_id_col`, and feature columns.
    """

    df_raw = df_raw.copy()

    # Convert date column to PeriodIndex
    df_raw[date_col] = pd.PeriodIndex(df_raw[date_col], freq=freq)

    # Convert every feature to float
    for feat in feature_cols:
        df_raw[feat] = df_raw[feat].astype(float)

    # Filter data
    df_raw = df_raw[df_raw[date_col] <= data_period_date]

    # Process ID column
    if id_cols:
        df_raw["id"] = df_raw[id_cols].astype(str).agg(join_char.join, axis=1)
    else:
        df_raw["id"] = "global"

    groupby_cols = ["id", date_col]

    # Select required columns
    columns = [date_col, "id"] + feature_cols
    df_X = df_raw[columns]

    # Clean min_cap
    if min_caps is not None:
        if isinstance(min_caps, dict):
            for feat_col, min_cap in min_caps.items():
                df_X.loc[df_X[feat_col] < min_cap, feat_col] = min_cap
        else:
            min_cap = min_caps
            for feat_col in feature_cols:
                df_X.loc[df_X[feat_col] < min_cap, feat_col] = min_cap

    # Aggregate (if needed)
    if isinstance(agg_methods, str):
        agg_methods = {col: agg_methods for col in feature_cols}

    if isinstance(agg_methods, dict):
        cols_diff = set(feature_cols) - set(agg_methods.keys())
        for col in cols_diff:
            agg_methods[col] = "sum"  # Use sum for the rest of the columns

        agg_methods = {c: agg_methods[c] for c in feature_cols}  ## Re-arrage columns

    df_X = df_X.groupby(groupby_cols).agg(agg_methods)

    X_prep = []
    unique_ids = df_X.index.get_level_values(0).unique()
    for id_ in unique_ids:
        df_X_i = df_X.loc[(id_,), feature_cols]
        df_X_i = fill_missing_dates(df_X_i, data_period_date, fillna)
        df_X_i["id"] = id_
        df_X_i = df_X_i.set_index("id", append=True)
        df_X_i.index = df_X_i.index.swaplevel()
        X_prep.append(df_X_i)

    df_X_prep = pd.concat(X_prep)
    df_X_prep = df_X_prep.sort_index()

    return df_X_prep


def prepare_multivar_timeseries(
    df_raw: pd.DataFrame,
    df_X_raw: pd.DataFrame,
    date_col: str,
    value_col: str,
    feature_cols: list[str],
    data_period_date: pd.Period,
    id_cols: list[str] | None = None,
    min_cap: float | int | None = 0,
    min_caps_X: float | int | dict[str, float | int] | None = 0,
    freq: str = "M",
    agg_method: Literal["sum", "mean"] = "sum",
    agg_methods_X: Literal["sum", "mean"] | dict[str, Literal["sum", "mean"]] = "sum",
    fillna: Literal["bfill", "ffill"] | int | float = 0,
    fillna_X: Literal["bfill", "ffill"] | int | float = 0,
    id_join_char: str = "_",
) -> dict[str, pd.DataFrame]:
    df_y = prepare_forecasting_df(
        df_raw=df_raw,
        date_col=date_col,
        value_col=value_col,
        data_period_date=data_period_date,
        id_cols=id_cols,
        min_cap=min_cap,
        freq=freq,
        agg_method=agg_method,
        fillna=fillna,
        join_char=id_join_char,
    )

    df_X = prepare_X_df(
        df_raw=df_X_raw,
        date_col=date_col,
        feature_cols=feature_cols,
        data_period_date=data_period_date,
        id_cols=id_cols,
        min_caps=min_caps_X,
        freq=freq,
        agg_methods=agg_methods_X,
        fillna=fillna_X,
        join_char="_",
    )

    df_total = df_y.merge(df_X, left_index=True, right_index=True, how="left")

    # Fill NA values for the missing external features
    if fillna_X in ["bfill", "ffill"]:
        if fillna_X == "bfill":
            df_total = df_total.bfill()
        elif fillna_X == "ffill":
            df_total = df_total.ffill()
    else:
        df_total = df_total.fillna(fillna_X)

    ret_dict = {}
    unique_ids = df_total.index.get_level_values(0).unique()
    for id_ in unique_ids:
        df_y_X = df_total.loc[(id_,), :]

        # Gracefully handle the missing sales after filtering
        if len(df_y_X) == 0:
            continue

        ret_dict[id_] = df_y_X

    return ret_dict
