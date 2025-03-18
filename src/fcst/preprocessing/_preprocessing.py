from collections.abc import Iterable
from typing import Tuple, overload

import pandas as pd

from .dataframe import prepare_forecasting_df
from .timeseries import extract_timeseries


@overload
def prepare_timeseries(
    df_raw: pd.DataFrame,
    date_col: str,
    value_col: str,
    data_period_date: pd.Period,
    id_cols: list[str],
    min_cap: int | None = 0,
    freq: str = "M",
    id_join_char: str = "_",
) -> Iterable[Tuple[str, pd.Series]]: ...


@overload
def prepare_timeseries(
    df_raw: pd.DataFrame,
    date_col: str,
    value_col: str,
    data_period_date: pd.Period,
    id_cols: None,
    min_cap: int | None = 0,
    freq: str = "M",
    id_join_char: str = "_",
) -> pd.Series: ...


def prepare_timeseries(
    df_raw: pd.DataFrame,
    date_col: str,
    value_col: str,
    data_period_date: pd.Period,
    id_cols: list[str] | None = None,
    min_cap: int | None = 0,
    freq: str = "M",
    id_join_char: str = "_",
):
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

        min_cap (int | None): Minimum value to cap before forecast
            If set, the value is used to set the minimum.
            For example, you might want to set 0 for sales.
            If None, use the existing values.

        freq (str): Frequency to resample and forecast (Default = "M")

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
        id_cols=id_cols,
        min_cap=min_cap,
        freq=freq,
        final_id_col="id",
        join_char=id_join_char,
    )

    id_col = "id" if id_cols else None

    return extract_timeseries(
        df_forecasting=df_forecasting,
        value_col=value_col,
        data_period_date=data_period_date,
        id_col=id_col,
    )
