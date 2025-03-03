from collections.abc import Iterable
from typing import Tuple, overload
import pandas as pd


def fill_missing_dates(
    series: pd.Series,
    data_period_date: pd.Period,
) -> pd.DataFrame:
    """Fills in missing dates with value 0 in the provided series"""

    # Find min date and create the full date range
    min_period = series.index.min()
    full_period_idx = pd.period_range(min_period, data_period_date, freq="M")

    # Re-index the dataframe with the full indices
    series = series.reindex(full_period_idx)
    series = series.fillna(0)  # Fill missing dates' values

    return series


@overload
def extract_timeseries(
    df_forecasting: pd.DataFrame,
    value_col: str,
    data_period_date: pd.Period,
    id_col: str,
) -> Iterable[Tuple[str, pd.Series]]: ...


@overload
def extract_timeseries(
    df_forecasting: pd.DataFrame,
    value_col: str,
    data_period_date: pd.Period,
    id_col: None,
) -> pd.Series: ...


def extract_timeseries(
    df_forecasting: pd.DataFrame,
    value_col: str,
    data_period_date: pd.Period,
    id_col: str | None = None,
):
    """Generates each time-series for forecasting and its corresponding ID

    Parameters
    ----------
        df_forecasting (pd.DataFrame): Preprocessed DF for forecasting
            Where the index is the pd.PeriodIndex,
            and the columns are id and value.
            The values are resampled to the specified `freq`.

        value_col (str): The value column to forecast

        date_period_date (pd.Period): Ending date for data to use for training

        id_col (str): ID column name used to distinguish time-series (Default is None)
            If None, the whole DataFrame is treated as a single series.
            The output will be 1 series.
            If a column ID is provided, the output will be multiple time-series.

    Returns
    -------
        Iterable[Tuple[str, pd.Series]]: A tuple of ID name and its time-series (when `id_col` is passed in)

        pd.Series: A single series is returned (when `id_col` is None)
    """

    if not id_col:
        series = df_forecasting[value_col]
        series = fill_missing_dates(series, data_period_date)
        return series.loc[series.index <= data_period_date]

    def ts_generator():
        unique_ids = df_forecasting[id_col].unique()
        for id_ in unique_ids:
            series = df_forecasting.loc[df_forecasting[id_col] == id_, value_col]
            series = fill_missing_dates(series, data_period_date)

            # Gracefully handle the missing sales after filtering
            if len(series) == 0:
                continue

            yield (id_, series.loc[series.index <= data_period_date])

    return ts_generator()
