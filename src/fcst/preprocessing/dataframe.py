import pandas as pd


def process_forecasting_df(
    df_raw: pd.DataFrame,
    id_cols: list[str],
    date_col: str,
    value_col: str,
    min_cap: int | None = 0,
    freq: str = "M",
    final_id_col: str = "id",
    final_date_col: str = "date",
) -> pd.DataFrame:
    """Process and prepares DF for forecasting

    Parameters
    ----------
    df_raw (pd.DataFrame): Raw DF that has a date column, other info, and the value to forecast

    id_cols (list[str]): A list containing the column names to create a unique time-series ID

    date_col (str): The date column to use in forecasting

    value_col (str): The value column to forecast

    min_cap (int | None): Minimum value to cap before forecast
        If set, the value is used to set the minimum.
        For example, you might want to set 0 for sales.
        If None, use the existing values.

    freq (str): Frequency to resample and forecast (Default = "M")

    final_id_col (str): The final concatenated ID column name

    final_date_col (str): The final date column name

    Returns
    -------
    pd.DataFrame:
        Where the index is the pd.PeriodIndex,
        and the columns are id and value.
        The values are resampled to the specified `freq`.
    """

    # Dynamically concatenate ID columns
    df_raw[final_id_col] = df_raw[id_cols].astype(str).agg("_".join, axis=1)

    df_raw[final_date_col] = pd.PeriodIndex(df_raw[date_col], freq=freq)

    df_raw = df_raw[[final_date_col, final_id_col, value_col]]

    # Clean min_cap
    df_raw.loc[df_raw[value_col] < min_cap, value_col] = min_cap

    # Sum group by frequency and sort
    df_forecasting = (
        df_raw.groupby([final_date_col, final_id_col]).sum(value_col).reset_index()
    )

    df_forecasting = df_forecasting.sort_values(by=[final_id_col, final_date_col])
    df_forecasting = df_forecasting.set_index(final_date_col)

    return df_forecasting
