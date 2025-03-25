import pandas as pd


def allocate_forecast(
    df_forecast: pd.DataFrame,
    df_actual: pd.DataFrame,
    forecast_col: str,
    actual_col: str,
    group_cols: list[str],
    fine_grain_cols: list[str],
    final_alloc_col: str = "forecast_allocated",
    prop_col: str = "proportion",
    actual_sum_col: str = "actual_sum",
    group_sum_col: str = "group_sum",
) -> pd.DataFrame:
    """
    Allocates forecasts based on proportions derived from actual data.

    Parameters
    ----------
        df_forecast (pd.DataFrame): DataFrame with aggregated level as index and a forecast column.

        df_actual (pd.DataFrame): DataFrame with fine-grained level as index and an actual column.

        forecast_col (str): Column name in df_forecast representing forecasted values.

        actual_col (str): Column name in df_actual representing actual sales (used to compute proportions).

        group_cols (list[str]): List of columns that define the aggregated level (e.g., ["region1", "region4", "group3"]).

        fine_grain_cols (list[str]): List of columns that define the fine-grained level (e.g., ["customer_code", "product_code", "region1", "region4", "group3"]).

        final_alloc_col (str): Allocated column name (Default = "forecast_allocated")

        prop_col (str): Proportion column (Default = "proportion")

        actual_sum_col (str): Column name of the actual summation in fine-grain level (Default = "actual_sum")

        group_sum_col (str): Column name of the summation in group level (Default = "group_sum")

    Returns
    -------
        pd.DataFrame: DataFrame with allocated forecasts
    """

    # Remove redundant columns from fine-grain columns
    fine_grain_cols = [c for c in fine_grain_cols if c not in (group_cols)]

    # Add back the columns to align 2 units of analysis
    fine_grain_cols = group_cols + fine_grain_cols

    # Step 1: Aggregate actual sales at the group level
    df_actual_sum = df_actual.groupby(fine_grain_cols)[[actual_col]].sum().reset_index()
    df_actual_sum.rename(columns={actual_col: actual_sum_col}, inplace=True)

    df_group_sum = (
        df_actual_sum.groupby(group_cols)[[actual_sum_col]].sum().reset_index()
    )
    df_group_sum.rename(columns={actual_sum_col: group_sum_col}, inplace=True)

    # Step 2: Compute proportions at the fine-grained level
    df_actual_sum = df_actual_sum.merge(df_group_sum, on=group_cols, how="inner")
    df_actual_sum[prop_col] = (
        df_actual_sum[actual_sum_col] / df_actual_sum[group_sum_col]
    )

    # Step 3: Join forecast data and allocate
    df_merged = df_forecast.merge(df_actual_sum, on=group_cols, how="left")

    # Handle missing values
    df_merged[prop_col] = df_merged[prop_col].fillna(0)
    df_merged[forecast_col] = df_merged[forecast_col].fillna(0)

    # Allocate
    df_merged[final_alloc_col] = df_merged[forecast_col] * df_merged[prop_col]

    # Step 4: Return only relevant columns with correct index
    cols_to_return = list(df_forecast.columns)
    add_dim_cols = [c for c in fine_grain_cols if c not in cols_to_return]
    measure_cols = [actual_sum_col, group_sum_col, prop_col, final_alloc_col]

    cols_to_return = cols_to_return + add_dim_cols + measure_cols

    return df_merged[cols_to_return]
