import pandas as pd
import numpy as np


def features_time_related(df, frequency="D"):
    """
    Classical time related features
    """
    frequency = "W" if "W" in frequency else "M" if "M" in frequency else "D"

    df["year"] = df["date"].dt.year
    df["month_of_year"] = df["date"].dt.month


    if frequency in ["D", "W"]:
        # Features available for both daily and weekly frequencies
        df["week_of_year"] = df["date"].dt.isocalendar().week

        # Trigonometric encoding for cyclic time
        df["sin_week_of_year"] = np.sin(2 * np.pi * df["week_of_year"] / 52).round(3)
        df["cos_week_of_year"] = np.cos(2 * np.pi * df["week_of_year"] / 52).round(3)

        df["sin_month_of_year"] = np.sin(2 * np.pi * df["month_of_year"] / 12).round(3)
        df["cos_month_of_year"] = np.cos(2 * np.pi * df["month_of_year"] / 12).round(3)

    if frequency == "D":
        # Daily-specific features
        df["day_of_month"] = df["date"].dt.day
        df["week_of_month"] = (df["date"].dt.day - 1) // 7 + 1

        df["sin_day_of_month"] = np.sin(2 * np.pi * df["day_of_month"] / 31).round(3)
        df["cos_day_of_month"] = np.cos(2 * np.pi * df["day_of_month"] / 31).round(3)

    if frequency == "W":
        # Compute week_of_month for weekly data
        df["first_day_of_month"] = df["date"] - pd.to_timedelta(
            df["date"].dt.day - 1, unit="D"
        )
        df["week_of_month"] = (
            (df["week_of_year"] - df["first_day_of_month"].dt.isocalendar().week) + 1
        ).clip(lower=1)
        df.drop(columns=["first_day_of_month"], inplace=True)

    if frequency == "M":
        # Monthly data only has month-related features
        df["sin_month_of_year"] = np.sin(2 * np.pi * df["month_of_year"] / 12).round(3)
        df["cos_month_of_year"] = np.cos(2 * np.pi * df["month_of_year"] / 12).round(3)
    return df


def features_lag(df, col, lags=[12], group_column="sku"):
    """
    Creates lagged features for a given column within groups in a pandas DataFrame.
    """
    for lag in lags:
        df[f"{col}_lag_{lag}"] = df.groupby(group_column, observed=False)[col].shift(
            lag
        )

    return df


def features_rolling(df, col, window_sizes, group_column="sku"):
    """
    Creates rolling features for a given column within groups in a pandas DataFrame.
    """
    for window in window_sizes:
        df[f"{col}_rolling_mean_{window}w"] = df.groupby(group_column, observed=False)[
            col
        ].transform(lambda x: x.shift(13).rolling(window, min_periods=1).mean())
        df[f"{col}_rolling_std_{window}w"] = df.groupby(group_column, observed=False)[
            col
        ].transform(lambda x: x.shift(13).rolling(window, min_periods=1).std())
        df[f"{col}_rolling_sum_{window}w"] = df.groupby(group_column, observed=False)[
            col
        ].transform(lambda x: x.shift(13).rolling(window, min_periods=1).sum())
        # df[f'{col}_rolling_sum_{window}w'] = df.groupby('sku', observed=False)[col].transform(lambda x: x.shift(13).rolling(window, min_periods=1).min())
        # df[f'{col}_rolling_sum_{window}w'] = df.groupby('sku', observed=False)[col].transform(lambda x: x.shift(13).rolling(window, min_periods=1).max())
    return df


def create_periods_feature(df, group_col, date_col, target_col):
    """
    Create a new feature 'feature_periods' that counts the number of weeks since
    the first non-zero signal for each group, based on the row order.

    Parameters:
    - df: pandas DataFrame
    - coll_agg: key of the dataframe (excluding date)
    - date_column: the column containing dates
    - target_col: the column used to start counting when its value is greater than 0

    Returns:
    - pandas DataFrame with new column feature_periods
    """

    df = df.sort_values(by=group_col + [date_col])

    # Create a mask to indicate rows where the signal_col is greater than 0
    df["signal_above_zero"] = df[target_col] > 0

    # Group by the coll_agg and create a cumulative sum of the signal_above_zero mask
    # Start counting periods only when the signal_col is greater than 0
    df["first_nonzero_signal"] = df.groupby(group_col)["signal_above_zero"].cumsum() > 0

    # Count periods only where the signal has been greater than zero
    df["feature_periods"] = df.groupby(group_col).cumcount() + 1
    df["feature_periods"] = df["feature_periods"].where(df["first_nonzero_signal"], 0)

    df["feature_periods"] = df["feature_periods"].astype("float64")
    df = df.reset_index(drop=True)
    df = df.drop(columns=["signal_above_zero", "first_nonzero_signal"])

    return df
