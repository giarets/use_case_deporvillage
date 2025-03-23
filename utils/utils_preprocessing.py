import pandas as pd
import numpy as np

COLS_NUMERICAL = ["quantity", "pvp"]
COLS_CATEGORICAL = ["product_id", "store", "seasonality", "brand", "family"]
COLS_TIMESTAMPS = ["date"]


def set_types(data):

    # Rename columns
    data = data.rename(columns={"fecha": "date"})

    # Types
    # for col in COLS_CATEGORICAL:
    #     data[col] = data[col].astype("category")
    #     data[col] = data[col].cat.remove_unused_categories()
    for col in COLS_NUMERICAL:
        data[col] = data[col].astype("float32")
    for col in COLS_TIMESTAMPS:
        data[col] = pd.to_datetime(data[col])
    return data


def aggregate_data(df, frequency="ME"):
    """
    Aggregates total quantity and revenue at the specified time frequency.
    Computes:
        - total_revenue: Total revenue generated.
    """
    df["date"] = pd.to_datetime(df["date"])

    df_grouped = (
        df.groupby(
            ["brand", "family", pd.Grouper(key="date", freq=frequency)]
        )
        .agg(
            # total_quantity=("quantity", "sum"),
            total_revenue=(
                "quantity",
                lambda x: np.sum(x * df.loc[x.index, "pvp"]),
            ),
        )
        .reset_index()
    )
    return df_grouped



def fill_in_missing_dates(df, group_col=["brand", "family"], date_col="date", freq="ME"):
    """
    Ensure that each group has all dates with a specified frequency from its 
    min to its max date. Missing rows will be forward-filled except for sales_units 
    and inventory_units which will have NaN values.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    group_col (list): Columns to group by.
    date_col (str): The name of the date column.
    freq (str): Frequency string for the date range (e.g., 'W-SAT' for weekly on Saturdays).

    Returns:
    pd.DataFrame: The completed DataFrame with all dates for each group.
    """
    if date_col not in df.columns:
        df = df.reset_index()

    df[date_col] = pd.to_datetime(df[date_col])
    original_dtype = df[group_col].dtypes
    df_dates_ranges = (
        df.groupby(group_col, observed=False)[date_col]
        .agg(["min", "max"])
        .reset_index()
    )

    df_complete = pd.DataFrame()

    # Generate all required dates for each group based on the specified frequency
    for _, row in df_dates_ranges.iterrows():
        dates = pd.date_range(start=row["min"], end=row["max"], freq=freq)

        # Create a DataFrame for this group with all dates
        temp_df = pd.DataFrame({**row.drop(["min", "max"]).to_dict(), date_col: dates})
        df_complete = pd.concat([df_complete, temp_df], ignore_index=True)

    df_complete = pd.merge(df_complete, df, on=group_col + [date_col], how="left")

    exclude_columns = []#["sales_units", "inventory_units"]
    fill_columns = [
        col
        for col in df_complete.columns
        if col not in exclude_columns + [group_col, date_col]
    ]
    df_complete[fill_columns] = df_complete.groupby(group_col)[fill_columns].ffill()

    if pd.api.types.is_categorical_dtype(original_dtype):
        df_complete[group_col] = df_complete[group_col].astype("category")

    return df_complete


def train_test_split(df, forecasting_horizon=12, target_col="y"):
    """
    Splits into training and testing set selecting last weeks from 
    the forecasting horizon.
    """

    if "date" in df.columns:
        df = df.set_index("date")
    df = df.sort_index()
    split_date = df.index.max() - pd.DateOffset(months=forecasting_horizon - 1)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train = X[X.index < split_date]
    X_test = X[X.index >= split_date]
    y_train = y[y.index < split_date]
    y_test = y[y.index >= split_date]

    return X_train, X_test, y_train, y_test
