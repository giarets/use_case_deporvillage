import pandas as pd
import matplotlib.pyplot as plt


def plot_time_series_by_category(
    df, category_col, target_col, time_col, agg_func="sum", window=None, legend=True
):
    """
    Plots a time series for each unique category in the specified categorical column.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        category_col (str): The name of the categorical column.
        target_col (str): The name of the target column to aggregate.
        time_col (str): The name of the column representing time (must be datetime-like).
        agg_func (str): The aggregation function to apply ('sum', 'mean', etc.).
    """
    if (
        time_col not in df.columns
        or category_col not in df.columns
        or target_col not in df.columns
    ):
        raise ValueError("One or more specified columns are not in the DataFrame.")

    # Ensure the time column is in datetime format
    df[time_col] = pd.to_datetime(df[time_col])

    # Group by the time column and the categorical column, then aggregate the target column
    grouped = (
        df.groupby([time_col, category_col])[target_col].agg(agg_func).reset_index()
    )

    # Pivot to create a separate time series for each category
    pivoted = grouped.pivot(index=time_col, columns=category_col, values=target_col)

    if window:
        pivoted = pivoted.rolling(window=window, min_periods=1).mean()

    # Plot each category's time series
    plt.figure(figsize=(12, 6))
    for category in pivoted.columns:
        plt.plot(pivoted.index, pivoted[category], label=f"{category_col}: {category}")

    plt.title(f"Time Series of {target_col} by {category_col}", fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel(f"{agg_func.capitalize()} of {target_col}", fontsize=12)
    if legend:
        plt.legend(title=category_col, fontsize=10)
    plt.grid()
    plt.tight_layout()
    plt.show()
