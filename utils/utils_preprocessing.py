import pandas as pd

COLS_NUMERICAL = ["quantity", "pvp"]
COLS_CATEGORICAL = ["product_id", "store", "seasonality", "brand", "family"]
COLS_TIMESTAMPS = ["date"]


def preprocess_data(data):

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
