from abc import ABC, abstractmethod
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class AbstractForecastingModel(ABC):
    """An abstract base class for time-series forecasting models."""

    def __init__(self, hyperparameters=None):
        self.hyperparameters = hyperparameters if hyperparameters is not None else {}
        self.model = None
        self.model = self.initialize_model()

    @abstractmethod
    def initialize_model(self):
        pass

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, y_pred, y_test):
        rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 3)
        return rmse

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, filename):
        model_and_metadata = {
            "model": self.model,
            "hyperparameters": self.hyperparameters,
        }
        with open(filename, "wb") as file:
            pickle.dump(model_and_metadata, file)

    def load_model(self, filename):
        with open(filename, "rb") as file:
            loaded_data = pickle.load(file)
            self.model = loaded_data["model"]
            self.hyperparameters = loaded_data["hyperparameters"]

    def cross_validate(self, df, n_splits=4):
        """
        Perform time-series cross-validation using TimeSeriesSplit and return average RMSE.
        If the approach is bottom-up it also outputs the aggregated RMSE.

        Parameters:
        -----------
        df : The input dataframe containing time-series data, including target variable ('y'),
             features, and time-related columns (such as 'product_number', 'id', 'year_week', etc.).
        n_splits : The number of splits for cross-validation. It controls the number of train-test
                   splits to perform.

        Returns:
        --------
        float
            The average Root Mean Square Error (RMSE) over all cross-validation splits.
        """
        metrics = []
        predictions_list = []
        unique_dates = pd.Series(df.index.unique()).sort_values()

        tss = TimeSeriesSplit(n_splits, test_size=12)

        for train_idx, test_idx in tss.split(unique_dates):
            train_dates, test_dates = (
                unique_dates.iloc[train_idx],
                unique_dates.iloc[test_idx],
            )

            train_data = df[df.index.isin(train_dates)]
            test_data = df[df.index.isin(test_dates)]

            X_train, y_train = train_data.drop(columns=["y"]), train_data["y"]
            X_test, y_test = test_data.drop(columns=["y"]), test_data["y"]

            print(
                f"Train [{X_train.index.min().date()} - {X_train.index.max().date()}]"
            )
            print(
                f"Predict [{X_test.index.min().date()} - {X_test.index.max().date()}]"
            )

            self.train(X_train, y_train)
            y_pred = self.predict(X_test)

            score = self.evaluate(y_pred, y_test)
            metrics.append(score)

            predictions_df = pd.DataFrame(
                {
                    "date": test_data.index,
                    "brand": test_data["brand"],
                    "family": test_data["family"],
                    "y_pred": y_pred,
                    "y": y_test,
                }
            )
            predictions_list.append(predictions_df)

        average_rmse = np.mean(metrics)
        print(f"Average RMSE from cross-validation: {average_rmse:.4f}")

        return average_rmse


class NaiveRollingMean(AbstractForecastingModel):
    """
    A naive forecasting model that predicts future values using the most recent rolling mean
    of inventory units based on a specified window size.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window = None
        self.column = self.initialize_model()
        self.col_group = ["brand", "family"]

    def initialize_model(self):
        if self.hyperparameters is None or "window" not in self.hyperparameters:
            raise ValueError("Hyperparameter 'window' is required but missing.")
        self.window = self.hyperparameters["window"]
        return f"total_revenue_rolling_mean_{self.window}w"

    def train(self, X_train, y_train):
        pass

    def predict(self, X):
        if self.column not in X.columns:
            raise ValueError(f"{self.column} is missing from input data.")

        last_values_per_sku = (
            X.sort_index()
            .groupby(self.col_group, observed=False)[self.column]
            .last()
            .to_dict()
        )
        X['brand_family'] = list(zip(X['brand'], X['family']))
        result = X['brand_family'].map(last_values_per_sku).fillna(0)

        return result.values


class NaiveLag(AbstractForecastingModel):
    """
    A naive forecasting model that predicts future values using a specific lag value.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lag = None
        self.column = self.initialize_model()
        self.col_group = ["brand", "family"]

    def initialize_model(self):
        if self.hyperparameters is None or "lag" not in self.hyperparameters:
            raise ValueError("Hyperparameter 'lag' is required but missing.")
        self.lag = self.hyperparameters["lag"]
        return f"total_revenue_lag_{self.lag}"

    def train(self, X_train, y_train):
        pass

    def predict(self, X):
        if self.column not in X.columns:
            raise ValueError(f"{self.column} is missing from input data.")

        last_values_per_sku = (
            X.sort_index().groupby(self.col_group, observed=False)[self.column].last().to_dict()
        )
        X['brand_family'] = list(zip(X['brand'], X['family']))
        result = X['brand_family'].map(last_values_per_sku).fillna(0)

        return result.values


class LightGBMForecastingModel(AbstractForecastingModel):
    """
    LightGBM model
    """

    def initialize_model(self):
        return LGBMRegressor(**self.hyperparameters)

    def plot_feature_importance(self, importance_type="split"):
        lgb.plot_importance(
            self.model,
            importance_type=importance_type,
            max_num_features=15,
            figsize=(10, 4),
        )
        plt.show()


class SarimaForecastingModel(AbstractForecastingModel):

    def initialize_model(self):
        """
        Ensure that the hyperparameters required for SARIMA are present.

        Required hyperparameters:
        - 'order': tuple, the (p, d, q) parameters for SARIMA.
        - 'seasonal_order': tuple, the (P, D, Q, s) parameters for seasonal SARIMA.
        """
        required_params = ["order", "seasonal_order"]
        for param in required_params:
            if param not in self.hyperparameters:
                raise ValueError(f"Hyperparameter '{param}' is required but missing.")

        self.order = self.hyperparameters["order"]
        self.seasonal_order = self.hyperparameters["seasonal_order"]
        return None

    def train(self, X_train, y_train):
        """
        Train the SARIMA model using the training data.

        Parameters:
        -----------
        X_train : pd.DataFrame
            Features (ignored by SARIMA, included for consistency).
        y_train : pd.Series
            The target time-series to train the model.
        """
        self.model = SARIMAX(
            y_train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

    def predict(self, X):
        """
        Predict future values using the SARIMA model.

        Parameters:
        -----------
        X : pd.DataFrame
            Dataframe with at least a time index and target column 'y'.
            Ignored for SARIMA as predictions are based only on the model state.

        Returns:
        --------
        np.array
            Predicted values for the specified time horizon.
        """
        if "steps_ahead" not in self.hyperparameters:
            raise ValueError("Hyperparameter 'steps_ahead' is required for prediction.")

        steps_ahead = self.hyperparameters["steps_ahead"]
        forecast = self.model.forecast(steps=steps_ahead)
        return forecast
    

class ExponentialSmoothingForecastingModel(AbstractForecastingModel):
    def initialize_model(self):
        """
        Ensure that the hyperparameters required for Exponential Smoothing are present.

        Required hyperparameters:
        - 'trend': str or None, type of trend component ('add', 'mul', or None).
        - 'seasonal': str or None, type of seasonal component ('add', 'mul', or None).
        - 'seasonal_periods': int, the number of periods in a season.
        """
        required_params = ["trend", "seasonal", "seasonal_periods"]
        for param in required_params:
            if param not in self.hyperparameters:
                raise ValueError(f"Hyperparameter '{param}' is required but missing.")

        self.trend = self.hyperparameters["trend"]
        self.seasonal = self.hyperparameters["seasonal"]
        self.seasonal_periods = self.hyperparameters["seasonal_periods"]
        return None

    def train(self, X_train, y_train):
        """
        Train the Exponential Smoothing model using the training data.

        Parameters:
        -----------
        X_train : pd.DataFrame
            Features (ignored by Exponential Smoothing, included for consistency).
        y_train : pd.Series
            The target time-series to train the model.
        """
        self.model = ExponentialSmoothing(
            y_train,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
        ).fit()

    def predict(self, X):
        """
        Predict future values using the Exponential Smoothing model.

        Parameters:
        -----------
        X : pd.DataFrame
            Dataframe with at least a time index and target column 'y'.
            Ignored for Exponential Smoothing as predictions are based only on the model state.

        Returns:
        --------
        np.array
            Predicted values for the specified time horizon.
        """
        if "steps_ahead" not in self.hyperparameters:
            raise ValueError("Hyperparameter 'steps_ahead' is required for prediction.")

        steps_ahead = self.hyperparameters["steps_ahead"]
        forecast = self.model.forecast(steps=steps_ahead)
        return forecast


class HierarchicalModel(AbstractForecastingModel):

    def initialize_model(self):
        return LGBMRegressor(**self.hyperparameters)

    @staticmethod
    def compute_proportions(data):
        # Compute proportions with historical data

        # Step 2: Aggregate total revenue by date
        df_revenue_agg = data.groupby('date')['total_revenue'].sum().reset_index()
        df_revenue_agg.rename(columns={'total_revenue': 'total_revenue_agg'}, inplace=True)

        # Step 3: Merge to calculate proportions
        df_proportions = data.merge(df_revenue_agg, on='date')

        # Step 4: Calculate the proportion for each brand/family
        df_proportions['proportion'] = (
            df_proportions['total_revenue'] / df_proportions['total_revenue_agg']
        )

        # Final DataFrame with proportions
        df_proportions = df_proportions[['date', 'brand', 'family', 'proportion']]

        # Combine all the monthly proportions into a single DataFrame
        df_proportions['month'] = df_proportions['date'].dt.month
        df_proportions = df_proportions.drop(columns='date')
        df_proportions = df_proportions.groupby(['brand', 'family', 'month'], as_index=False).agg({'proportion': 'mean'})
        return df_proportions
    
