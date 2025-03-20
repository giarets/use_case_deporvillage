import pandas as pd
import numpy as np
from scipy.stats import entropy


class FeatureEngineeringPipeline:

    def __init__(self, df, frequency="D"):
        self.df = df.copy()  # Work on a copy to avoid modifying the original DataFrame
        self.frequency = frequency
        self.df_grouped = None
        self.df_store_features = None
        self.df_seasonality_features = None
        self.df_final = None

    def run(self):

        self.aggregate_data()
        self.compute_store_features()
        self.compute_seasonality_features()

        self.df_final = self.df_grouped.merge(
            self.df_store_features, on=["brand", "family", "date"], how="left"
        ).merge(
            self.df_seasonality_features, on=["brand", "family", "date"], how="left"
        )

        self.compute_seasonality_change()
        return self.df_final.sort_values(["date"])

    def aggregate_data(self):
        """
        Aggregates total quantity and revenue at the specified time frequency.
        Computes:
            - total_quantity: Total quantity sold.
            - total_revenue: Total revenue generated.
            - avg_pvp: weighted average PVP for each brand + family.
        """
        self.df["date"] = pd.to_datetime(self.df["date"])

        self.df_grouped = (
            self.df.groupby(
                ["brand", "family", pd.Grouper(key="date", freq=self.frequency)]
            )
            .agg(
                total_quantity=("quantity", "sum"),
                total_revenue=(
                    "quantity",
                    lambda x: np.sum(x * self.df.loc[x.index, "pvp"]),
                ),
            )
            .reset_index()
        )

        self.df_grouped["avg_pvp"] = (
            self.df_grouped["total_revenue"] / self.df_grouped["total_quantity"]
        )
        return self.df_grouped

    def compute_store_features(self):
        """
        num_stores: Computes the number of unique stores selling a brand-family combination.
            This helps track how widely distributed sales are across different stores.

        store_sales_concentration:
            To understand whether sales are concentrated in a few stores or evenly
            distributed, we can use Shannon Entropy:
                - If entropy is high, sales are evenly distributed across stores.
                - If entropy is low, sales are concentrated in a few stores.

        avg_sales_per_store:
            To capture how much each store contributes on average.

        top_store_sales:
            To capture dominance of the most important store:
            Computes the total sales from the top-performing store.

        top_3_store_sales_ratio:
            Instead of just the top store, we can analyze the combined share of the top 3 stores.

        top_store_sales_ratio:
            To capture dominance of the most important store:
            Computes the ratio of sales from the top-performing store.
                - If the ratio is close to 1, a single store dominates sales.
                - If the ratio is low, sales are more evenly distributed.
        """
        df_store_features = (
            self.df.groupby(["brand", "family", "date", "store"])
            .agg(store_sales=("quantity", "sum"))
            .reset_index()
        )

        df_store_features["date"] = pd.to_datetime(df_store_features["date"])

        self.df_store_features = (
            df_store_features.groupby(
                ["brand", "family", pd.Grouper(key="date", freq=self.frequency)]
            )
            .agg(
                num_stores=("store", "nunique"),
                store_sales_concentration=(
                    "store_sales",
                    lambda x: entropy(x / x.sum()) if len(x) > 1 else 0,
                ),
                avg_sales_per_store=("store_sales", "mean"),
                top_store_sales=("store_sales", "max"),
                top_3_store_sales=("store_sales", lambda x: x.nlargest(3).sum()),
                top_store_sales_ratio=(
                    "store_sales",
                    lambda x: x.max() / x.sum() if x.sum() > 0 else 0,
                ),
            )
            .reset_index()
        )

        self.df_store_features["avg_sales_per_store"] = self.df_store_features[
            "avg_sales_per_store"
        ].round(2)
        self.df_store_features["store_sales_concentration"] = self.df_store_features[
            "store_sales_concentration"
        ].round(2)
        return self.df_store_features

    def compute_seasonality_features(self):
        """
        seasonality_sales_concentration:
            Similar to store entropy, we can measure how spread out seasonality types are.
            Higher entropy → sales spread across multiple seasonality types.

        mode_seasonality:
            To track the dominant season for a given brand-family-date.

        seasonality_change:
            Computes how frequently seasonality changes over time.
                - A high value means frequent changes in dominant seasonality.
                - A low value means consistency in seasonality trends.

        seasonality_ratios:
            Computes the proportion of sales attributed to each seasonality type.
        """

        df_seasonality_features = (
            self.df.groupby(["brand", "family", "date", "seasonality"])
            .agg(season_sales=("quantity", "sum"))
            .reset_index()
        )

        df_seasonality_features["date"] = pd.to_datetime(
            df_seasonality_features["date"]
        )

        # Compute seasonality ratios
        df_pivot = df_seasonality_features.pivot_table(
            index=["brand", "family", "date"],
            columns="seasonality",
            values="season_sales",
            aggfunc="sum",
            fill_value=0,
        )

        df_pivot = df_pivot.div(df_pivot.sum(axis=1), axis=0).fillna(0)
        df_pivot.columns = [f"seasonality_ratio_{col}" for col in df_pivot.columns]

        self.df_seasonality_features = (
            df_seasonality_features.groupby(
                ["brand", "family", pd.Grouper(key="date", freq=self.frequency)]
            )
            .agg(
                seasonality_sales_concentration=(
                    "season_sales",
                    lambda x: entropy(x / x.sum()) if x.sum() > 0 and len(x) > 1 else 0,
                )
            )
            .reset_index()
        )

        mode_seasonality_df = (
            df_seasonality_features.groupby(["brand", "family", "date"])
            .apply(self._most_frequent_seasonality)
            .reset_index(name="mode_seasonality")
        )

        self.df_seasonality_features = self.df_seasonality_features.merge(
            mode_seasonality_df, on=["brand", "family", "date"], how="left"
        ).merge(df_pivot.reset_index(), on=["brand", "family", "date"], how="left")

        return self.df_seasonality_features

    def compute_seasonality_change(self):
        """Computes how frequently seasonality changes over time."""

        if self.df_final is not None:
            self.df_final["seasonality_change"] = (
                self.df_final.groupby(["brand", "family"])["mode_seasonality"]
                .apply(lambda x: x.ne(x.shift()).astype(int))
                .reset_index(drop=True)
            )
        return self.df_final

    @staticmethod
    def _most_frequent_seasonality(group):
        """Returns the seasonality with the highest total sales."""

        counts = group.groupby("seasonality")["season_sales"].sum()
        return counts.idxmax() if len(counts) > 0 else "N-A"
