import pandas as pd
import numpy as np
from scipy.stats import entropy


def add_mode_seasonality(df, df_grouped):
    """
    To track the dominant season for a given brand-family-date.
    """
    mode_seasonality = df.groupby(['brand', 'family', 'date'])['seasonality'].agg(lambda x: x.mode()[0]).reset_index(name='mode_seasonality')
    return df_grouped.merge(mode_seasonality, on=['brand', 'family', 'date'], how='left')


def add_seasonality_sales_concentration(df, df_grouped):
    """
    Similar to store entropy, we can measure how spread out seasonality types are.
    Higher entropy → sales spread across multiple seasonality types.
    """
    def seasonality_entropy(group):
        season_sales = group.groupby('seasonality')['quantity'].sum()  # Sum sales per seasonality
        season_probs = season_sales / season_sales.sum()  # Normalize sales
        return entropy(season_probs) if len(season_probs) > 1 else 0

    seasonality_entropy_df = df.groupby(['brand', 'family', 'date']).apply(seasonality_entropy).reset_index(name='seasonality_entropy')
    return df_grouped.merge(seasonality_entropy_df, on=['brand', 'family', 'date'], how='left')


def add_seasonality_change(df_grouped):
    """
    Computes how frequently the dominant seasonality changes over time.
    - A high value means frequent changes in dominant seasonality.
	- A low value means consistency in seasonality trends.
    """
    df_grouped['seasonality_change'] = (
        df_grouped.groupby(['brand', 'family'])['mode_seasonality']
        .apply(lambda x: x.ne(x.shift()).astype(int))
        .reset_index(drop=True)
    )
    return df_grouped


def add_seasonality_ratios(df, df_grouped):
    """
    Computes the proportion of sales attributed to each seasonality type.
    """
    seasonality_dummies = pd.get_dummies(df['seasonality'])

    for col in seasonality_dummies.columns:
        df_grouped[f'seasonality_{col}_ratio'] = (
            df.groupby(['brand', 'family', 'date'])['quantity']
            .apply(lambda x: np.sum(x[df.loc[x.index, 'seasonality'] == col]) / np.sum(x))
            .reset_index(drop=True)
        )

    return df_grouped


