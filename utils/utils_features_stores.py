import pandas as pd
import numpy as np
from scipy.stats import entropy


def add_num_stores(df, df_grouped):
    """
    Computes the number of unique stores selling a brand-family combination.
    This helps track how widely distributed sales are across different stores.
    """
    num_stores = df.groupby(['brand', 'family', 'date'])['store'].nunique().reset_index(name='num_stores')
    return df_grouped.merge(num_stores, on=['brand', 'family', 'date'], how='left')


def add_store_sales_concentration(df, df_grouped):
    """
    To understand whether sales are concentrated in a few stores or evenly 
    distributed, we can use Shannon Entropy:
	- If entropy is high, sales are evenly distributed across stores.
	- If entropy is low, sales are concentrated in a few stores.
    """
    # def store_entropy(group):
    #     store_counts = group['store'].value_counts(normalize=True)
    #     return entropy(store_counts) if len(store_counts) > 1 else 0
    def store_entropy(group):
        store_sales = group.groupby('store')['quantity'].sum()  # Sum sales per store
        store_probs = store_sales / store_sales.sum()  # Normalize sales
        return entropy(store_probs) if len(store_probs) > 1 else 0

    store_entropy_df = df.groupby(['brand', 'family', 'date']).apply(store_entropy).reset_index(name='store_entropy')
    return df_grouped.merge(store_entropy_df, on=['brand', 'family', 'date'], how='left')


def add_top_store_sales_ratio(df, df_grouped):
    """
    To capture dominance of the most important store:
    Computes the ratio of sales from the top-performing store.
    - If the ratio is close to 1, a single store dominates sales.
	- If the ratio is low, sales are more evenly distributed.
    """
    top_store_sales = df.groupby(['brand', 'family', 'date', 'store'])['quantity'].sum().reset_index()
    top_store_sales = top_store_sales.groupby(['brand', 'family', 'date'])['quantity'].max().reset_index(name='top_store_sales')

    df_grouped = df_grouped.merge(top_store_sales, on=['brand', 'family', 'date'], how='left')
    df_grouped['top_store_sales_ratio'] = df_grouped['top_store_sales'] / df_grouped['total_quantity']
    
    return df_grouped


def add_top_3_store_sales_ratio(df, df_grouped):
    """
    Instead of just the top store, we can analyze the combined share of the top 3 stores.
    """
    top_3_store_sales = (
        df.groupby(['brand', 'family', 'date', 'store'])['quantity']
        .sum()
        .groupby(['brand', 'family', 'date'])
        .apply(lambda x: x.nlargest(3).sum())
        .reset_index(name='top_3_store_sales')
    )

    df_grouped = df_grouped.merge(top_3_store_sales, on=['brand', 'family', 'date'], how='left')
    df_grouped['top_3_store_sales_ratio'] = df_grouped['top_3_store_sales'] / df_grouped['total_quantity']
    
    return df_grouped


def add_avg_sales_per_store(df_grouped):
    """
    To capture how much each store contributes on average.
    """
    df_grouped['avg_sales_per_store'] = df_grouped['total_quantity'] / df_grouped['num_stores']
    return df_grouped
