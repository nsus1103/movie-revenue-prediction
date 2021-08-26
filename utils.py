### auxiliary functions ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_quantiles(feature, target, var_names, bins=10):

    df = pd.DataFrame({'feature': feature, 'target': target})
    df['quantile'] = np.floor(bins * df['feature'].rank(method='first', ascending=True) / df.shape[0])

    if (df.feature.isna().sum() > 0):
        df = pd.concat([
            df[~df.feature.isna()].groupby('quantile').agg({'target': 'mean', 'feature': 'mean'}).sort_values(
                'quantile').reset_index(drop=False),
            pd.DataFrame(
                {'quantile': np.NAN, 'feature': np.NAN, 'target': df[df.feature.isna()].agg({'target': 'mean'})})
        ])
    else:
        df = df.groupby('quantile').agg({'target': 'mean', 'feature': 'mean'}).sort_values('quantile').reset_index(
            drop=False)

    df = df[["quantile", "feature", "target"]]
    df.rename(columns={'feature': var_names[0], 'target': var_names[1]}, inplace=True)

    return df

def plot_quantiles(x, y):
    fig = plt.figure(figsize=(5, 5))
    plt.scatter(x,y,marker='x', c='r')
    plt.xlabel(f"{x.name}")
    plt.ylabel(f"{y.name}")
    plt.show()
    return

def get_groups(df, feature):
    return df.groupby(feature).agg(count=('total_revenue','count'), mean_revenue=('total_revenue', 'mean'), mean_budget=('budget','mean')).sort_values('count', ascending=False).reset_index(drop=False)

def encode_category(df, cols):
    for col in cols:
        df[col].fillna('NA', inplace=True)
        categories = set()

        for cats in df[col]:
            list_cats = cats.replace(' ', '').split(',')
            for i in range(len(list_cats)):
                categories.add(list_cats[i])

        # Create columns for countries
        for cat in categories:
            df[f'{col}_{cat}'] = 0

        for i, row in df.iterrows():
            list_cats = row[col].replace(' ', '').split(',')
            for cat in list_cats:
                df.at[i, f'{col}_{cat}'] = 1

        df.fillna(0, inplace=True)

    # df.drop(cols, axis=1, inplace=True)

    return df

def create_others(df, feature, threshold):
    df[f"n_{feature}"] = df.groupby(feature)["total_revenue"].transform("count")
    df[f"{feature}2"] = df.apply(lambda row: row[feature] if row[f"n_{feature}"] >= threshold else "Other", axis=1)
    return df

def rmse(target, pred):
    # Given a target and a prediction returns the Root Mean Squared Error
    return ((target-pred)**2).mean()**0.5


def tbr(target, pred, pct=0.2):
    # Returns the ratio between the top #pct and the bottom #pct
    df = pd.DataFrame({'target': target, 'pred': pred})

    n = df.shape[0]
    top = df.sort_values("pred", ascending=False)[0:int(np.floor(pct*n))].target.sum()
    bot = df.sort_values("pred", ascending=True)[0:int(np.floor(pct*n))].target.sum()
    return top/bot
