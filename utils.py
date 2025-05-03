import polars as pl 
import statsmodels.api as sm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from datetime import timedelta
import yfinance as yf
import pandas as pd

def compute_sentiment_ear(sentiment: pl.Series) -> pl.Series:
    """
    Computes Ex-Ante Residual (EAR) from an AR(1) model on sentiment.

    Args:
        sentiment (pl.Series): Sentiment time series.

    Returns:
        pl.Series: EAR time series (NaN at first index to maintain alignment).
    """
    sentiment_np = sentiment.to_numpy()
    y = sentiment_np[1:]
    X = sm.add_constant(sentiment_np[:-1])
    
    model = sm.OLS(y, X).fit()
    residuals = y - model.predict(X)
    residuals = np.insert(residuals, 0, np.nan)  # align with original

    return pl.Series("sentiment_ear", residuals)

def local_projection_irf_polars(y: pl.Series, x: pl.Series, control: pl.Series, max_horizon=30) -> dict:
    """
    Estimate IRFs using local projections with Polars.

    Args:
        y (pl.Series): Target variable (e.g., sentiment or return).
        x (pl.Series): Shock variable (same length as y).
        control (pl.Series): Control variable (same length as y).
        max_horizon (int): Max forecast horizon (e.g., 30 days).

    Returns:
        dict: IRF coefficients by horizon.
    """
    irf_results = {}
    df = pl.DataFrame({
        "y": y,
        "x": x,
        "control": control
    })

    for h in range(1, max_horizon + 1):
        # compute rolling lagged  sum for horizon h
        y_future = df["y"].shift(-1).rolling_sum(h).alias("y_future")

        # build temp frame with shifted cumulative y
        temp_df = df.with_columns([
            y_future
        ]).drop_nulls()

        # converting to np
        y_np = temp_df["y_future"].to_numpy()
        X_np = temp_df.select(["x", "control"]).to_numpy()
        X_np = sm.add_constant(X_np)

        model = sm.OLS(y_np, X_np).fit()
        irf_results[h] = model.params[1]  # coefficient on 'x' (shock)

    return irf_results

def get_logistic_regression(df: pl.DataFrame, penalty: str, X: list, y: list):
    model = LogisticRegression(penalty=penalty, multi_class='multinomial')

    ind = -int(len(X)/3)

    X_test = X[ind:]
    X_train = X[:ind]

    y_test = y[ind:]
    y_train = y[:ind]

    reg = model.fit(X_train, y_train)

    out_of_sample = reg.predict(X_test)
    in_sample = reg.predict(X_train)

    print('Accuracy:', accuracy_score(y_train, in_sample))
    print('Precision:', precision_score(y_train, in_sample, average='weighted'))
    print('Recall:', recall_score(y_train, in_sample, average='weighted'))
    print('F1 score:', f1_score(y_train, in_sample, average='weighted'))
    print('Confusion matrix:\n', confusion_matrix(y_train, in_sample))

    print('Accuracy (Out of Sample):', accuracy_score(y_test, out_of_sample))
    print('Precision (Out of Sample):', precision_score(y_test, out_of_sample, average='weighted'))
    print('Recall (Out of Sample):', recall_score(y_test, out_of_sample, average='weighted'))
    print('F1 score (Out of Sample):', f1_score(y_test, out_of_sample, average='weighted'))
    print('Confusion matrix (Out of Sample):\n', confusion_matrix(y_test, out_of_sample))

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    cm = confusion_matrix(y_train, in_sample)
    axs[0].matshow(cm, cmap=plt.cm.Blues)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axs[0].text(j, i, cm[i, j], ha='center', va='center')
    axs[0].set_xlabel('Predicted Class')
    axs[0].set_ylabel('True Class')
    axs[0].set_title('In Sample')

    cm = confusion_matrix(y_test, out_of_sample)
    axs[1].matshow(cm, cmap=plt.cm.Blues)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axs[1].text(j, i, cm[i, j], ha='center', va='center')
    axs[1].set_xlabel('Predicted Class')
    axs[1].set_ylabel('True Class')
    axs[1].set_title('Out of Sample')

    plt.tight_layout()
    plt.show()
    
    return

def get_btc_returns(scores: pl.DataFrame):
    min_date = scores["date"].min()
    max_date = scores["date"].max()

    start_str = min_date.strftime("%Y-%m-%d")
    end_str   = (max_date + timedelta(days=1)).strftime("%Y-%m-%d")

    df_pd = yf.download(
        tickers="BTC-USD",
        start=start_str,
        end=end_str,
        interval="1d",
        progress=False
    )


    if isinstance(df_pd.columns, pd.MultiIndex):
        df_pd.columns = df_pd.columns.get_level_values(0)

    df_pd = df_pd.reset_index()
    df_price = (
        pl.from_pandas(df_pd)
        .select([
            pl.col("Date").alias("date").cast(pl.Date),
            pl.col("Close").alias("close")
        ])
        .with_columns(pl.col('close').log().diff().alias('returns'))
        .with_columns(pl.when(pl.col('returns').gt(0)).then(1).otherwise(0).alias('actual'))
    )
    return df_price
