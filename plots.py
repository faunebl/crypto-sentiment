import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from datetime import datetime, date


def _ensure_columns(df: pl.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Colonnes manquantes dans le DataFrame: {missing}")


def _parse_date(val) -> date:
    """Convertit une valeur en datetime.date"""
    if isinstance(val, date):
        return val
    if isinstance(val, str):
        try:
            return datetime.strptime(val, "%Y-%m-%d").date()
        except ValueError:
            raise ValueError(f"Impossible de parser la date: {val}")
    # pour d'autres types (e.g. numpy.datetime64)
    try:
        return val.astype('M8[D]').astype(date)
    except Exception:
        raise TypeError(f"Type de date non reconnu: {type(val)}")


def plot_score_time_series(df: pl.DataFrame) -> None:
    """
    Affiche l'évolution du score et la direction réelle (actual) au cours du temps.
    Colonnes requises: 'date', 'score', 'actual'.
    """
    _ensure_columns(df, ['date', 'score', 'actual'])
    raw_dates = df['date'].to_list()
    dates = [_parse_date(d) for d in raw_dates]
    scores = df['score'].to_numpy()
    actuals = df['actual'].to_numpy()

    fig, ax = plt.subplots()
    ax.plot(dates, scores, label='Score')
    ax.scatter(dates, actuals, s=20, label='Direction réelle (actual)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Score / Actual')
    ax.set_title("Score et direction réelle dans le temps")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix_heatmap(df: pl.DataFrame) -> None:
    """
    Affiche la matrice de confusion entre actual et predicted.
    Colonnes requises: 'actual', 'predicted'.
    Les lignes contenant NaN dans 'actual' ou 'predicted' sont exclues.
    """
    _ensure_columns(df, ['actual', 'predicted'])
    # Exclusion des valeurs manquantes
    df_clean = df.filter(
        pl.col('actual').is_not_null() & pl.col('predicted').is_not_null()
    )
    if df_clean.height < df.height:
        print(f"Attention: {df.height - df_clean.height} lignes avec NaN ont été supprimées.")
    y_true = df_clean['actual'].to_numpy()
    y_pred = df_clean['predicted'].to_numpy()

    cm = confusion_matrix(y_true, y_pred)
    classes = np.unique(np.concatenate((y_true, y_pred)))

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_xlabel('Prédit')
    ax.set_ylabel('Réel')
    ax.set_title('Matrice de confusion')
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()


def plot_score_distribution(df: pl.DataFrame) -> None:
    """
    Affiche la distribution des scores.
    Colonne requise: 'score'.
    """
    _ensure_columns(df, ['score'])
    scores = df['score'].to_numpy()

    fig, ax = plt.subplots()
    ax.hist(scores, bins='auto')
    ax.set_xlabel('Score')
    ax.set_ylabel('Fréquence')
    ax.set_title('Distribution des scores')
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pl.DataFrame) -> None:
    """
    Affiche la heatmap des corrélations entre toutes les colonnes numériques.
    Colonnes numériques: 'increase','decrease','diff','score','actual','predicted'.
    """
    num_cols = ['increase', 'decrease', 'diff', 'score', 'actual', 'predicted']
    _ensure_columns(df, num_cols)
    # Conversion et suppression NaN pour calcul de corrélation
    arrs = []
    for c in num_cols:
        series = df[c].to_numpy()
        if np.isnan(series).any():
            series = series[~np.isnan(series)]
        arrs.append(series)
    # Alignement des longueurs (min shared length)
    min_len = min(len(a) for a in arrs)
    data = np.vstack([a[:min_len] for a in arrs]).T
    corr = np.corrcoef(data, rowvar=False)

    fig, ax = plt.subplots()
    im = ax.imshow(corr, vmin=-1, vmax=1)
    ax.set_xticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=45)
    ax.set_yticks(range(len(num_cols)))
    ax.set_yticklabels(num_cols)
    ax.set_title('Corrélations')
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha='center', va='center')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()


def plot_score_vs_actual_scatter(df: pl.DataFrame) -> None:
    """
    Nuage de points entre score et actual.
    Colonnes requises: 'score', 'actual'.
    Les paires avec NaN sont exclues.
    """
    _ensure_columns(df, ['score', 'actual'])
    arr = np.vstack([
        df['score'].to_numpy(),
        df['actual'].to_numpy()
    ]).T
    # Filtrer les NaN
    arr = arr[~np.isnan(arr).any(axis=1)]
    scores, actuals = arr[:,0], arr[:,1]

    fig, ax = plt.subplots()
    ax.scatter(scores, actuals)
    ax.set_xlabel('Score')
    ax.set_ylabel('Actual')
    ax.set_title('Score vs Actual')
    plt.tight_layout()
    plt.show()
