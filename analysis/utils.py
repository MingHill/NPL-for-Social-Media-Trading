import sys
from pathlib import Path

import pandas as pd
import yaml
import requests
import time
import matplotlib.pyplot as plt

CONFIG_PATH = "config.yaml"

def plot_target_distribution(df, column: str = "target") -> None:
    """
    Visualize the distribution of a numeric column with helpful context.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the column to plot.
    column : str
        Name of the numeric column whose distribution you want to inspect.
    """
    # Drop missing values and cache the series for convenience
    series = df[column].dropna()

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8, 5))

    # Histogram with an adaptive bin count
    ax.hist(series, bins="auto")             # no explicit color → uses default
    ax.set_title(f"Distribution of `{column}`")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    # Add mean and median reference lines
    mean_val = series.mean()
    median_val = series.median()
    ax.axvline(mean_val,   linestyle="--", linewidth=1, label=f"Mean = {mean_val:.2f}")
    ax.axvline(median_val, linestyle=":",  linewidth=1, label=f"Median = {median_val:.2f}")

    ax.legend()
    plt.tight_layout()
    plt.show()


def get_coin_data(coin_address: str):
    url = f"https://api.rugcheck.xyz/v1/tokens/{coin_address}/report"
    resp = requests.get(url)
    resp.raise_for_status()
    time.sleep(2)
    data_dict = resp.json()
    return data_dict

def _load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        sys.exit(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "preprocess" not in cfg:
        sys.exit("Config must contain a top‑level 'preprocess' section")
    return cfg

def _read_data(path_pattern: str) -> pd.DataFrame:
    paths = list(Path().glob(path_pattern)) if any(c in path_pattern for c in "*?[") else [Path(path_pattern)]
    if not paths:
        sys.exit(f"No files match pattern: {path_pattern}")
    frames = [pd.read_csv(p) for p in paths]
    # frames = []
    # for p in paths:
    #     try:
    #         df = pd.read_csv(p)
    #     except pd.errors.ParserError as e:
    #         print(
    #             f"Warning: ParserError reading '{p}', retrying with python engine and skipping bad lines: {e}",
    #             file=sys.stderr,
    #         )
    #         df = pd.read_csv(p, engine='python', on_bad_lines='skip')
    #     frames.append(df)
    return pd.concat(frames, ignore_index=True)