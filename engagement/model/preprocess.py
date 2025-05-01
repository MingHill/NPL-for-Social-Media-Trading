#!/usr/bin/env python
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import _load_yaml, _read_data, CONFIG_PATH

def _log1p_base(x: pd.Series, base: float = 10.0) -> pd.Series:
    """Apply log(1+x)/log(base) element‑wise; returns zero where x<=0."""
    log = np.log1p(np.maximum(x, 0)) / np.log(base)
    return log.fillna(0.0)

def sigmoid(x: pd.Series) -> pd.Series:
    return 1 / (1 + np.exp(-x))

def _simple_shared(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    processed_df = df.copy()
    log_base = cfg["global"]["log_base"]

    # keep the tweet_id with the highest delta out of all the same tweet_id
    processed_df = processed_df.sort_values(by='delta_seconds', ascending=False).groupby('tweet_id').first().reset_index()
    
    # sort by tweet_created_at
    processed_df = processed_df.sort_values(by='tweet_created_at', ascending=True)

    # log the difference between tweet_created_at and user_created_at
    # The time difference between tweet creation and user creation
    # A small difference might indicate a throwaway account
    diff_sec = (processed_df["tweet_created_at"] - processed_df["user_created_at"])
    processed_df['tweet_created_at_user_created_at_diff_log'] = _log1p_base(diff_sec, log_base)
    #processed_df['tweet_age_log'] = _log1p_base(processed_df['delta_seconds'], log_base)
    processed_df['tweet_age_sigmoid'] = sigmoid(processed_df['delta_seconds'])

    # convert to datetime
    processed_df["tweet_created_at"] = pd.to_datetime(processed_df["tweet_created_at"], unit="s", errors="coerce")
    processed_df["user_created_at"]  = pd.to_datetime(processed_df["user_created_at"], unit="s", errors="coerce")

    # Tweet created at hod, dow, sin, cos
    processed_df["tweet_created_at_hour"] = processed_df['tweet_created_at'].dt.hour
    processed_df["tweet_created_at_day_of_week"] = processed_df['tweet_created_at'].dt.dayofweek
    processed_df["tweet_created_at_hour_sin"] = np.sin(processed_df["tweet_created_at_hour"] * (2 * np.pi / 24))
    processed_df["tweet_created_at_hour_cos"] = np.cos(processed_df["tweet_created_at_hour"] * (2 * np.pi / 24))
    processed_df["tweet_created_at_day_of_week_sin"] = np.sin(processed_df["tweet_created_at_day_of_week"] * (2 * np.pi / 7))
    processed_df["tweet_created_at_day_of_week_cos"] = np.cos(processed_df["tweet_created_at_day_of_week"] * (2 * np.pi / 7))

    # User created at hod, dow
    processed_df["user_created_at_hour"] = processed_df['user_created_at'].dt.hour
    processed_df["user_created_at_day_of_week"] = processed_df['user_created_at'].dt.dayofweek
    processed_df["user_created_at_hour_sin"] = np.sin(processed_df["user_created_at_hour"] * (2 * np.pi / 24))
    processed_df["user_created_at_hour_cos"] = np.cos(processed_df["user_created_at_hour"] * (2 * np.pi / 24))
    processed_df["user_created_at_day_of_week_sin"] = np.sin(processed_df["user_created_at_day_of_week"] * (2 * np.pi / 7))
    processed_df["user_created_at_day_of_week_cos"] = np.cos(processed_df["user_created_at_day_of_week"] * (2 * np.pi / 7))

    # More temporal features
    processed_df["is_weekend"]   = processed_df["tweet_created_at_day_of_week"].isin([5, 6]).astype(int)
    processed_df["is_night_hour"] = processed_df["tweet_created_at_hour"].between(0, 6).astype(int)


    # log followers count
    processed_df['user_followers_count_log'] = _log1p_base(processed_df['user_followers_count'], log_base)

    # log following count
    processed_df['user_following_count_log'] = _log1p_base(processed_df['user_following_count'], log_base)

    # follower to following ratio
    processed_df['user_followers_to_following_ratio_log'] = processed_df['user_followers_count_log'] - processed_df['user_following_count_log']

    def create_input_text(row):
        username = row["username"] if row["username"] is not None else ""
        user_description = row["user_description"] if row["user_description"] is not None else ""
        tweet_text = row["tweet_text"] if row["tweet_text"] is not None else ""
        return f"Username: {username}\nUser Description: {user_description}\nTweet: {tweet_text}"
    
    processed_df["input_text"] = processed_df.apply(create_input_text, axis=1)

    processed_df["tweet_char_len"] = processed_df["tweet_text"].fillna("").str.len()
    processed_df["tweet_word_len"] = processed_df["tweet_text"].fillna("").str.split().str.len()
    processed_df["description_word_len"] = processed_df["user_description"].fillna("").str.split().str.len()
    processed_df["username_len"] = processed_df["username"].fillna("").str.len()

    tweet_text_lower = processed_df["tweet_text"].fillna("").str.lower()
    processed_df["has_hashtag"] = tweet_text_lower.str.contains("#").astype(int)
    processed_df["has_url"] = tweet_text_lower.str.contains("http://|https://").astype(int)
    processed_df["has_mention"] = tweet_text_lower.str.contains("@").astype(int)

    return processed_df

def _simple_v1(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    target = cfg["preprocess"]["simple_v1"]["target"]
    target_log = cfg["preprocess"]["simple_v1"]["target_log"]
    log_base = cfg["global"]["log_base"]
    processed_df = _simple_shared(df, cfg)
    
    # target
    if target_log:
        processed_df['target'] = _log1p_base(processed_df[target], log_base)
    else:
        processed_df['target'] = processed_df[target]

    # drop unnecessary columns
    drop_columns = ["tweet_id", "observation_id", "observed_at", "delta_seconds", "user_last_updated", 'user_id', 'tweet_last_updated', 'monitor_flag', 'tweet_created_at', 'user_created_at', 'user_followers_count', 'user_following_count', 'favorite_count', 'retweet_count', 'reply_count', 'quote_count', 'bookmark_count', 'view_count', 'tweet_text', 'username', 'user_description']

    processed_df = processed_df.drop(columns=drop_columns)

    return processed_df

def _simple_v2(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    log_base = cfg["global"]["log_base"]
    processed_df = _simple_shared(df, cfg)

    # Combined features (sum of favorite_count, retweet_count, reply_count, quote_count, bookmark_count, view_count)
    processed_df['target'] = processed_df[['favorite_count', 'retweet_count', 'reply_count', 'quote_count', 'bookmark_count', 'view_count']].sum(axis=1)
    processed_df['target'] = _log1p_base(processed_df['target'], log_base)

    # drop unnecessary columns
    drop_columns = ["tweet_id", "observation_id", "observed_at", "delta_seconds", "user_last_updated", 'user_id', 'tweet_last_updated', 'monitor_flag', 'tweet_created_at', 'user_created_at', 'user_followers_count', 'user_following_count', 'favorite_count', 'retweet_count', 'reply_count', 'quote_count', 'bookmark_count', 'view_count', 'tweet_text', 'username', 'user_description']

    processed_df = processed_df.drop(columns=drop_columns)

    return processed_df
    
    

def _run(cfg: dict) -> None:
    preprocess_cfg = cfg["preprocess"]
    strategy = preprocess_cfg["strategy"]
    input_csv: str = preprocess_cfg.get("input_csv")
    output_dir: Path = Path(preprocess_cfg.get("output_dir", "data/processed"))
    val_size: float = preprocess_cfg.get("val_size", 0.15)
    test_size: float = preprocess_cfg.get("test_size", 0.15)
    seed: int = cfg["global"].get("seed", 42)

    if not input_csv:
        sys.exit("'input_csv' is required in the preprocess section of the config")
    if val_size + test_size >= 1:
        sys.exit("val_size + test_size must be < 1.0")

    # Read -------------------------------------------------------------------
    df = _read_data(input_csv)
    if strategy == "simple_v1":
        df = _simple_v1(df, cfg)
    elif strategy == "simple_v2":
        df = _simple_v2(df, cfg)

    # Split ------------------------------------------------------------------
    train_df, temp_df = train_test_split(
        df, test_size=val_size + test_size, random_state=seed, shuffle=True
    )
    relative_val = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=1 - relative_val, random_state=seed, shuffle=True
    )

    # Persist ----------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "validation.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print(
        f"Saved splits to '{output_dir}': "
        f"{len(train_df):,} train / {len(val_df):,} val / {len(test_df):,} test rows."
    )

if __name__ == "__main__":
    preprocess_cfg = _load_yaml(CONFIG_PATH)
    _run(preprocess_cfg)
