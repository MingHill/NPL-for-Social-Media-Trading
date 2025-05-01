import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).parent / "main.db"

def get_connection(db_path=DEFAULT_DB_PATH):
    """
    Returns a sqlite3 connection to the given database path.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path=DEFAULT_DB_PATH):
    """
    Initializes the database by creating the coin_table if it doesn't exist.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS coin_table (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_signature TEXT UNIQUE,
            exchange TEXT,
            creation_timestamp TEXT,
            coin_address TEXT,
            pair_address TEXT,
            coin_name TEXT,
            coin_symbol TEXT,
            creator TEXT,
            initial_liquidity REAL,
            initial_price REAL,
            ath_time_delta REAL,
            ath_price_gain REAL,
            rugpull_flag INTEGER DEFAULT 0,
            rugpull_timestamp TEXT,
            rugpull_time_delta REAL
        )
        """
    )
    conn.commit()
    conn.close()

def create_tweets_table(db_path=DEFAULT_DB_PATH):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS tweets_table (
            observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            coin_address TEXT NOT NULL,
            pair_address TEXT NOT NULL,
            tweet_id TEXT NOT NULL,
            tweet_created_at INTEGER NOT NULL,
            tweet_text TEXT,
            tweet_sentiment_label TEXT,
            tweet_sentiment_label_prob REAL,
            tweet_engagement_score REAL,
            user_id TEXT NOT NULL,
            username TEXT,
            user_description TEXT,
            user_followers_count INTEGER,
            user_following_count INTEGER,
            verified INTEGER,
            is_blue_verified INTEGER,
            user_created_at INTEGER,
            user_last_updated INTEGER,
            FOREIGN KEY (coin_address) REFERENCES coin_table(coin_address),
            FOREIGN KEY (pair_address) REFERENCES coin_table(pair_address)
        )
        """
    )
    conn.commit()
    conn.close()

def create_completed_table(db_path=DEFAULT_DB_PATH):
    # completed coins
    # coin_address - from coin_table - string
    # pair_address - from coin_table - string

    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS completed_table (
            coin_address TEXT PRIMARY KEY,
            pair_address TEXT,
            FOREIGN KEY (coin_address) REFERENCES coin_table(coin_address),
            FOREIGN KEY (pair_address) REFERENCES coin_table(pair_address)
        )
        """
    )
    conn.commit()
    conn.close()

def create_coin_socials_table(db_path=DEFAULT_DB_PATH):
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS coin_socials_table (
            coin_address TEXT PRIMARY KEY,
            twitter TEXT,
            telegram TEXT,
            website TEXT,
            createdOn TEXT,
            createdOn_site TEXT,
            FOREIGN KEY (coin_address) REFERENCES coin_table(coin_address)
        )
        """
    )
    conn.commit()
    conn.close()

def insert_coin_event(event_signature, exchange, creation_timestamp, coin_address, pair_address, coin_name, coin_symbol, creator, initial_liquidity, initial_price, db_path=DEFAULT_DB_PATH):
    """
    Inserts a new coin event into the coin_table.
    Returns the new row id or None on error.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT OR IGNORE INTO coin_table
                (event_signature, exchange, creation_timestamp, coin_address, pair_address, coin_name, coin_symbol, creator, initial_liquidity, initial_price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (event_signature, exchange, creation_timestamp, coin_address, pair_address, coin_name, coin_symbol, creator, initial_liquidity, initial_price),
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.Error as e:
        print(f"DB Insert Error: {e}")
        return None
    finally:
        conn.close()

def insert_tweet(tweet_dict, db_path=DEFAULT_DB_PATH):
    """
    Inserts a new tweet into the tweets_table.
    Returns the new row id or None on error.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO tweets_table (coin_address, pair_address, tweet_id, tweet_created_at, tweet_text, tweet_sentiment_label, tweet_sentiment_label_prob, tweet_engagement_score, user_id, username, user_description, user_followers_count, user_following_count, verified, is_blue_verified, user_created_at, user_last_updated)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (tweet_dict["coin_address"], tweet_dict["pair_address"], tweet_dict["tweet_id"], tweet_dict["tweet_created_at"], tweet_dict["tweet_text"], tweet_dict["tweet_sentiment_label"], tweet_dict["tweet_sentiment_label_prob"], tweet_dict["tweet_engagement_score"], tweet_dict["user_id"], tweet_dict["username"], tweet_dict["user_description"], tweet_dict["user_followers_count"], tweet_dict["user_following_count"], tweet_dict["verified"], tweet_dict["is_blue_verified"], tweet_dict["user_created_at"], tweet_dict["user_last_updated"])
    )
    conn.commit()
    conn.close()

def update_ath(event_signature, ath_time_delta, ath_price_gain, db_path=DEFAULT_DB_PATH):
    """
    Updates ATH metrics for a given event_signature.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE coin_table
        SET ath_time_delta = ?, ath_price_gain = ?
        WHERE event_signature = ?
        """,
        (ath_time_delta, ath_price_gain, event_signature),
    )
    conn.commit()
    conn.close()

def update_rugpull(event_signature, rugpull_flag, rugpull_timestamp=None, rugpull_time_delta=None, db_path=DEFAULT_DB_PATH):
    """
    Updates rugpull info for a given event_signature.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE coin_table
        SET rugpull_flag = ?, rugpull_timestamp = ?, rugpull_time_delta = ?
        WHERE event_signature = ?
        """,
        (1 if rugpull_flag else 0, rugpull_timestamp, rugpull_time_delta, event_signature),
    )
    conn.commit()
    conn.close()