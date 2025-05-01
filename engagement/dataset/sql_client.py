"""
Direct SQL client implementation for storing tweet data.
This eliminates socket-based communication in favor of direct database access.
"""

import sqlite3
import logging
import time
import json
import os
from pathlib import Path
import threading

# Create logs directory if it doesn't exist
logs_dir = Path('logs')
logs_dir.mkdir(exist_ok=True)

# Define log file path
log_file = logs_dir / 'sql_client.log'

# Set up logging with file handler for persistent logs
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG level for detailed information

# Clean up old log file if it exists
if log_file.exists():
    try:
        log_file.unlink()  # Delete the existing log file
        print(f"Deleted old log file: {log_file}")
    except Exception as e:
        print(f"Warning: Could not delete old log file {log_file}: {e}")

# Create a file handler to save logs to a file
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Keep console handler for real-time monitoring
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)


class DirectSQLClient:
    """A direct SQL client that interacts with an SQLite database for storing tweet data."""
    
    def __init__(self, db_path="tweets.db"):
        """Initialize the SQL client with the path to the database file."""
        self.db_path = db_path
        self.lock = threading.Lock()  # Lock for thread safety
        logger.info(f"Initializing DirectSQLClient with database: {self.db_path}")
        self._ensure_tables()
        self._migrate_database_if_needed()
    
    def _ensure_tables(self):
        """Ensure that all required tables and indices exist in the database."""
        with self.lock:
            logger.debug(f"Verifying database tables in {self.db_path}")
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Create users table with new fields
                    cursor.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                        id TEXT PRIMARY KEY,
                        username TEXT NOT NULL,
                        user_created_at INTEGER,
                        description TEXT,
                        verified INTEGER,
                        is_blue_verified INTEGER,
                        last_updated INTEGER DEFAULT (strftime('%s', 'now'))
                    )
                    ''')
                    
                    # Create tweets table
                    cursor.execute('''
                    CREATE TABLE IF NOT EXISTS tweets (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        created_at INTEGER NOT NULL,
                        text TEXT NOT NULL,
                        monitor_flag INTEGER DEFAULT 1,
                        last_updated INTEGER DEFAULT (strftime('%s', 'now')),
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                    ''')
                    
                    # Create tweet_observations table with user_id field
                    cursor.execute('''
                    CREATE TABLE IF NOT EXISTS tweet_observations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        tweet_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        observed_at INTEGER NOT NULL,
                        delta_seconds INTEGER NOT NULL,
                        user_followers_count INTEGER,
                        user_following_count INTEGER,
                        favorite_count INTEGER,
                        retweet_count INTEGER,
                        reply_count INTEGER,
                        quote_count INTEGER,
                        bookmark_count INTEGER,
                        view_count INTEGER,
                        FOREIGN KEY (tweet_id) REFERENCES tweets (id),
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                    ''')
                    
                    # Create indices for better performance
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_tweet_user_id ON tweets (user_id)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_tweet_last_updated ON tweets (last_updated)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_observation_tweet_id ON tweet_observations (tweet_id)')
                    cursor.execute('CREATE INDEX IF NOT EXISTS idx_observation_user_id ON tweet_observations (user_id)')
                    
                    conn.commit()
                    logger.info("Database tables verified successfully")
            except sqlite3.Error as e:
                logger.error(f"SQLite error creating tables: {str(e)}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error creating tables: {str(e)}", exc_info=True)
                raise
    
    def _migrate_database_if_needed(self):
        """Check if database migration is needed and perform it if necessary."""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Check for old schema version (missing user_id in observations table)
                    old_schema = False
                    try:
                        # First check if observations table exists
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tweet_observations'")
                        if cursor.fetchone():
                            # Check if it has user_id column
                            cursor.execute("PRAGMA table_info(tweet_observations)")
                            columns = [col[1] for col in cursor.fetchall()]
                            if 'user_id' not in columns:
                                old_schema = True
                                logger.info("Found tweet_observations table without user_id field, migration needed")
                        
                        # Also check for the older followers_count migration
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
                        if cursor.fetchone():
                            cursor.execute("PRAGMA table_info(users)")
                            columns = [col[1] for col in cursor.fetchall()]
                            if 'followers_count' in columns:
                                old_schema = True
                                logger.info("Found users table with old schema, migration needed")
                            
                        if not old_schema:
                            logger.debug("Database already using latest schema, no migration needed")
                            return
                    except sqlite3.OperationalError as e:
                        logger.debug(f"Error checking schema: {str(e)}")
                        # If tables don't exist yet, no migration needed
                        return
                    
                    if old_schema:
                        logger.info("Starting database migration...")
                        
                        # Create temporary tables with new schema
                        cursor.execute("CREATE TABLE users_new (id TEXT PRIMARY KEY, username TEXT NOT NULL, user_created_at INTEGER, description TEXT, verified INTEGER, is_blue_verified INTEGER, last_updated INTEGER DEFAULT (strftime('%s', 'now')))")
                        cursor.execute("CREATE TABLE tweet_observations_new (id INTEGER PRIMARY KEY AUTOINCREMENT, tweet_id TEXT NOT NULL, user_id TEXT NOT NULL, observed_at INTEGER NOT NULL, delta_seconds INTEGER NOT NULL, user_followers_count INTEGER, user_following_count INTEGER, favorite_count INTEGER, retweet_count INTEGER, reply_count INTEGER, quote_count INTEGER, bookmark_count INTEGER, view_count INTEGER, FOREIGN KEY (tweet_id) REFERENCES tweets (id), FOREIGN KEY (user_id) REFERENCES users (id))")
                        
                        # Check if users table has followers_count (old schema)
                        has_followers_count = False
                        cursor.execute("PRAGMA table_info(users)")
                        user_columns = [col[1] for col in cursor.fetchall()]
                        if 'followers_count' in user_columns:
                            has_followers_count = True
                        
                        # Copy users data to new table
                        if has_followers_count:
                            cursor.execute("""
                            INSERT INTO users_new (id, username, user_created_at, description, verified, is_blue_verified, last_updated)
                            SELECT id, username, 0, '', 0, 0, last_updated
                            FROM users
                            """)
                        else:
                            # Just copy existing user data
                            cursor.execute("""
                            INSERT INTO users_new (id, username, user_created_at, description, verified, is_blue_verified, last_updated)
                            SELECT id, username, user_created_at, description, verified, is_blue_verified, last_updated
                            FROM users
                            """)
                        
                        # Get all observations 
                        cursor.execute("SELECT * FROM tweet_observations")
                        observations = cursor.fetchall()
                        
                        # Get column names
                        cursor.execute("PRAGMA table_info(tweet_observations)")
                        columns = [column[1] for column in cursor.fetchall()]
                        
                        # Check if we need to get followers/following counts from users
                        users_data = {}
                        if has_followers_count:
                            cursor.execute("SELECT id, followers_count, following_count FROM users")
                            users_data = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}
                        
                        # For each observation, create a new one with user_id
                        for obs in observations:
                            obs_dict = {columns[i]: obs[i] for i in range(len(columns))}
                            tweet_id = obs_dict.get('tweet_id')
                            
                            # Get the user_id for this tweet
                            cursor.execute("SELECT user_id FROM tweets WHERE id = ?", (tweet_id,))
                            user_id_result = cursor.fetchone()
                            
                            if user_id_result:
                                user_id = user_id_result[0]
                                
                                # Set followers/following counts if migrating from old schema
                                user_followers_count = obs_dict.get('user_followers_count', 0)
                                user_following_count = obs_dict.get('user_following_count', 0)
                                
                                if has_followers_count and user_id in users_data:
                                    followers_count, following_count = users_data[user_id]
                                    user_followers_count = followers_count or 0
                                    user_following_count = following_count or 0
                                
                                # Insert into new observations table with user_id
                                cursor.execute("""
                                INSERT INTO tweet_observations_new 
                                (tweet_id, user_id, observed_at, delta_seconds, user_followers_count, user_following_count, 
                                 favorite_count, retweet_count, reply_count, quote_count, bookmark_count, view_count)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    tweet_id,
                                    user_id,
                                    obs_dict.get('observed_at'),
                                    obs_dict.get('delta_seconds'),
                                    user_followers_count,
                                    user_following_count,
                                    obs_dict.get('favorite_count', 0),
                                    obs_dict.get('retweet_count', 0),
                                    obs_dict.get('reply_count', 0),
                                    obs_dict.get('quote_count', 0),
                                    obs_dict.get('bookmark_count', 0),
                                    obs_dict.get('view_count', 0)
                                ))
                        
                        # Drop old tables and rename new ones
                        cursor.execute("DROP TABLE users")
                        cursor.execute("DROP TABLE tweet_observations")
                        cursor.execute("ALTER TABLE users_new RENAME TO users")
                        cursor.execute("ALTER TABLE tweet_observations_new RENAME TO tweet_observations")
                        
                        # Recreate indices
                        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tweet_user_id ON tweets (user_id)')
                        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tweet_last_updated ON tweets (last_updated)')
                        cursor.execute('CREATE INDEX IF NOT EXISTS idx_observation_tweet_id ON tweet_observations (tweet_id)')
                        cursor.execute('CREATE INDEX IF NOT EXISTS idx_observation_user_id ON tweet_observations (user_id)')
                        
                        conn.commit()
                        logger.info("Database migration completed successfully")
            
            except sqlite3.Error as e:
                logger.error(f"SQLite error during migration: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error during migration: {str(e)}", exc_info=True)
    
    def add_or_update_user(self, user_data, conn=None):
        """Add or update a user in the database.
        
        Args:
            user_data: Dictionary containing user data
            conn: Optional SQLite connection to use (if None, creates a new connection)
        """
        logger.debug(f"Adding or updating user: {user_data['username']} (ID: {user_data['id']})")
        try:
            # Only acquire lock and create connection if not provided
            close_conn = False
            if conn is None:
                with self.lock:
                    conn = sqlite3.connect(self.db_path)
                    close_conn = True
            
            cursor = conn.cursor()
            current_timestamp = int(time.time())
            cursor.execute('''
            INSERT INTO users (id, username, user_created_at, description, verified, is_blue_verified, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                username = excluded.username,
                user_created_at = excluded.user_created_at,
                description = excluded.description,
                verified = excluded.verified,
                is_blue_verified = excluded.is_blue_verified,
                last_updated = excluded.last_updated
            ''', (
                user_data['id'],
                user_data['username'],
                user_data.get('user_created_at', 0),
                user_data.get('description', ''),
                user_data.get('verified', 0),
                user_data.get('is_blue_verified', 0),
                current_timestamp
            ))
            
            # Only commit if we created the connection
            if close_conn:
                conn.commit()
                conn.close()
            
            rows_affected = cursor.rowcount
            logger.debug(f"User {user_data['username']} {'added' if rows_affected > 0 else 'updated'}")
            return rows_affected
        except sqlite3.Error as e:
            # Only rollback if we created the connection
            if conn is not None and close_conn:
                conn.rollback()
                conn.close()
            logger.error(f"SQLite error adding/updating user {user_data['id']}: {str(e)}")
            return 0
        except Exception as e:
            # Only rollback if we created the connection
            if conn is not None and close_conn:
                conn.rollback() 
                conn.close()
            logger.error(f"Unexpected error adding/updating user {user_data['id']}: {str(e)}", exc_info=True)
            return 0
    
    def add_tweet(self, tweet_data, conn=None):
        """Add a tweet to the database if it doesn't already exist.
        
        Args:
            tweet_data: Dictionary containing tweet data
            conn: Optional SQLite connection to use (if None, creates a new connection)
        """
        logger.debug(f"Adding tweet ID {tweet_data['id']} from user ID {tweet_data['user_id']}")
        try:
            # Only acquire lock and create connection if not provided
            close_conn = False
            if conn is None:
                with self.lock:
                    conn = sqlite3.connect(self.db_path)
                    close_conn = True
            
            cursor = conn.cursor()
            
            # Check if tweet already exists
            cursor.execute("SELECT id FROM tweets WHERE id = ?", (tweet_data['id'],))
            existing_tweet = cursor.fetchone()
            
            if existing_tweet:
                logger.debug(f"Tweet {tweet_data['id']} already exists in database")
                
                # Clean up resources if we created the connection
                if close_conn:
                    conn.close()
                
                return False
            
            # Insert new tweet
            current_timestamp = int(time.time())
            cursor.execute('''
            INSERT INTO tweets (id, user_id, created_at, text, monitor_flag, last_updated)
            VALUES (?, ?, ?, ?, 1, ?)
            ''', (
                tweet_data['id'],
                tweet_data['user_id'],
                tweet_data['created_at'],
                tweet_data['text'],
                current_timestamp
            ))
            
            # Only commit if we created the connection
            if close_conn:
                conn.commit()
                conn.close()
            
            logger.debug(f"Tweet {tweet_data['id']} successfully added to database")
            return True
        except sqlite3.Error as e:
            # Only rollback if we created the connection
            if conn is not None and close_conn:
                conn.rollback()
                conn.close()
            logger.error(f"SQLite error adding tweet {tweet_data['id']}: {str(e)}")
            return False
        except Exception as e:
            # Only rollback if we created the connection
            if conn is not None and close_conn:
                conn.rollback()
                conn.close()
            logger.error(f"Unexpected error adding tweet {tweet_data['id']}: {str(e)}", exc_info=True)
            return False
    
    def add_tweet_observation(self, observation_data, conn=None):
        """Add an observation for a tweet.
        
        Args:
            observation_data: Dictionary containing observation data
            conn: Optional SQLite connection to use (if None, creates a new connection)
        """
        logger.debug(f"Adding observation for tweet ID {observation_data['tweet_id']}")
        try:
            # Only acquire lock and create connection if not provided
            close_conn = False
            if conn is None:
                with self.lock:
                    conn = sqlite3.connect(self.db_path)
                    close_conn = True
            
            cursor = conn.cursor()
            
            # Insert observation
            cursor.execute('''
            INSERT INTO tweet_observations (
                tweet_id, user_id, observed_at, delta_seconds, 
                user_followers_count, user_following_count, favorite_count, 
                retweet_count, reply_count, quote_count, 
                bookmark_count, view_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                observation_data['tweet_id'],
                observation_data['user_id'],
                observation_data['observed_at'],
                observation_data['delta_seconds'],
                observation_data.get('user_followers_count', 0),
                observation_data.get('user_following_count', 0),
                observation_data.get('favorite_count', 0),
                observation_data.get('retweet_count', 0),
                observation_data.get('reply_count', 0),
                observation_data.get('quote_count', 0),
                observation_data.get('bookmark_count', 0),
                observation_data.get('view_count', 0)
            ))
            
            # Only commit if we created the connection
            if close_conn:
                conn.commit()
                conn.close()
            
            last_id = cursor.lastrowid
            logger.debug(f"Observation ID {last_id} added for tweet {observation_data['tweet_id']}")
            
            # Log engagement metrics for analysis
            metrics = {
                'user_id': observation_data['user_id'],
                'user_followers_count': observation_data.get('user_followers_count', 0),
                'user_following_count': observation_data.get('user_following_count', 0),
                'favorite_count': observation_data.get('favorite_count', 0),
                'retweet_count': observation_data.get('retweet_count', 0),
                'reply_count': observation_data.get('reply_count', 0),
                'quote_count': observation_data.get('quote_count', 0),
                'bookmark_count': observation_data.get('bookmark_count', 0),
                'view_count': observation_data.get('view_count', 0)
            }
            logger.debug(f"Tweet {observation_data['tweet_id']} metrics: {json.dumps(metrics)}")
            
            return last_id
        except sqlite3.Error as e:
            # Only rollback if we created the connection
            if conn is not None and close_conn:
                conn.rollback()
                conn.close()
            logger.error(f"SQLite error adding observation for tweet {observation_data['tweet_id']}: {str(e)}")
            return None
        except Exception as e:
            # Only rollback if we created the connection
            if conn is not None and close_conn:
                conn.rollback()
                conn.close()
            logger.error(f"Unexpected error adding observation for tweet {observation_data['tweet_id']}: {str(e)}", exc_info=True)
            return None
    
    def store_tweets(self, tweet_package):
        """
        Store a tweet package in the database.
        tweet_package should contain:
        {
            'user': {id, username, followers_count, following_count},
            'tweet': {id, user_id, created_at, text},
            'observation': {tweet_id, created_at, observed_at, favorite_count, retweet_count, ...}
        }
        """
        try:
            tweet_id = tweet_package['tweet']['id']
            logger.debug(f"Processing tweet package for tweet ID: {tweet_id}")
            
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    try:
                        # Start a transaction
                        conn.execute('BEGIN TRANSACTION')
                        
                        # Pass the connection to the helper methods to avoid nested locking
                        logger.debug(f"Adding/updating user: {tweet_package['user']['username']}")
                        user_result = self.add_or_update_user(tweet_package['user'], conn=conn)
                        
                        logger.debug(f"Adding tweet: {tweet_package['tweet']['id']}")
                        is_new_tweet = self.add_tweet(tweet_package['tweet'], conn=conn)
                        logger.debug(f"Tweet {'added as new' if is_new_tweet else 'already exists'}")
                        
                        logger.debug(f"Adding observation for tweet: {tweet_package['observation']['tweet_id']}")
                        observation_id = self.add_tweet_observation(tweet_package['observation'], conn=conn)
                        logger.debug(f"Observation added with ID: {observation_id}")
                        
                        # Commit the transaction
                        conn.commit()
                        logger.info(f"Successfully processed tweet package: {tweet_id}")
                        return True
                    except sqlite3.Error as e:
                        # Rollback in case of error
                        conn.rollback()
                        logger.error(f"SQLite error processing tweet package {tweet_id}: {str(e)}")
                        return False
                    except Exception as e:
                        # Rollback in case of error
                        conn.rollback()
                        logger.error(f"Unexpected error processing tweet package {tweet_id}: {str(e)}", exc_info=True)
                        return False
        except KeyError as e:
            logger.error(f"Invalid tweet package format: missing {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in store_tweets: {str(e)}", exc_info=True)
            return False
    
    def get_tweets_to_monitor(self, limit=100):
        """Get tweets that should be monitored (monitor_flag=1)."""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    logger.debug(f"Getting up to {limit} tweets to monitor")
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute('''
                    SELECT 
                        t.id, t.user_id, t.created_at, t.text, t.monitor_flag, t.last_updated,
                        u.username, u.user_created_at, u.description, u.verified, u.is_blue_verified,
                        o.user_id as observation_user_id, o.user_followers_count, o.user_following_count, o.favorite_count, 
                        o.retweet_count, o.reply_count, o.quote_count, o.bookmark_count, o.view_count,
                        o.observed_at
                    FROM tweets t
                    JOIN users u ON t.user_id = u.id
                    LEFT JOIN (
                        SELECT 
                            tweet_id, user_id,
                            MAX(observed_at) as latest_observation,
                            user_followers_count, user_following_count, favorite_count, 
                            retweet_count, reply_count, quote_count, bookmark_count, view_count,
                            observed_at
                        FROM tweet_observations
                        GROUP BY tweet_id
                    ) o ON t.id = o.tweet_id
                    WHERE t.monitor_flag = 1
                    ORDER BY t.last_updated ASC
                    LIMIT ?
                    ''', (limit,))
                    results = [dict(row) for row in cursor.fetchall()]
                    logger.debug(f"Retrieved {len(results)} tweets to monitor")
                    return results
            except sqlite3.Error as e:
                logger.error(f"SQLite error getting tweets to monitor: {str(e)}")
                return []
            except Exception as e:
                logger.error(f"Unexpected error getting tweets to monitor: {str(e)}", exc_info=True)
                return []
            
    def get_tweets_to_monitor_reverse(self, limit=100):
        """Get tweets that should be monitored (monitor_flag=1)."""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    logger.debug(f"Getting up to {limit} tweets to monitor")
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute('''
                    SELECT 
                        t.id, t.user_id, t.created_at, t.text, t.monitor_flag, t.last_updated,
                        u.username, u.user_created_at, u.description, u.verified, u.is_blue_verified,
                        o.user_id as observation_user_id, o.user_followers_count, o.user_following_count, o.favorite_count, 
                        o.retweet_count, o.reply_count, o.quote_count, o.bookmark_count, o.view_count,
                        o.observed_at
                    FROM tweets t
                    JOIN users u ON t.user_id = u.id
                    LEFT JOIN (
                        SELECT 
                            tweet_id, user_id,
                            MAX(observed_at) as latest_observation,
                            user_followers_count, user_following_count, favorite_count, 
                            retweet_count, reply_count, quote_count, bookmark_count, view_count,
                            observed_at
                        FROM tweet_observations
                        GROUP BY tweet_id
                    ) o ON t.id = o.tweet_id
                    WHERE t.monitor_flag = 1
                    ORDER BY t.last_updated DESC
                    LIMIT ?
                    ''', (limit,))
                    results = [dict(row) for row in cursor.fetchall()]
                    logger.debug(f"Retrieved {len(results)} tweets to monitor")
                    return results
            except sqlite3.Error as e:
                logger.error(f"SQLite error getting tweets to monitor: {str(e)}")
                return []
            except Exception as e:
                logger.error(f"Unexpected error getting tweets to monitor: {str(e)}", exc_info=True)
                return []
    
    def update_tweet_last_updated(self, tweet_id):
        """Update the last_updated timestamp for a tweet."""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    logger.debug(f"Updating last_updated timestamp for tweet {tweet_id}")
                    cursor = conn.cursor()
                    current_timestamp = int(time.time())
                    cursor.execute('''
                    UPDATE tweets
                    SET last_updated = ?
                    WHERE id = ?
                    ''', (current_timestamp, tweet_id))
                    conn.commit()
                    rows_updated = cursor.rowcount
                    logger.debug(f"Updated timestamp for {rows_updated} rows for tweet {tweet_id}")
                    return rows_updated
            except sqlite3.Error as e:
                logger.error(f"SQLite error updating tweet timestamp: {str(e)}")
                return 0
            except Exception as e:
                logger.error(f"Unexpected error updating tweet timestamp: {str(e)}", exc_info=True)
                return 0
    
    def update_monitor_flag(self, tweet_id, flag_value):
        """Update the monitor flag for a tweet."""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    logger.debug(f"Updating monitor flag for tweet {tweet_id} to {flag_value}")
                    cursor = conn.cursor()
                    cursor.execute('''
                    UPDATE tweets
                    SET monitor_flag = ?
                    WHERE id = ?
                    ''', (flag_value, tweet_id))
                    conn.commit()
                    rows_updated = cursor.rowcount
                    logger.debug(f"Updated {rows_updated} rows for tweet {tweet_id}")
                    return rows_updated
            except sqlite3.Error as e:
                logger.error(f"SQLite error updating monitor flag: {str(e)}")
                return 0
            except Exception as e:
                logger.error(f"Unexpected error updating monitor flag: {str(e)}", exc_info=True)
                return 0
    
    def get_last_n_observations(self, tweet_id, n=5):
        """Get the last N observations for a specific tweet.
        
        Args:
            tweet_id: The ID of the tweet to get observations for
            n: The number of observations to retrieve (default: 5)
            
        Returns:
            List of observation dictionaries, ordered by observed_at DESC
        """
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    logger.debug(f"Getting last {n} observations for tweet {tweet_id}")
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute('''
                    SELECT 
                        id, tweet_id, user_id, observed_at, delta_seconds,
                        user_followers_count, user_following_count, favorite_count,
                        retweet_count, reply_count, quote_count, bookmark_count, view_count
                    FROM tweet_observations
                    WHERE tweet_id = ?
                    ORDER BY observed_at DESC
                    LIMIT ?
                    ''', (tweet_id, n))
                    results = [dict(row) for row in cursor.fetchall()]
                    logger.debug(f"Retrieved {len(results)} observations for tweet {tweet_id}")
                    return results
            except sqlite3.Error as e:
                logger.error(f"SQLite error getting observations for tweet {tweet_id}: {str(e)}")
                return []
            except Exception as e:
                logger.error(f"Unexpected error getting observations for tweet {tweet_id}: {str(e)}", exc_info=True)
                return [] 