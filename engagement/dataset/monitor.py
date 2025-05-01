import asyncio
import logging
import time
from pathlib import Path
from datetime import datetime
import sys
import sqlite3
import random
import argparse

from account_service import AccountClient
from sql_client import DirectSQLClient
import twikit
from twikit.errors import TooManyRequests, AccountSuspended, AccountLocked, Unauthorized

# Set up argument parser
parser = argparse.ArgumentParser(description='Twitter monitoring service')
parser.add_argument('--reverse', action='store_true', default=False, help='Optional reverse parameter')

# Create logs directory if it doesn't exist
logs_dir = Path('logs')
logs_dir.mkdir(exist_ok=True)

# Define log file path
log_file = logs_dir / 'monitor.log'

# Set up logging with file handler for persistent logs
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG level to get more detailed logs

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

# Suppress excessive httpx logging
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

async def get_twitter_client(account):
    """Initialize and return an authenticated Twitter client using account from account service."""
    # Extract account information
    username = account.username
    email = account.email
    password = account.password
    user_agent = account.user_agent
    cookies_path = Path(account.get_cookies_path())
    
    # Create client
    client = twikit.Client('en-US', user_agent=user_agent)

    if cookies_path.exists():
        try:
            client.load_cookies(cookies_path)
            logger.info(f"Loaded cookies for account {username}")
        except Exception as e:
            logger.warning(f"Failed to load cookies for {username}: {e}, attempting login")
            await login_account(client, username, email, password, cookies_path)
    else:
        await login_account(client, username, email, password, cookies_path)
    
    return client

async def login_account(client, username, email, password, cookies_path):
    """Login to Twitter account and save cookies."""
    try:
        await client.login(
            auth_info_1=username,
            auth_info_2=email,
            password=password
        )
        logger.info(f"Successfully logged in as {username}")
        
        # Save cookies
        cookies_path.parent.mkdir(parents=True, exist_ok=True)
        client.save_cookies(cookies_path)
        logger.info(f"Saved cookies to {cookies_path}")
    except Exception as e:
        logger.error(f"Failed to login account {username}: {e}")
        raise

async def get_tweet_by_id(client, tweet_id):
    """Fetch a tweet by its ID from Twitter."""
    try:
        tweet = await client.get_tweet_by_id(tweet_id)
        return tweet
    except Unauthorized:
        raise
    except TooManyRequests:
        raise
    except AccountSuspended:
        raise
    except AccountLocked:
        raise
    except Exception as e:
        logger.error(f"Failed to get tweet {tweet_id}: {e}")
        return None
    
def datetime_to_timestamp(dt) -> int:
    """Convert a datetime object to Unix timestamp (seconds since epoch).
    
    Args:
        dt: Datetime object or string
        
    Returns:
        Unix timestamp as an integer
    """
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    
    # Convert to timestamp (seconds since epoch)
    return int(dt.timestamp())

def create_observation_from_tweet(tweet: twikit.Tweet, user_id: str):
    """Create an observation dictionary from a tweet object."""
    current_timestamp = int(time.time())
    tweet_created_timestamp = datetime_to_timestamp(tweet.created_at_datetime)
    
    observation = {
        'tweet_id': tweet.id,
        'user_id': user_id,
        'created_at': tweet_created_timestamp,
        'observed_at': current_timestamp,
        'delta_seconds': current_timestamp - tweet_created_timestamp,
        'user_followers_count': tweet.user.followers_count or 0,
        'user_following_count': tweet.user.following_count or 0,
        'favorite_count': tweet.favorite_count or 0,
        'retweet_count': tweet.retweet_count or 0,
        'reply_count': getattr(tweet, 'reply_count', 0) or 0,
        'quote_count': getattr(tweet, 'quote_count', 0) or 0,
        'bookmark_count': getattr(tweet, 'bookmark_count', 0) or 0,
        'view_count': getattr(tweet, 'view_count', 0) or 0
    }
    
    return observation

def are_metrics_stable(observations, min_observations=3):
    """Check if engagement metrics have remained stable across observations.
    
    Args:
        observations: List of observation dictionaries
        min_observations: Minimum number of observations required to check stability
        
    Returns:
        Boolean indicating if metrics are stable (True) or have changed (False)
    """
    if len(observations) < min_observations:
        logger.debug(f"Not enough observations ({len(observations)}) to determine stability, need {min_observations}")
        return False
    
    # Check only the last min_observations observations
    observations = observations[:min_observations]

    # # log the observations
    # for obs in observations:
    #     logger.debug(f"Observation: {obs}")
    
    # Define the metrics to compare
    metrics = ['favorite_count', 'retweet_count', 'reply_count', 'quote_count', 'bookmark_count']
    
    # Get the first observation's metrics
    first_obs = observations[0]
    metric_values = {metric: first_obs[metric] for metric in metrics}
    
    # Log the metrics we're comparing
    logger.debug(f"Checking stability of metrics across {min_observations} observations")
    for idx, metric in enumerate(metrics):
        metric_list = [obs[metric] for obs in observations]
        logger.debug(f"  {metric}: {metric_list}")
    
    # Compare all observations to see if any metrics changed
    for obs in observations[1:]:
        for metric in metrics:
            if obs[metric] != metric_values[metric]:
                # Metric changed, not stable
                logger.debug(f"Detected change in {metric}: {metric_values[metric]} vs {obs[metric]}")
                return False
    
    # If we reach here, all observations had identical metrics
    logger.info("All metrics are stable across the last 3 observations")
    return True

async def monitor_tweet(client, sql_client, tweet_data):
    """Monitor a single tweet by fetching it and creating a new observation."""
    tweet_id = tweet_data['id']
    user_id = tweet_data['user_id']
    logger.info(f"Monitoring tweet {tweet_id} from user {tweet_data['username']}")
    
    # First, check if metrics have been stable in recent observations
    previous_observations = sql_client.get_last_n_observations(tweet_id, 3)
    if len(previous_observations) >= 3 and are_metrics_stable(previous_observations):
        logger.info(f"Tweet {tweet_id} has stable metrics across last 3 observations, marking as done")
        sql_client.update_monitor_flag(tweet_id, 0)
        # Still update last_updated to prevent immediate retrieval in case monitor is restarted
        sql_client.update_tweet_last_updated(tweet_id)
        return True
    
    # Fetch current tweet data from Twitter
    tweet = await get_tweet_by_id(client, tweet_id)
    
    if not tweet:
        logger.warning(f"Failed to retrieve tweet {tweet_id}, may have been deleted")
        # Update the last_updated timestamp even for failed retrievals
        # to prevent constant retrying of unavailable tweets
        sql_client.update_tweet_last_updated(tweet_id)
        
        # If the tweet has a field for failed_attempts, we could increment it here
        # and disable monitoring after several failures
        # For now, we'll just stop monitoring after this attempt
        sql_client.update_monitor_flag(tweet_id, 0)
        logger.info(f"Disabled monitoring for tweet {tweet_id} as it could not be retrieved")
        return False
    
    # Create a new observation
    observation = create_observation_from_tweet(tweet, user_id)
    
    # Add the observation to the database
    result = sql_client.add_tweet_observation(observation)
    if result:
        logger.info(f"Added new observation for tweet {tweet_id}")
        logger.info(f"Engagement metrics - Likes: {observation['favorite_count']}, Retweets: {observation['retweet_count']}, " +
                   f"Replies: {observation['reply_count']}, Views: {observation['view_count']}")
        
        # Update the tweet's last_updated timestamp
        sql_client.update_tweet_last_updated(tweet_id)
        logger.debug(f"Updated last_updated timestamp for tweet {tweet_id}")
        
        # Check if metrics are stable now that we've added a new observation
        updated_observations = sql_client.get_last_n_observations(tweet_id, 3)
        if len(updated_observations) >= 3 and are_metrics_stable(updated_observations):
            logger.info(f"Tweet {tweet_id} has stable metrics across last 3 observations, marking as done")
            sql_client.update_monitor_flag(tweet_id, 0)
        
        return True
    else:
        logger.error(f"Failed to add observation for tweet {tweet_id}")
        return False

async def get_monitoring_stats(sql_client):
    """Get statistics about tweet monitoring.
    
    Returns:
        Dictionary with monitoring statistics
    """
    try:
        # This would be better as a SQL query, but for now we'll use Python
        # Get all tweets
        conn = sqlite3.connect(sql_client.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get total tweet count
        cursor.execute("SELECT COUNT(*) FROM tweets")
        total_tweets = cursor.fetchone()[0]
        
        # Get monitored tweet count
        cursor.execute("SELECT COUNT(*) FROM tweets WHERE monitor_flag = 1")
        monitored_tweets = cursor.fetchone()[0]
        
        # Get completed tweet count
        cursor.execute("SELECT COUNT(*) FROM tweets WHERE monitor_flag = 0")
        completed_tweets = cursor.fetchone()[0]
        
        # Get total observation count
        cursor.execute("SELECT COUNT(*) FROM tweet_observations")
        total_observations = cursor.fetchone()[0]
        
        # Get average observations per tweet
        if total_tweets > 0:
            cursor.execute("""
            SELECT AVG(cnt) FROM (
                SELECT COUNT(*) as cnt 
                FROM tweet_observations 
                GROUP BY tweet_id
            )
            """)
            avg_observations = cursor.fetchone()[0]
        else:
            avg_observations = 0
            
        # Close connection
        conn.close()
            
        return {
            "total_tweets": total_tweets,
            "monitored_tweets": monitored_tweets,
            "completed_tweets": completed_tweets,
            "total_observations": total_observations,
            "avg_observations_per_tweet": round(avg_observations, 2) if avg_observations else 0
        }
    except Exception as e:
        logger.error(f"Error getting monitoring stats: {e}")
        return {
            "total_tweets": 0,
            "monitored_tweets": 0,
            "completed_tweets": 0,
            "total_observations": 0,
            "avg_observations_per_tweet": 0
        }

async def main():
    """Main function to run the tweet monitoring service."""
    # Parse command line arguments
    args = parser.parse_args()
    reverse = args.reverse
    
    logger.info(f"Starting tweet monitoring service with reverse_order={reverse}")
    
    account = None
    account_client = None
    
    try:
        # Initialize SQL client
        sql_client = DirectSQLClient()
        logger.info("DirectSQLClient initialized")
        
        # Get an account from the account service
        account_client = AccountClient()
        account = account_client.get_account()
        if not account:
            logger.error("Failed to get account from account service")
            return
        logger.info(f"Using Twitter account: {account.username}")
        
        # Initialize Twitter client
        client = await get_twitter_client(account)
        logger.info("Twitter client initialized")
        
        # Main monitoring loop
        while True:
            try:
                # wait 10 seconds with a 70% chance
                random_value = random.random()
                if random_value < 0.25:
                    logger.info("Waiting another 10 seconds")
                    await asyncio.sleep(10)
                    continue
                
                # Display monitoring stats
                stats = await get_monitoring_stats(sql_client)
                logger.info(f"Monitoring stats: {stats['monitored_tweets']} monitored, " +
                            f"{stats['completed_tweets']} completed, {stats['total_observations']} total observations")
                
                # Get tweets to monitor, sorted by least recently updated first
                random_tweet_count = random.randint(1, 20)
                if reverse:
                    tweets_to_monitor = sql_client.get_tweets_to_monitor_reverse(limit=random_tweet_count)
                else:
                    tweets_to_monitor = sql_client.get_tweets_to_monitor(limit=random_tweet_count)
                if not tweets_to_monitor:
                    logger.info("No tweets to monitor, waiting...")
                    await asyncio.sleep(30)  # Wait longer if there are no tweets
                    continue
                
                logger.info(f"Got {len(tweets_to_monitor)} tweets to monitor")
                
                # Log the tweets we're about to monitor in order
                for i, tweet in enumerate(tweets_to_monitor):
                    last_updated_time = datetime.fromtimestamp(tweet['last_updated'])
                    logger.info(f"Queue position {i+1}: Tweet {tweet['id']} (last updated: {last_updated_time.strftime('%Y-%m-%d %H:%M:%S')})")
                
                # Process each tweet
                for tweet_data in tweets_to_monitor:
                    await monitor_tweet(client, sql_client, tweet_data)
                    # Sleep between requests to avoid rate limits
                    await asyncio.sleep(10)

            except Unauthorized:
                logger.error("Unauthorized, shutting down")
                raise
            except TooManyRequests:
                logger.warning("Rate limit exceeded, waiting 15 minutes")
                await asyncio.sleep(900)
            except AccountSuspended:
                logger.error("Account suspended, waiting 15 minutes")
                await asyncio.sleep(900)
            except AccountLocked:
                logger.error("Account locked, shutting down")
                raise
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(30)  # Wait a bit longer on error
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
    finally:
        # Return the account to the account service
        if account and account_client:
            try:
                account_client.return_account(account)
                logger.info(f"Account {account.username} returned to account service")
            except Exception as e:
                logger.error(f"Error returning account: {e}")
        
        logger.info("Tweet monitoring service shut down")

if __name__ == "__main__":
    try:
        # Run the async main function
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nMonitor terminated by keyboard interrupt") 