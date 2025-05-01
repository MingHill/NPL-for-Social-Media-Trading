# get account from account_service.py

import twikit
import os
import shutil
from pathlib import Path
import logging
import time
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Union, cast
from account_service import AccountClient
from account import Account
from sql_client import DirectSQLClient
import sys
import random
from twikit.errors import TooManyRequests, AccountSuspended, AccountLocked, Unauthorized
# Create logs directory if it doesn't exist
logs_dir = Path('logs')
logs_dir.mkdir(exist_ok=True)

# Define log file path
log_file = logs_dir / 'tweet_processing.log'

# Set up logging with file handler for persistent logs
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG level to get more detailed logs

# Clean up old log file if it exists
if log_file.exists():
    try:
        log_file.unlink()  # Delete the existing log file
        logger.info(f"Deleted old log file: {log_file}")
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

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

# Rate limit tracking
RATE_LIMITS: Dict[str, List[float]] = {}
RATE_LIMIT_WINDOW: int = 15 * 60  # 15 minutes in seconds

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

async def get_twitter_client(account: Account) -> Tuple[twikit.Client, Account]:
    """Initialize and return an authenticated Twitter client using account from account service.
    
    Returns:
        Tuple containing (twitter_client, account)
    
    Raises:
        Exception: If no accounts are available from the account service
    """    
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

async def login_account(
    client: twikit.Client, 
    username: str, 
    email: str, 
    password: str, 
    cookies_path: Path
) -> None:
    """Login to Twitter account and save cookies.
    
    Args:
        client: Twitter client to authenticate
        username: Twitter username
        email: Account email
        password: Account password
        cookies_path: Path to save cookies to
        
    Raises:
        Exception: If login fails
    """
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

def check_rate_limit(endpoint: str) -> bool:
    """Check if we're within rate limits for a given endpoint.
    
    Args:
        endpoint: The API endpoint to check rate limits for
        
    Returns:
        True if under rate limit, False if rate limited
    """
    now = time.time()
    if endpoint not in RATE_LIMITS:
        RATE_LIMITS[endpoint] = []

    # Remove old timestamps
    RATE_LIMITS[endpoint] = [t for t in RATE_LIMITS[endpoint] if now - t < RATE_LIMIT_WINDOW]

    # Check limits based on endpoint
    if endpoint == 'tweet':
        return len(RATE_LIMITS[endpoint]) < 300  # 300 tweets per 15 minutes
    elif endpoint == 'dm':
        return len(RATE_LIMITS[endpoint]) < 1000  # 1000 DMs per 15 minutes
    return True

async def search_twitter(
    client: twikit.Client, 
    query: str, 
    sort_by: str = 'Top', 
    count: int = 10
) -> Union[List[Any], str]:
    """Search twitter with a query.
    
    Args:
        client: Authenticated Twitter client
        query: Search query string
        sort_by: Sort method ('Top' or 'Latest')
        count: Number of tweets to return
        
    Returns:
        List of tweet objects or error message string
    """
    try:
        tweets = await client.search_tweet(query, product=sort_by, count=count)
        return list(tweets)
    except Unauthorized:
        raise
    except TooManyRequests:
        raise
    except AccountSuspended:
        raise
    except AccountLocked:
        raise
    except Exception as e:
        logger.error(f"Failed to search tweets: {e}")
        return f"Failed to search tweets: {e}"

def process_tweets(tweets: List[twikit.Tweet], sql_client: DirectSQLClient) -> int:
    """Process tweets and create packages to send to SQL client.
    
    Args:
        tweets: List of tweet objects from Twitter API
        sql_client: SQL client to store tweets with
        
    Returns:
        Number of successfully processed tweets
    """
    processed_count = 0
    current_timestamp = int(time.time())  # Current time as Unix timestamp
    
    logger.info(f"Starting to process {len(tweets)} tweets")
    
    for tweet in tweets:
        try:
            # print tweet lang
            # language check
            if tweet.lang != 'en':
                logger.debug(f"Skipping non-English tweet: {tweet.text}")
                continue

            # Extract tweet data
            tweet_id = tweet.id
            tweet_text = tweet.text
            
            # Log basic tweet information
            logger.debug(f"Processing tweet ID: {tweet_id}")
            logger.debug(f"Tweet text: {tweet_text[:50]}..." if len(tweet_text) > 50 else f"Tweet text: {tweet_text}")
            
            # Convert created_at to timestamp
            tweet_created_timestamp = datetime_to_timestamp(tweet.created_at_datetime)
            
            # Extract user data
            user_id = tweet.user.id
            username = tweet.user.screen_name
            
            # Get user account creation timestamp
            user_created_at = 0
            if hasattr(tweet.user, 'created_at_datetime') and tweet.user.created_at_datetime:
                user_created_at = datetime_to_timestamp(tweet.user.created_at_datetime)
            
            # Extract user verification status
            verified = 0
            if hasattr(tweet.user, 'verified') and tweet.user.verified:
                verified = 1
                
            is_blue_verified = 0
            if hasattr(tweet.user, 'is_blue_verified') and tweet.user.is_blue_verified:
                is_blue_verified = 1
                
            # Extract user description
            description = ""
            if hasattr(tweet.user, 'description') and tweet.user.description:
                description = tweet.user.description
            
            # Extract follower/following counts for observation
            followers_count = tweet.user.followers_count or 0
            following_count = tweet.user.following_count or 0
            
            # Extract and sanitize view_count - ensure it's always 0 if None
            view_count = getattr(tweet, 'view_count', 0)
            if view_count is None:
                view_count = 0
                logger.debug(f"Tweet {tweet_id} had None view_count, set to 0")
            
            # Create observation data with user_id and follower/following counts
            observation: Dict[str, Any] = {
                'tweet_id': tweet_id,
                'user_id': user_id,  # Add user_id to observation
                'created_at': tweet_created_timestamp,
                'observed_at': current_timestamp,
                'delta_seconds': current_timestamp - tweet_created_timestamp,
                'user_followers_count': followers_count,
                'user_following_count': following_count,
                'favorite_count': tweet.favorite_count or 0,
                'retweet_count': tweet.retweet_count or 0,
                'reply_count': getattr(tweet, 'reply_count', 0) or 0,
                'quote_count': getattr(tweet, 'quote_count', 0) or 0,
                'bookmark_count': getattr(tweet, 'bookmark_count', 0) or 0,
                'view_count': view_count  # Using the sanitized view_count
            }
            
            # Create user data with new fields
            user: Dict[str, Any] = {
                'id': user_id,
                'username': username,
                'user_created_at': user_created_at,
                'description': description,
                'verified': verified,
                'is_blue_verified': is_blue_verified
            }
            
            # Create tweet data
            tweet_data: Dict[str, Any] = {
                'id': tweet_id,
                'user_id': user_id,
                'created_at': tweet_created_timestamp,
                'text': tweet_text
            }
            
            # Create the complete package
            tweet_package: Dict[str, Dict[str, Any]] = {
                'user': user,
                'tweet': tweet_data,
                'observation': observation
            }
            
            # Log the tweet package (but redact very long text fields)
            safe_log_package = json.dumps({
                'user': {**user, 'description': user['description'][:50] + ('...' if len(user['description']) > 50 else '')},
                'tweet': {**tweet_data, 'text': tweet_data['text'][:100] + ('...' if len(tweet_data['text']) > 100 else '')},
                'observation': observation
            }, indent=2)
            logger.debug(f"Tweet package prepared: {safe_log_package}")
            
            # Store in database
            logger.debug(f"Attempting to store tweet {tweet_id} in database")
            success = sql_client.store_tweets(tweet_package)
            if success:
                processed_count += 1
                logger.info(f"Successfully stored tweet {tweet_id} from {username}")
            else:
                logger.error(f"Failed to store tweet {tweet_id} - SQL client returned failure")
                
        except Exception as e:
            logger.error(f"Error processing tweet: {str(e)}", exc_info=True)  # Include stack trace
    
    logger.info(f"Processed {processed_count} tweets successfully out of {len(tweets)} attempted")
    return processed_count


async def main() -> None:
    """Main function to run the script for testing."""
    account = None
    account_client = None
    try:
        # Set up diagnostic information
        logger.info(f"Starting tweet processing script - PID: {os.getpid()}")
        logger.info(f"Python version: {sys.version}")
        
        # Check if tweets.db exists and log its size
        db_path = Path("tweets.db")
        if db_path.exists():
            db_size = db_path.stat().st_size / 1024  # Convert to KB
            logger.info(f"Found tweets.db: {db_size:.2f} KB")
        else:
            logger.warning("tweets.db not found - it will be created by the SQL client")
        
        # Initialize clients
        logger.info("Initializing Direct SQL client...")
        sql_client = DirectSQLClient()

        # initialize account client
        account_client = AccountClient()
        account = account_client.get_account()
        if account is None:
            logger.error("No account available from account service")
            raise Exception("No account available")
        logger.info(f"Using Twitter account: {account.username}")

        
        logger.info("Initializing Twitter client...")
        client = await get_twitter_client(account)
        
        # Test query parameters
        query = "bitcoin"
        tweets_total = 0
        processed_total = 0

        while True:
            try:
                # wait another 20 seconds with a 50% chance
                random_value = random.random()
                if random_value < 0.45:
                    logger.info("Waiting another 20 seconds")
                    await asyncio.sleep(20)
                    continue
                elif random_value < 0.95:
                    pass
                elif random_value < 0.99:
                    # trigger medium wait of 5 minutes
                    logger.info("Triggering medium wait of 5 minutes")
                    await asyncio.sleep(300)
                    continue
                else:
                    logger.info("Triggering long wait of 15 minutes")
                    await asyncio.sleep(900)
                    continue

                # query
                logger.info(f"Searching Twitter for query: '{query}'")
                tweets = await search_twitter(client, query, sort_by="Latest", count=20)
                
                if isinstance(tweets, str):
                    # An error occurred
                    logger.error(f"Search failed: {tweets}")
                    await asyncio.sleep(20)
                    continue
                    
                logger.info(f"Found {len(tweets)} tweets for query: '{query}'")
                
                # Process tweets
                stored_count = process_tweets(tweets, sql_client)
                processed_total += stored_count
                tweets_total += len(tweets)
                
                # Wait before next query to avoid rate limits
                await asyncio.sleep(20)     
                
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
                logger.error(f"Error in main function: {str(e)}", exc_info=True)
                await asyncio.sleep(20)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected. Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}", exc_info=True)
    finally:
        # return account to account service if it was acquired
        if account is not None and account_client is not None:
            try:
                account_client.return_account(account)
                logger.info(f"Account {account.username} returned to account service")
            except Exception as e:
                logger.error(f"Error returning account to service: {e}")
        logger.info("Tweet processing script completed")

if __name__ == "__main__":
    # Make sure we import sys for Python version info
    import sys
    
    try:
        # Run the main async function
        asyncio.run(main())
    except KeyboardInterrupt:
        # This catch is for the case where KeyboardInterrupt happens before 
        # asyncio.run starts or after it completes
        print("\nScript terminated by keyboard interrupt")