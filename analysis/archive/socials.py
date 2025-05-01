import asyncio
from datetime import datetime
import twikit
import time
from accounts.model import Account
from account_client import get_account, return_account   # assuming the names
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union
from twikit.errors import TooManyRequests, AccountSuspended, AccountLocked, Unauthorized
import db_utils
import random
import pandas as pd
import yaml
import joblib
from preprocess import _simple_v2
from sentence_transformers import SentenceTransformer
from scipy.sparse import hstack, csr_matrix
from xgboost import XGBRegressor  # type: ignore
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import sys
RATE_LIMITS: Dict[str, List[float]] = {}
RATE_LIMIT_WINDOW: int = 15 * 60  # 15 minutes in seconds
config_path = Path(__file__).parent / "config.yaml"
config = yaml.safe_load(config_path.read_text()) or {}

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
        print(f"Successfully logged in as {username}")
        
        # Save cookies
        cookies_path.parent.mkdir(parents=True, exist_ok=True)
        client.save_cookies(cookies_path)
        print(f"Saved cookies to {cookies_path}")
    except Exception as e:
        print(f"Failed to login account {username}: {e}")
        raise

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
    cookies_path = Path(account.cookies_path)
    
    # Create client
    client = twikit.Client('en-US', user_agent=user_agent)

    if cookies_path.exists():
        try:
            client.load_cookies(cookies_path)
            print(f"Loaded cookies for account {username}")
        except Exception as e:
            print(f"Failed to load cookies for {username}: {e}, attempting login")
            await login_account(client, username, email, password, cookies_path)
    else:
        await login_account(client, username, email, password, cookies_path)
    
    return client

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

def engagement_predict(df: pd.DataFrame, model: XGBRegressor, embedder: SentenceTransformer) -> float:
    """
    Return the engagement-score prediction for every row in *df*.
    Expects the columns ``target`` and ``input_text`` to be present.
    """

    # ------------------------------------------------------------------
    # 1) Separate the columns we need
    # ------------------------------------------------------------------
    text_series = df.pop("input_text")            # keep the text separately
    if "target" in df.columns:                    # drop the training target
        df = df.drop(columns=["target"])

    # ------------------------------------------------------------------
    # 2) Numeric features – make sure everything is float
    # ------------------------------------------------------------------
    numeric_np = df.select_dtypes(exclude=["object"]).to_numpy(dtype=float)
    numeric_sparse = csr_matrix(numeric_np)       # shape = (n_rows, n_numeric)

    # ------------------------------------------------------------------
    # 3) Sentence-Transformer embeddings
    #    • always pass a *list* of str
    #    • ask for NumPy
    # ------------------------------------------------------------------
    embeddings = embedder.encode(
        text_series.tolist(),                     # <-- LIST, not ndarray
        convert_to_numpy=True,
        show_progress_bar=False
    )                                             # shape = (n_rows, emb_dim)

    if embeddings.ndim == 1:                      # happens for a single row
        embeddings = embeddings.reshape(1, -1)

    emb_sparse = csr_matrix(embeddings)           # keep everything sparse

    # ------------------------------------------------------------------
    # 4) Concatenate horizontally
    # ------------------------------------------------------------------
    combined = hstack([numeric_sparse, emb_sparse])

    # ------------------------------------------------------------------
    # 5) Predict
    # ------------------------------------------------------------------
    return float(model.predict(combined)[0])

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
        return_tweets = []
        tweets = await client.search_tweet(query, product=sort_by, count=count)
        return_tweets.extend(list(tweets))
        more_tweets = await tweets.next()
        while len(more_tweets) > 0:
            return_tweets.extend(more_tweets)
            more_tweets = await more_tweets.next()
        return return_tweets
    except Unauthorized:
        raise
    except TooManyRequests:
        raise
    except AccountSuspended:
        raise
    except AccountLocked:
        raise
    except Exception as e:
        print(f"Failed to search tweets: {e}")
        return f"Failed to search tweets: {e}"

async def get_tweets_from_user_link(client: twikit.Client, user_link: str) -> Tuple[List[twikit.Tweet], int]:
    # get user id
    try:
        username = user_link.split("/")[-1]
        user_id = await client.get_user_by_screen_name(username)
        if user_id is None:
            return []
        return_tweets = []
        tweets = await client.get_user_tweets(user_id.id, 'Tweets')
        return_tweets.extend(list(tweets))
        more_tweets = await tweets.next()
        while len(more_tweets) > 0:
            return_tweets.extend(more_tweets)
            latest_tweet = return_tweets[-1]
            latest_tweet_timestamp = datetime_to_timestamp(latest_tweet.created_at_datetime)
            if latest_tweet_timestamp < 1745280000:
                break
            more_tweets = await more_tweets.next()
        return return_tweets
    except Unauthorized:
        raise
    except TooManyRequests:
        raise
    except AccountSuspended:
        raise
    except AccountLocked:
        raise
    except Exception as e:
        print(f"Failed to search tweets: {e}")
        return []
    
def get_sentiment(text, tokenizer, model):
    def preprocess_tweet(text):
        """Preprocess tweet by replacing usernames and URLs with placeholders"""
        words = []
        for word in text.split():
            if word.startswith('@'):
                words.append('@user')
            elif word.startswith('http'):
                words.append('http')
            else:
                words.append(word)
        return ' '.join(words)

    def process_text(text, tokenizer):
        inputs = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        # Convert dict of tensors to tensors and remove batch dimension
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
    
    text = preprocess_tweet(text)
    inputs = process_text(text, tokenizer)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs  = torch.softmax(logits, dim=-1).squeeze(0).cpu()   # [num_labels]
        pred   = torch.argmax(probs).item()           # int

    # Optional: map id → label if the model has that info
    MAP = {0: "neutral", 1: "positive", 2: "negative"}
    return MAP[pred], float(probs.numpy()[pred])

from contextlib import asynccontextmanager

@asynccontextmanager
async def borrowed_account():
    acc = await get_account()
    print("✅  Got", acc.username)
    try:
        yield acc
    finally:
        await return_account(acc)
        print("↩️  Returned", acc.username)

# For each account, we need to get the tweets that the account has made in the last 4 days


async def main():
    async with borrowed_account() as acc:
        twitter_client = await get_twitter_client(acc)
        engagement_model = joblib.load("models/engagement/model.joblib")
        engagement_embedder = SentenceTransformer(config["train_regressor"]["embedding"]["sentence_transformers"]["model_name"])
        sentiment_tokenizer = AutoTokenizer.from_pretrained("models/sentiment")
        sentiment_model = AutoModelForSequenceClassification.from_pretrained("models/sentiment")

        twitter_grouped = pd.read_csv("data/post/twitter_grouped.csv")
        tweets_df = pd.read_csv("data/post/tweets.csv")
        # get list of user links
        # filter out users that have already been processed
        twitter_grouped = twitter_grouped[twitter_grouped['processed'] == 0]
        user_list = twitter_grouped['twitter'].tolist()
        print(len(user_list))
        # For each user link
        # get the tweets made by that user in the last 4 days
        # insert each tweet into the tweets table
        try:
            for user_link in user_list:
                try:
                    # wait another 20 seconds with a 50% chance
                    random_value = random.random()
                    if random_value < 0.35:
                        print("Waiting another 20 seconds")
                        await asyncio.sleep(20)

                    # TODO
                    tweets: List[twikit.Tweet] = await get_tweets_from_user_link(twitter_client, user_link)

                    if isinstance(tweets, str):
                        # An error occurred
                        print(f"Search failed: {tweets}")
                        await asyncio.sleep(20)
                        continue

                    # process the tweets
                    for tweet in tweets:
                        user = tweet.user
                        observed_at = int(datetime.now().timestamp())
                        tweet_created_at = datetime_to_timestamp(tweet.created_at_datetime)
                        favorite_count = 0 if tweet.favorite_count is None else int(tweet.favorite_count)
                        tweet_dict = {
                            "observation_id": 0,
                            "tweet_id": tweet.id,
                            "observed_at": observed_at,
                            "tweet_created_at": tweet_created_at,
                            "delta_seconds": 3600*24,
                            "user_followers_count": user.followers_count,
                            "user_following_count": user.following_count,
                            "favorite_count": 0 if tweet.favorite_count is None else int(tweet.favorite_count),
                            "retweet_count": 0 if tweet.retweet_count is None else int(tweet.retweet_count),
                            "reply_count": 0 if tweet.reply_count is None else int(tweet.reply_count),
                            "quote_count": 0 if tweet.quote_count is None else int(tweet.quote_count),
                            "bookmark_count": 0 if tweet.bookmark_count is None else int(tweet.bookmark_count),
                            "view_count": 0 if tweet.view_count is None else int(tweet.view_count),
                            "tweet_text": tweet.text,
                            "monitor_flag": 0,
                            "tweet_last_updated": observed_at,
                            "user_id": user.id,
                            "username": user.name,
                            "user_description": user.description,
                            "verified": user.verified,
                            "is_blue_verified": user.is_blue_verified,
                            "user_created_at": datetime_to_timestamp(user.created_at_datetime),
                            "user_last_updated": observed_at,
                        }
                        
                        # turn tweet_dict into a dataframe
                        df = pd.DataFrame([tweet_dict])
                        processed_df = _simple_v2(df, config)

                        # predict the engagement score
                        engagement_score = engagement_predict(processed_df, engagement_model, engagement_embedder)
                        sentiment_label, sentiment_label_prob = get_sentiment(tweet.text, sentiment_tokenizer, sentiment_model)

                        # append  the scores/labels to the tweet_dict
                        tweet_dict["tweet_sentiment_label"] = sentiment_label
                        tweet_dict["tweet_sentiment_label_prob"] = sentiment_label_prob
                        tweet_dict["tweet_engagement_score"] = engagement_score
                        tweet_dict["coin_address"] = "unknown"
                        tweet_dict["pair_address"] = "unknown"

                        # insert the tweet into the tweets table using concat
                        tweets_df = pd.concat([tweets_df, pd.DataFrame([tweet_dict])], ignore_index=True)
                        

                    # mark the user as processed
                    twitter_grouped.loc[twitter_grouped['twitter'] == user_link, 'processed'] = 1
                    twitter_grouped.to_csv("data/post/twitter_grouped.csv", index=False)
                    tweets_df.to_csv("data/post/tweets.csv", index=False)
                    await asyncio.sleep(20)
                except Unauthorized:
                    print("Unauthorized, shutting down")
                    raise
                except TooManyRequests:
                    print("Rate limit exceeded, waiting 15 minutes")
                    await asyncio.sleep(900)
                except AccountSuspended:
                    print("Account suspended, waiting 15 minutes")
                    await asyncio.sleep(900)
                except AccountLocked:
                    print("Account locked, shutting down")
                    raise
                except Exception as e:
                    print(f"Error in main function: {str(e)}")
                    await asyncio.sleep(20)
        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Shutting down gracefully...")
        except Exception as e:
            print(f"Error in main function: {str(e)}")
        finally:
            await return_account(acc)
            print("↩️  Returned", acc.username)

if __name__ == "__main__":
    asyncio.run(main())
