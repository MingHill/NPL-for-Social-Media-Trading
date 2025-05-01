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
from preprocess import _simple_v2
import joblib
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
    text_series = df.pop("input_text")            # keep the text separately
    if "target" in df.columns:                    # drop the training target
        df = df.drop(columns=["target"])

    numeric_np = df.select_dtypes(exclude=["object"]).to_numpy(dtype=float)
    numeric_sparse = csr_matrix(numeric_np)       # shape = (n_rows, n_numeric)

    embeddings = embedder.encode(
        text_series.tolist(),                     # <-- LIST, not ndarray
        convert_to_numpy=True,
        show_progress_bar=False
    )                                             # shape = (n_rows, emb_dim)

    if embeddings.ndim == 1:                      # happens for a single row
        embeddings = embeddings.reshape(1, -1)

    emb_sparse = csr_matrix(embeddings)           # keep everything sparse

    combined = hstack([numeric_sparse, emb_sparse])

    return float(model.predict(combined)[0])

async def search_twitter(
    client: twikit.Client, 
    query: str, 
    sort_by: str = 'Top', 
    count: int = 10
) -> Union[List[Any], str]:
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

async def main():
    acc = await get_account()
    if acc is None:
        print("⛔  No account available – is the server running and loaded?")
        return

    print("✅  Got", acc.username)
    twitter_client = await get_twitter_client(acc)

    engagement_model = joblib.load("models/engagement/model.joblib")
    engagement_embedder = SentenceTransformer(config["train_regressor"]["embedding"]["sentence_transformers"]["model_name"])
    sentiment_tokenizer = AutoTokenizer.from_pretrained("models/sentiment")
    sentiment_model = AutoModelForSequenceClassification.from_pretrained("models/sentiment")

    # Get coins from the coin table where the rug flag is true
    conn = db_utils.get_connection()
    db_utils.create_completed_table()
    db_utils.create_tweets_table()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT c.* FROM coin_table c"
        " LEFT JOIN completed_table comp ON c.coin_address = comp.coin_address AND c.pair_address = comp.pair_address"
        " WHERE c.rugpull_flag = 1 AND comp.pair_address IS NULL"
    )
    rows = cursor.fetchall()
    try:
        for row in rows:
            try:
                # wait another 20 seconds with a 50% chance
                random_value = random.random()
                if random_value < 0.35:
                    print("Waiting another 20 seconds")
                    await asyncio.sleep(20)
                coin_address = row["coin_address"]
                pair_address = row["pair_address"]
                # Search for the coin address on twitter
                query = f"{coin_address} OR {pair_address}"
                tweets: List[twikit.Tweet] = await search_twitter(twitter_client, query, sort_by="Latest", count=20)

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
                    tweet_dict = {
                        "observation_id": 0,
                        "tweet_id": tweet.id,
                        "observed_at": observed_at,
                        "tweet_created_at": tweet_created_at,
                        "delta_seconds": 3600*24,
                        "user_followers_count": user.followers_count,
                        "user_following_count": user.following_count,
                        "favorite_count": int(tweet.favorite_count),
                        "retweet_count": int(tweet.retweet_count),
                        "reply_count": int(tweet.reply_count),
                        "quote_count": int(tweet.quote_count),
                        "bookmark_count": int(tweet.bookmark_count),
                        "view_count": int(tweet.view_count),
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
                    tweet_dict["coin_address"] = coin_address
                    tweet_dict["pair_address"] = pair_address

                    # insert the tweet into the tweets table
                    db_utils.insert_tweet(tweet_dict)

                # mark done in completed table
                cursor.execute(
                    "INSERT INTO completed_table (coin_address, pair_address) VALUES (?, ?)",
                    (coin_address, pair_address)
                )
                conn.commit()
                print(f"✅ Marked coin {coin_address} with pair {pair_address} as completed")
                
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
