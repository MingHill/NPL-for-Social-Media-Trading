import yaml
from pathlib import Path
import requests
import pandas as pd
from typing import List
import db_utils
import sys
import time
from tqdm import tqdm
config_path = Path(__file__).parent / "config.yaml"
config = yaml.safe_load(config_path.read_text()) or {}
base_url = "https://api.geckoterminal.com/api/v2"
AGGREGATES = [1, 5]
RATE_LIMIT_PER_MINUTE = 30
WAIT_TIME_SECONDS = (60 / RATE_LIMIT_PER_MINUTE) + 2

def build_ohlcv_url(pair_address: str, network: str = "solana", timeframe: str = "minute", aggregate: int = 1, limit: int = 1000, before_timestamp: int = None) -> str:
    return_url = f"{base_url}/networks/{network}/pools/{pair_address}/ohlcv/{timeframe}?aggregate={aggregate}&limit={limit}"
    if before_timestamp:
        return_url += f"&before_timestamp={before_timestamp}"
    return return_url

def get_ohlcv_data(pair_address: str, limit: int = 1000, aggregate: int = 1, before_timestamp: int = None) -> List[List]:
    ohlcv_list = []
    while True:
        url = build_ohlcv_url(pair_address, before_timestamp=before_timestamp, limit=limit, aggregate=aggregate)
        response = requests.get(url)
        time.sleep(WAIT_TIME_SECONDS)
        response.raise_for_status()
        extension = response.json()["data"]["attributes"]["ohlcv_list"]
        ohlcv_list.extend(extension)
        if len(extension) < limit:
            # no more data to fetch
            break
        before_timestamp = extension[-1][0]
    return ohlcv_list

def resolve_ath(row):
    pair_address = row["pair_address"]
    token_release_timestamp = int(row["creation_timestamp"])
    rugpull_timestamp = row["rugpull_timestamp"]
    if pd.isna(rugpull_timestamp) or rugpull_timestamp is None:
        rugpull_timestamp = int(time.time())
    else:
        rugpull_timestamp = int(rugpull_timestamp)
    found = False
    for aggregate in AGGREGATES:
        try:
            ohlcv_data = get_ohlcv_data(pair_address, limit=1000, aggregate=aggregate, before_timestamp=rugpull_timestamp)
        except Exception as e:
            print(f"Error fetching OHLCV data for {pair_address} with aggregate {aggregate}: {e}")
            return None, None, False
        # get the last timestamp
        if len(ohlcv_data) == 0:
            continue
        last_timestamp = ohlcv_data[-1][0]
        # Get the difference between the last timestamp, and the token release date.
        time_diff = last_timestamp - token_release_timestamp
        max_time_diff = aggregate * 60
        if time_diff < max_time_diff:
            found = True
            break
    if found:
        # original closing price
        original_closing_price = ohlcv_data[-1][4]
        max_closing_price = original_closing_price
        max_timestamp = last_timestamp
        # find the highest closing price
        for ohlcv in ohlcv_data:
            if ohlcv[4] > max_closing_price:
                max_closing_price = ohlcv[4]
                max_timestamp = ohlcv[0]
        # calculate the time delta
        ath_time_delta = max_timestamp - token_release_timestamp
        ath_price_gain = ((max_closing_price - original_closing_price) / original_closing_price) * 100
        return ath_time_delta, ath_price_gain, False
    else:
        return None, None, True

def main_old():
    # For each coin that has been rugged, and does not have ATH data.
    conn = db_utils.get_connection()
    cursor = conn.cursor()
    try:
        while True:
            cursor.execute(
                """
                SELECT *
                FROM coin_table
                WHERE rugpull_flag = 1 AND ath_time_delta IS NULL
                """
            )
            rows = cursor.fetchall()
            for row in rows:
                ath_time_delta, ath_price_gain, delete_flag = resolve_ath(row)
                if delete_flag:
                    print(f"Deleting {row['pair_address']} because it has no OHLCV data.")
                    #sys.exit()
                    cursor.execute(
                        """
                        DELETE FROM coin_table WHERE pair_address = ?
                        """, (row['pair_address'],)
                    )
                    # check if the coin exists in the completed_table
                    cursor.execute(
                        """
                        SELECT * FROM completed_table WHERE pair_address = ?
                        """, (row['pair_address'],)
                    )
                    if cursor.fetchone():
                        cursor.execute(
                            """
                            DELETE FROM completed_table WHERE pair_address = ?
                            """, (row['pair_address'],)
                        )
                        # delete from the tweets table
                        cursor.execute(
                            """
                            DELETE FROM tweets_table WHERE pair_address = ?
                            """, (row['pair_address'],)
                        )
                    conn.commit()
                elif ath_time_delta is not None:
                    print(f"Updating {row['pair_address']} with ATH data.")
                    cursor.execute(
                        """
                        UPDATE coin_table SET ath_time_delta = ?, ath_price_gain = ? WHERE pair_address = ?
                        """, (ath_time_delta, ath_price_gain, row['pair_address'])
                    )
                    conn.commit()
    except KeyboardInterrupt:
        print("Keyboard interrupt. Exiting...")
        sys.exit()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cursor.close()
        conn.close()

def main():
    try:
        while True:
            coins_df = pd.read_csv('data/post/main_3.csv')
            for index, row in tqdm(coins_df.iterrows(), total=len(coins_df), desc="Processing coins"):
                coin_address = row['coin_address']
                ath_time_delta = row['ath_time_delta']
                if pd.notna(ath_time_delta):
                    continue
                ath_time_delta, ath_price_gain, delete_flag = resolve_ath(row)
                if delete_flag:
                    continue
                else:
                    # update the coins_df 
                    coins_df.at[index, 'ath_time_delta'] = ath_time_delta
                    coins_df.at[index, 'ath_price_gain'] = ath_price_gain
                coins_df.to_csv('data/post/main_3.csv', index=False)
            
    except KeyboardInterrupt:
        print("Keyboard interrupt. Exiting...")
        sys.exit()
    except Exception as e:
        print(f"Error: {e}")
    

if __name__ == "__main__":
    main()
