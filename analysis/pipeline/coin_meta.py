import requests
import db_utils
import sys
import time

def get_coin_meta(coin_address: str):
    url = f"https://api.rugcheck.xyz/v1/tokens/{coin_address}/report"
    resp = requests.get(url)
    resp.raise_for_status()
    time.sleep(2)
    data_dict = resp.json()
    try:
        uri = data_dict["tokenMeta"]["uri"]
        resp = requests.get(uri)
        resp.raise_for_status()
        time.sleep(2)
        return resp.json()
    except KeyError:
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching coin meta for {coin_address}: {e}")
        time.sleep(2)
        return None
    except Exception as e:
        print(f"Error fetching coin meta for {coin_address}: {e}")
        time.sleep(2)
        return None

def main():
    db_utils.create_coin_socials_table()
    conn = db_utils.get_connection()
    cursor = conn.cursor()
    # get coin addresses from coin_table that don't have socials
    try:
        while True:
            cursor.execute("""
                SELECT c.coin_address 
                FROM coin_table c
                LEFT JOIN coin_socials_table cs ON c.coin_address = cs.coin_address
                WHERE cs.coin_address IS NULL
            """)
            rows = cursor.fetchall()
            for row in rows:
                coin_address = row["coin_address"]
                coin_meta = get_coin_meta(coin_address)
                if coin_meta is None:
                    continue

                if "extensions" in coin_meta:
                    extensions = coin_meta["extensions"]
                else:
                    continue

                if "twitter" in extensions:
                    twitter = extensions["twitter"]
                else:
                    twitter = None

                if "telegram" in extensions:
                    telegram = extensions["telegram"]
                else:
                    telegram = None
                
                if "website" in extensions:
                    website = extensions["website"]
                else:
                    website = None

                if "createdOn" in coin_meta:
                    createdOn = coin_meta["createdOn"]
                else:
                    createdOn = None

                if "creator" in coin_meta:
                    if "site" in coin_meta["creator"]:
                        createdOn_site = coin_meta["creator"]["site"]
                    else:
                        createdOn_site = None
                else:
                    createdOn_site = None

                cursor.execute("INSERT INTO coin_socials_table (coin_address, twitter, telegram, website, createdOn, createdOn_site) VALUES (?, ?, ?, ?, ?, ?)", (coin_address, twitter, telegram, website, createdOn, createdOn_site))
                conn.commit()
    except KeyboardInterrupt:
        print("Keyboard interrupt. Exiting...")
        sys.exit()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    main()
