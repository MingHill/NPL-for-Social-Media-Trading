import time
import argparse
import yaml
from pathlib import Path
from solana.rpc.api import Client
from solders.pubkey import Pubkey
import json
import db_utils
import os
from tqdm import tqdm
from decode import decode_raydium, decode_pumpfun
from handler import get_coin_meta

config_path = Path(__file__).parent / "config.yaml"
config = yaml.safe_load(config_path.read_text()) or {}
threshold = config["monitor"]["threshold"]
# RPC endpoint
rpc_http = config["solana"]["rpc_http"]
# program IDs to scan for withdraw instructions
amm_program_ids = {p["id"] for p in config.get("solana", {}).get("programs", {}).values()}

def creator_mechanism(coin_address: str, pair_address: str, creator: str, exchange: str) -> bool:
    """
    Mechanism 1: detect if the creator has withdrawn liquidity from the pair.
    Fetch recent transactions for the pair_address, look for any 'withdraw' instruction.
    Returns True if a withdraw instruction is found, False otherwise.
    """

    # initialize RPC client
    client = Client(rpc_http)
    try:
        # fetch signatures involving the pair (liquidity pool)
        resp = client.get_signatures_for_address(Pubkey.from_string(creator), limit=100)
        time.sleep(2)
        sigs = resp.value
    except Exception:
        return False, None

    # scan each transaction for a withdraw instruction
    for info in sigs:
        sig = info.signature
        if not sig:
            continue
        try:
            # fetch parsed transaction details
            tx = client.get_transaction(sig, encoding='jsonParsed', commitment='confirmed', max_supported_transaction_version=0)
            time.sleep(0.25)
            # Save the transaction to a file
            with open('transaction_temp.json', 'w') as f:
                f.write(tx.to_json())
            with open('transaction_temp.json', 'r') as f:
                tx_dict = json.load(f)
        except Exception as e:
            print(f"Error fetching transaction {sig}: {e}")
            # check if the file exists
            if os.path.exists('transaction_temp.json'):
                os.remove('transaction_temp.json')
            continue
        # delete the transaction file
        os.remove('transaction_temp.json')

        if tx_dict is None:
            continue
        if tx_dict["result"] is None:
            continue
       
        # parse the transaction
        instructions = tx_dict["result"]["transaction"]["message"]["instructions"]
        signature = tx_dict["result"]["transaction"]["signatures"][0]
        # if signature == "5NgJ6ToLpyMKXrVZtMMiUngHoa7SkUbMbLA2Cpn1PQ6Xr32huU8FEveNfdxKbnKKYBuDfX9aTap8oKLnjLs1xtB7":
        #     print("Here")
        log_messages = tx_dict["result"]["meta"]["logMessages"]
        log_string = " ".join(log_messages)
        blocktime = tx_dict["result"]["blockTime"]
        # Check #1:
        if "raydium" in exchange.lower():
            for instruction in instructions:
                if 'data' in instruction:
                    data = instruction['data']
                    decoded = decode_raydium(data)
                    if decoded['name'] == 'Withdraw':
                        return True, blocktime
        if "pump" in exchange.lower():
            for instruction in instructions:
                if 'data' in instruction:
                    data = instruction['data']
                    decoded = decode_pumpfun(data)
                    if decoded['name'] == 'withdraw':
                        return True, blocktime
        # Check #2
        if "Withdraw" in log_string:
            for instruction in instructions:
                if "accounts" in instruction:
                    accounts = instruction["accounts"]
                    if coin_address in accounts and pair_address in accounts:
                        return True, blocktime
    return False, None

def check_rugpull():
    """
    Query the database for coins not yet marked as rugpull (flag=0),
    check current liquidity/price against initial values,
    and update the rugpull flag if conditions are met.
    """
    while True:
        conn = db_utils.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT *"
            " FROM coin_table WHERE rugpull_flag = 0"
        )
        rows = cursor.fetchall()
        for row in tqdm(rows):
            sig = row["event_signature"]
            exchange = row["exchange"]
            created_ts = int(row["creation_timestamp"])
            coin_address = row["coin_address"]
            pair_address = row["pair_address"]
            coin_name = row["coin_name"]
            coin_symbol = row["coin_symbol"]
            creator = row["creator"]
            initial_liquidity = row["initial_liquidity"]
            initial_price = row["initial_price"]
            
            # Call mechanism 1
            # Check if the creator has called a withdraw function
            rugpull_flag, rugpull_time = creator_mechanism(coin_address, pair_address, creator, exchange)
            if rugpull_flag and rugpull_time:
                # Update the rugpull flag and time, and time delta
                rugpull_time_delta = rugpull_time - created_ts
                cursor.execute(
                    "UPDATE coin_table SET rugpull_flag = 1, rugpull_timestamp = ?, rugpull_time_delta = ? WHERE event_signature = ?",
                    (rugpull_time, rugpull_time_delta, sig)
                )
                conn.commit()
        conn.close()


if __name__ == "__main__":
    check_rugpull()