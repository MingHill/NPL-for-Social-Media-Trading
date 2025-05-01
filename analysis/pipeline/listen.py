import asyncio
import json
from typing import Set
import pandas as pd
import websockets
from tabulate import tabulate
from solana.rpc.async_api import AsyncClient
from solders.pubkey import Pubkey
from solders.signature import Signature
import argparse
from typing import List

# Redis queue imports and configuration
import redis
import yaml
from pathlib import Path
import time
# Load Redis table name from config.yaml
config_path = Path(__file__).parent / "config.yaml"
config = yaml.safe_load(config_path.read_text()) or {}
redis_table = config.get("solana", {}).get("redis_table")
redis_client = redis.Redis()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--tags", type=List[str], required=True)
    parser.add_argument("--rpc_url", type=str, required=True, default="https://rpc.ankr.com/solana/2d5c9345dd12734162edf47a0f035e7c4640e1b3403248fd720ec49a290ea24e")
    parser.add_argument("--wss_url", type=str, required=True, default="wss://rpc.ankr.com/solana/ws/2d5c9345dd12734162edf47a0f035e7c4640e1b3403248fd720ec49a290ea24e")
    return parser.parse_args()

async def listen(args) -> None:
    seen_sigs: Set[str] = set()
    pid = Pubkey.from_string(args.id)

    async with (
        AsyncClient(args.rpc_url) as http,
        websockets.connect(args.wss_url) as ws,
    ):
        # subscribe to program logs
        await ws.send(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "logsSubscribe",
                    "params": [
                        {"mentions": [str(  pid)]},
                        {"commitment": "confirmed"},
                    ],
                }
            )
        )
        ack = json.loads(await ws.recv())
        print("Subscription set up. ID:", ack.get("result"))

        # stream log notifications
        async for raw in ws:
            msg = json.loads(raw)

            if msg.get("method") != "logsNotification":
                continue
            value = msg["params"]["result"]["value"]
            if value["err"] is not None:
                continue
            sig = value["signature"]
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)
            
            logs_string = " ".join(value["logs"])
            if all(tag in logs_string for tag in args.tags):
                # Push signature and program name to Redis queue
                # timestamp int
                block_time = msg["params"]["result"]["context"]["slot"]
                # get block time from rpc
                response = await http.get_block_time(block_time)
                timestamp = response.value
                redis_client.rpush(redis_table, f"{sig},{args.name},{timestamp}")

if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(listen(args))
    except KeyboardInterrupt:
        print("\nListener stopped by user.")
