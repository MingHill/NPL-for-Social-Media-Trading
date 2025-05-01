#!/usr/bin/env python

# This script is used to view items in a Redis queue (list).
import redis
import argparse
import sys

def view_redis_queue(host: str, port: int, queue_name: str):
    """Connects to Redis and prints items from the specified queue."""
    try:
        r = redis.Redis(host=host, port=port, db=0, decode_responses=True)
        r.ping() # Check connection
        print(f"Successfully connected to Redis at {host}:{port}")
    except redis.exceptions.ConnectionError as e:
        print(f"Error connecting to Redis at {host}:{port}: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        queue_length = r.llen(queue_name)
        if queue_length == 0:
            print(f"Queue '{queue_name}' is empty or does not exist.")
            return

        print(f"Items in queue '{queue_name}' ({queue_length} total):")
        # Retrieve all items from the list (0 to -1 includes all elements)
        items = r.lrange(queue_name, 0, -1)

        for i, item in enumerate(items):
            print(f"  {i+1}: {item}")

    except redis.exceptions.ResponseError as e:
        print(f"Error accessing queue '{queue_name}': {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View items in a Redis queue (list).")
    parser.add_argument("--queue_name", default = "solana_signatures", help="Name of the Redis queue (list) to view.")
    parser.add_argument("--host", default="localhost", help="Redis server host (default: localhost)")
    parser.add_argument("--port", type=int, default=6379, help="Redis server port (default: 6379)")

    args = parser.parse_args()

    view_redis_queue(args.host, args.port, args.queue_name)
