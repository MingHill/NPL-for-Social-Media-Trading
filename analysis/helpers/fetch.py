import asyncio
import json
from solana.rpc.async_api import AsyncClient
from solders.signature import Signature

# RPC HTTP endpoint
rpc_url = "https://rpc.ankr.com/solana/2d5c9345dd12734162edf47a0f035e7c4640e1b3403248fd720ec49a290ea24e"

# Transaction signature to fetch details for
signature = "37euL4zavUmAqcTm1QxR9xcK92gZErmHADdeBD2C4ngw11YNPJeXNqzDDrVUGMG7jkQ8rwVZUCQToDQ1QCvZy1kp"

async def fetch_transaction_details():
    async with AsyncClient(rpc_url) as client:
        sig_obj = Signature.from_string(signature)
        resp = await client.get_transaction(
            sig_obj,
            encoding="jsonParsed",
            commitment="confirmed",
            max_supported_transaction_version=0
        )
        if resp.value is None:
            print(f"No transaction found for signature: {signature}")
        else:
            with open("transaction.json", "w") as f:
                f.write(resp.to_json())

if __name__ == "__main__":
    asyncio.run(fetch_transaction_details())