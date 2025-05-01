# client.py
import asyncio, json, struct
from accounts.model import Account

HEADER = "!I"

async def _send_recv(cmd: str, payload: dict | None = None) -> str:
    reader, writer = await asyncio.open_connection("127.0.0.1", 9999)
    req = json.dumps({"v": 1, "cmd": cmd, "payload": payload or {}}).encode()
    writer.write(struct.pack(HEADER, len(req)) + req)
    await writer.drain()

    raw_hdr = await reader.readexactly(struct.calcsize(HEADER))
    (n,) = struct.unpack(HEADER, raw_hdr)
    data = await reader.readexactly(n)
    writer.close(); await writer.wait_closed()
    return data.decode()

async def get_account() -> Account | None:
    txt = await _send_recv("get")
    return Account(**json.loads(txt)) if txt else None

async def return_account(acc: Account) -> None:
    await _send_recv("return", json.loads(acc.to_json()))

if __name__ == "__main__":
    acc = asyncio.run(get_account())
    print(acc)
    asyncio.run(return_account(acc))
