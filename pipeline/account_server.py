# server.py
import asyncio, json, logging, struct
from asyncio import StreamReader, StreamWriter
from pathlib import Path
from accounts.model import Account


CMD_GET = "get"
CMD_RETURN = "return"
HEADER = "!I"                      # 4‑byte unsigned length prefix (network byte order)

class AccountPool:
    def __init__(self, dir_: str = "accounts") -> None:
        self._queue: asyncio.Queue[Account] = asyncio.Queue()
        for acc_json in Path(dir_).rglob("account.json"):
            data = json.loads(Path(acc_json).read_text())
            data["path"] = str(acc_json)            # keep absolute path
            self._queue.put_nowait(Account(**data))
        print("Loaded %d accounts", self._queue.qsize())

    async def get(self) -> Account | None:
        if self._queue.empty():
            return None
        return await self._queue.get()

    def put(self, acc: Account) -> None:
        self._queue.put_nowait(acc)

class AccountServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 9999):
        self._pool = AccountPool()
        self._host, self._port = host, port

    async def _handle(self, rd: StreamReader, wr: StreamWriter) -> None:
        try:
            while True:
                raw_hdr = await rd.readexactly(struct.calcsize(HEADER))
                (n,) = struct.unpack(HEADER, raw_hdr)
                body = await rd.readexactly(n)
                req = json.loads(body)
                cmd = req.get("cmd")
                if cmd == CMD_GET:
                    acc = await self._pool.get()
                    payload = acc.to_json() if acc else ""
                elif cmd == CMD_RETURN:
                    acc = Account(**req["payload"])
                    self._pool.put(acc)
                    payload = "ok"
                else:
                    payload = f"error:unknown‑cmd {cmd}"
                resp = payload.encode()
                wr.write(struct.pack(HEADER, len(resp)) + resp)
                await wr.drain()
        except asyncio.IncompleteReadError:
            pass  # client closed
        finally:
            wr.close()
            await wr.wait_closed()

    async def start(self) -> None:
        srv = await asyncio.start_server(self._handle, self._host, self._port)
        print("Listening on %s:%d", self._host, self._port)
        async with srv:
            await srv.serve_forever()

if __name__ == "__main__":
    try:
        asyncio.run(AccountServer().start())
    except KeyboardInterrupt:
        print("Shutting down…")
