import sys
import asyncio
from pathlib import Path
from types import SimpleNamespace
import yaml
from listen import listen
import websockets                                   


async def run_listener(args):
    """
    Call listen(args) forever.
    If the Web‑Socket drops (or any other error bubbles up),
    wait five seconds and reconnect.
    """
    while True:
        try:
            await listen(args)                      # original coroutine
        except (websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.ConnectionClosedOK) as e:
            print(f"[{args.name}] connection closed: {e!s} – reconnecting in 5 s")
            await asyncio.sleep(5)
        except Exception as e:
            # Catch‐all so that *any* unexpected error doesn’t kill the whole app
            print(f"[{args.name}] unexpected error: {e!s} – reconnecting in 5 s")
            await asyncio.sleep(5)

async def main():
    # Locate and read configuration file
    config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        return
    config = yaml.safe_load(config_path.read_text()) or {}
    sol_cfg: dict = config.get("solana", {})
    rpc_url: str = sol_cfg.get("rpc_http")
    wss_url: str = sol_cfg.get("rpc_wss")
    programs: dict = sol_cfg.get("programs", {})

    if not programs:
        print("No programs defined in config.yaml under 'solana.programs'.", file=sys.stderr)
        return

    # Create and start listener tasks for each program
    tasks = []
    for key in programs.keys():
        prog: dict = programs.get(key)
        pid = prog.get("id")
        tags = prog.get("tags")
        name = prog.get("name", key)
        if not pid or not tags:
            print(f"Skipping program '{key}': missing 'id' or 'tags'.", file=sys.stderr)
            continue
        # Prepare arguments for listen
        args = SimpleNamespace(
            id=pid,
            name=name,
            tags=tags,
            rpc_url=rpc_url,
            wss_url=wss_url,
        )
        print(f"Starting listener for program '{name}' (ID: {pid}) with tags '{tags}'")
        tasks.append(asyncio.create_task(run_listener(args)))

    # Run all listeners concurrently
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("All listeners stopped by user.")