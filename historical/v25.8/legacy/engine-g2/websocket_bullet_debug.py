import asyncio
import json
import os
import websockets

PAIRS = ["A", "B", "C", "D", "E", "F", "G", "H"]
SYNC_DIR = "sync_ipc"
HOST = "0.0.0.0"
PORT = 9876

clients = set()

def path_for(pair):
    return os.path.join(SYNC_DIR, f"overlay_bullet_point_{pair}.jsonl")

async def send_to_all(msg: str, pair: str, obj: dict):
    if not clients:
        print(
            f"[multi_jsonl_ws] DROP no_clients pair={pair} bullet={obj.get('bullet_id')} "
            f"point={obj.get('point_index')} x={obj.get('x')} y={obj.get('y')}",
            flush=True
        )
        return

    dead = []
    ok = 0
    for ws in list(clients):
        try:
            await ws.send(msg)
            ok += 1
        except Exception as e:
            print(f"[multi_jsonl_ws] send_error pair={pair}: {e}", flush=True)
            dead.append(ws)

    for ws in dead:
        clients.discard(ws)

    print(
        f"[multi_jsonl_ws] SENT clients={len(clients)} ok={ok} pair={obj.get('pair_id')} "
        f"bullet={obj.get('bullet_id')} point={obj.get('point_index')} "
        f"x={obj.get('x')} y={obj.get('y')}",
        flush=True
    )

async def handler(ws):
    clients.add(ws)
    peer = getattr(ws, "remote_address", None)
    print(f"[multi_jsonl_ws] client connected peer={peer}; clients={len(clients)}", flush=True)
    try:
        await ws.wait_closed()
    finally:
        clients.discard(ws)
        print(f"[multi_jsonl_ws] client disconnected; clients={len(clients)}", flush=True)

async def tail_pair(pair: str):
    path = path_for(pair)
    print(f"[multi_jsonl_ws] tail pair={pair} path={path}", flush=True)

    while not os.path.exists(path):
        print(f"[multi_jsonl_ws] waiting file: {path}", flush=True)
        await asyncio.sleep(0.5)

    with open(path, "r", encoding="utf-8") as f:
        f.seek(0, os.SEEK_END)

        while True:
            line = f.readline()
            if not line:
                await asyncio.sleep(0.01)
                continue

            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                if obj.get("type") != "overlay_bullet_point":
                    continue

                obj.setdefault("pair_id", pair)
                obj.setdefault("camera_alias", pair)

                msg_pair = str(obj.get("pair_id", ""))
                if msg_pair not in PAIRS:
                    print(f"[multi_jsonl_ws] skip bad_pair={msg_pair}", flush=True)
                    continue

                msg = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
            except Exception as e:
                print(f"[multi_jsonl_ws] bad line pair={pair}: {e}", flush=True)
                continue

            await send_to_all(msg, pair, obj)

async def main():
    # ping_interval=None：先禁用 Python websockets 的 ping，避免 UE 插件因为 ping/pong 兼容性立刻断开。
    async with websockets.serve(
        handler,
        HOST,
        PORT,
        max_size=2_000_000,
        ping_interval=None,
    ):
        print(f"[multi_jsonl_ws] listen ws://{HOST}:{PORT} pairs={PAIRS}", flush=True)
        await asyncio.gather(*(tail_pair(pair) for pair in PAIRS))

asyncio.run(main())
