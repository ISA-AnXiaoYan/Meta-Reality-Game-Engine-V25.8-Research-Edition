
import asyncio
import json
import os
import time
import websockets

PAIR = os.environ.get("PAIR", "C")
HOST = "0.0.0.0"
PORT = 9877
W = 1624
H = 1240

clients = set()

def make_msg(pair, bullet_id, seq, x, y, px=None, py=None):
    return {
        "type": "overlay_bullet_point",
        "schema_version": 2,
        "sync_run_id": "calib_run",
        "pair_id": pair,
        "camera_alias": pair,
        "bullet_id": bullet_id,
        "bullet_seq_id": seq,
        "point_index": seq,
        "point_ts_us": int(time.time() * 1000000) + seq,
        "x": float(x),
        "y": float(y),
        "display_width": W,
        "display_height": H,
        "confidence": 1.0,
        "status": "tracking",
        "state": "track",
        "semantic_state": "track",
        "prev_x": float(x if px is None else px),
        "prev_y": float(y if py is None else py),
        "prev_valid": px is not None and py is not None,
        "source": "ue5_calibration_pattern",
    }

async def send_all(obj):
    msg = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    dead = []
    for ws in list(clients):
        try:
            await ws.send(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)

async def stroke(pair, bullet_id, points):
    seq = 0
    prev = None
    for x, y in points:
        if prev is None:
            obj = make_msg(pair, bullet_id, seq, x, y)
        else:
            obj = make_msg(pair, bullet_id, seq, x, y, prev[0], prev[1])
        await send_all(obj)
        prev = (x, y)
        seq += 1
        await asyncio.sleep(0.04)

async def pattern_loop():
    print(f"[calib_ws] sending calibration pattern pair={PAIR} ws://{HOST}:{PORT}", flush=True)
    while True:
        # 1. 外边框：应该贴合整张视频画面的边缘
        await stroke(PAIR, 9100, [
            (0, 0), (W, 0), (W, H), (0, H), (0, 0)
        ])

        # 2. 对角线：用于判断方向是否翻转
        await stroke(PAIR, 9101, [
            (0, 0), (W, H)
        ])

        # 3. 左上角 L 形标记：应该出现在画面左上角
        await stroke(PAIR, 9102, [
            (80, 80), (420, 80), (80, 80), (80, 360)
        ])

        await asyncio.sleep(0.6)

async def handler(ws):
    clients.add(ws)
    print(f"[calib_ws] client connected; clients={len(clients)}", flush=True)
    try:
        await ws.wait_closed()
    finally:
        clients.discard(ws)
        print(f"[calib_ws] client disconnected; clients={len(clients)}", flush=True)

async def main():
    async with websockets.serve(handler, HOST, PORT, max_size=2_000_000):
        print(f"[calib_ws] listen ws://{HOST}:{PORT} pair={PAIR}", flush=True)
        await pattern_loop()

asyncio.run(main())