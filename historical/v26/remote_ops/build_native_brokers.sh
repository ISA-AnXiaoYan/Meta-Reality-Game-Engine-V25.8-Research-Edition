#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-$HOME/PycharmProjects/Ids_Test_3.9/test622yolo_gpu_627}"
PY="${PY:-$HOME/miniconda3/envs/ids_recv/bin/python3}"
BUILD_DIR="${BUILD_DIR:-native_event_bridge/build}"
LOG="${LOG:-sync_ipc/native_broker_build_latest.log}"
STATUS_JSON="${STATUS_JSON:-sync_ipc/native_broker_build_status.json}"

cd "$PROJECT"
mkdir -p "$(dirname "$LOG")" "$(dirname "$STATUS_JSON")"

start_wall_us="$("$PY" - <<'PY'
import time
print(int(time.time() * 1_000_000))
PY
)"
state="OK"
error=""

{
  echo "[build_native_brokers] start $(date -Is)"
  echo "[build_native_brokers] project=$PROJECT"
  echo "[build_native_brokers] build_dir=$BUILD_DIR"
  echo "[build_native_brokers] normalize shell line endings"
  find remote_ops native_event_bridge -maxdepth 1 -type f -name '*.sh' -exec sed -i 's/\r$//' {} + 2>/dev/null || true
  echo "[build_native_brokers] cmake build"
  bash native_event_bridge/build_native_event_bridge.sh "$BUILD_DIR"
  echo "[build_native_brokers] binary checks"
  for bin in hal_event_fifo_broker event_file_fifo_broker hal_event_pipe_bridge; do
    path="$BUILD_DIR/$bin"
    if [ ! -x "$path" ]; then
      echo "[build_native_brokers][ERROR] missing executable: $path" >&2
      exit 3
    fi
    echo "[build_native_brokers] ok $path"
  done
  "$BUILD_DIR/hal_event_fifo_broker" --help >/tmp/hal_event_fifo_broker_help.txt 2>&1 || true
  "$BUILD_DIR/event_file_fifo_broker" --help >/tmp/event_file_fifo_broker_help.txt 2>&1 || true
  echo "[build_native_brokers] end $(date -Is)"
} >"$LOG" 2>&1 || {
  rc=$?
  state="BROKEN"
  error="build failed rc=$rc"
}

end_wall_us="$("$PY" - <<'PY'
import time
print(int(time.time() * 1_000_000))
PY
)"

"$PY" - "$STATUS_JSON" "$PROJECT" "$BUILD_DIR" "$LOG" "$state" "$error" "$start_wall_us" "$end_wall_us" <<'PY'
import json
import os
import sys
from pathlib import Path

status_path = Path(sys.argv[1])
project = Path(sys.argv[2])
build_dir = Path(sys.argv[3])
log_path = Path(sys.argv[4])
state = sys.argv[5]
error = sys.argv[6]
start_wall_us = int(sys.argv[7])
end_wall_us = int(sys.argv[8])

binaries = {}
for name in ("hal_event_fifo_broker", "event_file_fifo_broker", "hal_event_pipe_bridge"):
    path = project / build_dir / name
    binaries[name] = {
        "path": str(path),
        "exists": path.exists(),
        "executable": os.access(path, os.X_OK),
        "size": path.stat().st_size if path.exists() else 0,
    }

payload = {
    "schema_version": 1,
    "kind": "native_broker_build_status",
    "state": state,
    "error": error,
    "project": str(project),
    "build_dir": str(project / build_dir),
    "log": str(project / log_path),
    "start_wall_us": start_wall_us,
    "end_wall_us": end_wall_us,
    "elapsed_ms": round((end_wall_us - start_wall_us) / 1000.0, 3),
    "binaries": binaries,
}
status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
if state != "OK":
    raise SystemExit(3)
PY

