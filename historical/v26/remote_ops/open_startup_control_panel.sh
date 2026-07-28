#!/usr/bin/env bash
set -u

PROJECT="${PROJECT:-$HOME/PycharmProjects/Ids_Test_3.9/test622yolo_gpu_627}"
PY="${PY:-$HOME/miniconda3/envs/ids_recv/bin/python3}"
PROFILE="${PROFILE:-launch_profiles/627_event_overlay_bev.json}"
DISPLAY="${DISPLAY:-:1}"

export DISPLAY
if [ -z "${XAUTHORITY:-}" ] && [ -f "$HOME/.Xauthority" ]; then
  export XAUTHORITY="$HOME/.Xauthority"
fi
if [ -z "${XDG_RUNTIME_DIR:-}" ] && [ -d "/run/user/$(id -u)" ]; then
  export XDG_RUNTIME_DIR="/run/user/$(id -u)"
fi
if [ -z "${DBUS_SESSION_BUS_ADDRESS:-}" ] && [ -S "${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/bus" ]; then
  export DBUS_SESSION_BUS_ADDRESS="unix:path=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/bus"
fi

cd "$PROJECT" || exit 1
mkdir -p sync_ipc/locks

LOG="sync_ipc/v25_startup_control_panel.log"
LOCK="sync_ipc/locks/v25_startup_control_panel.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
  command -v notify-send >/dev/null 2>&1 && \
    notify-send "元现实游戏引擎" "启动控制面板已经打开" >/dev/null 2>&1 || true
  exit 0
fi

{
  echo "[startup_control_panel] start $(date -Is)"
  echo "[startup_control_panel] project=$PROJECT profile=$PROFILE display=$DISPLAY"
} >> "$LOG"

if [ ! -x "$PY" ]; then
  echo "[startup_control_panel][ERROR] python not executable: $PY" >> "$LOG"
  command -v notify-send >/dev/null 2>&1 && \
    notify-send -u critical "元现实游戏引擎" "启动控制面板失败：Python 环境不可用" >/dev/null 2>&1 || true
  exit 2
fi

if ! "$PY" -c 'import PySide6' >> "$LOG" 2>&1; then
  echo "[startup_control_panel][ERROR] PySide6 unavailable" >> "$LOG"
  command -v notify-send >/dev/null 2>&1 && \
    notify-send -u critical "元现实游戏引擎" "启动控制面板失败：PySide6 不可用" >/dev/null 2>&1 || true
  exit 3
fi

"$PY" remote_ops/v25_replay_control_panel.py \
  --project-root . \
  --profile "$PROFILE" \
  --mode live \
  --replay-timing realtime \
  --gui >> "$LOG" 2>&1
rc="$?"
echo "[startup_control_panel] exit rc=$rc at $(date -Is)" >> "$LOG"
if [ "$rc" -ne 0 ]; then
  command -v notify-send >/dev/null 2>&1 && \
    notify-send -u critical "元现实游戏引擎" "启动控制面板异常退出，请检查 sync_ipc/v25_startup_control_panel.log" >/dev/null 2>&1 || true
fi
exit "$rc"
