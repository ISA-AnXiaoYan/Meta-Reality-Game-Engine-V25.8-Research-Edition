#!/usr/bin/env bash
set -u
shopt -s nullglob

SYNC="sync_ipc"
PACK_TS="$(date +%Y%m%d_%H%M%S)"
PACK_NAME="hit_debug_pack_${PACK_TS}.tar.gz"

echo "========== [1/7] 检查当前目录 =========="
pwd
if [ ! -f "launch_fusion_system.py" ]; then
  echo "[ERROR] 当前目录没有 launch_fusion_system.py，请确认你在项目根目录。"
  exit 1
fi
if [ ! -f "ids_8cam_fusion_config.json" ]; then
  echo "[ERROR] 当前目录没有 ids_8cam_fusion_config.json。"
  exit 1
fi
mkdir -p "$SYNC"

echo
echo "========== [2/7] 等待 sync_start.flag =========="
for i in $(seq 1 120); do
  if [ -f "$SYNC/sync_start.flag" ]; then
    echo "[OK] sync_start.flag 已出现："
    ls -lh "$SYNC/sync_start.flag"
    break
  fi
  if [ "$i" -eq 120 ]; then
    echo "[ERROR] 等了 120 秒还没有 sync_start.flag。"
    echo "请检查第一个终端里的 launcher 是否报错，尤其是 EVENT / TRIGGER / HIT 是否退出。"
    exit 1
  fi
  echo "等待 sync_start.flag ... ${i}s"
  sleep 1
done

echo
echo "========== [3/7] 检查 trigger CSV 是否增长 =========="
sleep 2

trigger_files=("$SYNC"/event_trigger_*.csv)
if [ ${#trigger_files[@]} -eq 0 ]; then
  echo "[ERROR] 没有找到 event_trigger_*.csv。"
  echo "说明 trigger 可能没有开始写，或者 sync_ipc 路径不对。"
  exit 1
fi

echo "[INFO] 第一次统计："
wc -l "$SYNC"/event_trigger_*.csv || true
SUM1=$(wc -l "$SYNC"/event_trigger_*.csv 2>/dev/null | tail -n 1 | awk '{print $1}')

sleep 5

echo "[INFO] 第二次统计："
wc -l "$SYNC"/event_trigger_*.csv || true
SUM2=$(wc -l "$SYNC"/event_trigger_*.csv 2>/dev/null | tail -n 1 | awk '{print $1}')

echo "[INFO] trigger 行数：${SUM1} -> ${SUM2}"

if [ -z "${SUM1:-}" ] || [ -z "${SUM2:-}" ] || [ "$SUM2" -le "$SUM1" ]; then
  echo "[WARN] trigger CSV 行数没有增长。"
  echo "这通常说明 trigger 没跑起来，或者程序卡住。你仍然可以继续测试，但结果可能没有意义。"
else
  echo "[OK] trigger CSV 正在增长。"
fi

echo
echo "========== [4/7] 检查 human_result 是否写入 =========="
sleep 2

human_files=("$SYNC"/human_result_*.jsonl)
if [ ${#human_files[@]} -eq 0 ]; then
  echo "[WARN] 没有找到 human_result_*.jsonl。"
else
  ls -lh "$SYNC"/human_result_*.jsonl 2>/dev/null || true
  echo
  wc -l "$SYNC"/human_result_*.jsonl 2>/dev/null || true
fi

echo
echo "========== [5/7] 检查 HIT 日志 =========="
echo "[INFO] launcher_logs 文件："
ls -lh "$SYNC"/launcher_logs 2>/dev/null || true

echo
echo "[INFO] HIT 相关日志末尾："
for f in "$SYNC"/launcher_logs/*HIT*.log "$SYNC"/launcher_logs/*hit*.log; do
  if [ -f "$f" ]; then
    echo
    echo "----- $f -----"
    tail -n 80 "$f"
  fi
done

echo
echo "========== [6/7] 现在开始打枪测试 =========="
echo "请现在开始测试："
echo "  1）打第 1 枪"
echo "  2）等 5 秒"
echo "  3）打第 2 枪"
echo "  4）等 5 秒"
echo "  5）打第 3 枪"
echo "  6）等 5 秒"
echo "  7）打第 4 枪"
echo "  8）等 5 秒"
echo "  9）打第 5 枪"
echo
read -p "打完 5 枪后，按 Enter 继续统计并打包日志..."

echo
echo "========== [7/7] 统计事件与打包 =========="
echo
echo "[INFO] 子弹事件 / HIT debug / HIT candidate 文件："
ls -lh \
  "$SYNC"/overlay_bullet_event*.jsonl \
  "$SYNC"/overlay_bullet_point*.jsonl \
  "$SYNC"/hit_judge_debug*.jsonl \
  "$SYNC"/hit_candidate*.jsonl \
  2>/dev/null || true

echo
echo "[INFO] 行数统计："
wc -l "$SYNC"/overlay_bullet_event*.jsonl 2>/dev/null || true
wc -l "$SYNC"/overlay_bullet_point*.jsonl 2>/dev/null || true
wc -l "$SYNC"/hit_judge_debug*.jsonl 2>/dev/null || true
wc -l "$SYNC"/hit_candidate*.jsonl 2>/dev/null || true

echo
echo "[INFO] /dev/shm MVS latest："
ls -lh /dev/shm/mvs_latest_* > "$SYNC/dev_shm_mvs_latest_ls.txt" 2>/dev/null || true
cat "$SYNC/dev_shm_mvs_latest_ls.txt" 2>/dev/null || true

echo
echo "[INFO] 开始打包：$PACK_NAME"

tar -czf "$PACK_NAME" \
  ids_8cam_fusion_config.json \
  "$SYNC"/_runtime_config \
  "$SYNC"/launcher_logs \
  "$SYNC"/launcher_run_*.log \
  "$SYNC"/hit_judge_debug*.jsonl \
  "$SYNC"/hit_candidate*.jsonl \
  "$SYNC"/overlay_bullet_event*.jsonl \
  "$SYNC"/overlay_bullet_point*.jsonl \
  "$SYNC"/event_trigger_*.csv \
  "$SYNC"/human_result_*.jsonl \
  "$SYNC"/mvs_latest_*.json \
  "$SYNC"/mvs_frame_audit_*.csv \
  "$SYNC"/frame_bundle_audit_*.csv \
  "$SYNC"/event_human_mask_stats_*.jsonl \
  "$SYNC"/dev_shm_mvs_latest_ls.txt \
  2>/dev/null

echo
echo "[DONE] 已生成："
ls -lh "$PACK_NAME"

echo
echo "现在可以回到第一个终端按 Ctrl+C 停止程序。"
echo "然后把这个压缩包发给我：$PACK_NAME"
