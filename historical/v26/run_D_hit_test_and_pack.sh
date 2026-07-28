#!/usr/bin/env bash
set -u
shopt -s nullglob

SYNC="sync_ipc"
TS="$(date +%Y%m%d_%H%M%S)"
PACK_NAME="hit_debug_pack_D_${TS}.tar.gz"

echo "========== [1/8] 检查目录 =========="
pwd

if [ ! -f "launch_fusion_system.py" ]; then
  echo "[ERROR] 当前目录没有 launch_fusion_system.py"
  exit 1
fi

if [ ! -f "ids_8cam_fusion_config.json" ]; then
  echo "[ERROR] 当前目录没有 ids_8cam_fusion_config.json"
  exit 1
fi

echo
echo "========== [2/8] 等待 sync_start.flag =========="
for i in $(seq 1 120); do
  if [ -f "$SYNC/sync_start.flag" ]; then
    echo "[OK] sync_start.flag 已出现"
    ls -lh "$SYNC/sync_start.flag"
    break
  fi

  if [ "$i" -eq 120 ]; then
    echo "[ERROR] 120 秒内没有等到 sync_start.flag"
    echo "请检查第一个终端是否有 EVENT / TRIGGER / HIT 退出。"
    exit 1
  fi

  echo "等待同步启动 ... ${i}s"
  sleep 1
done

echo
echo "========== [3/8] 检查 D 路 trigger 是否增长 =========="
sleep 2

if [ ! -f "$SYNC/event_trigger_D.csv" ]; then
  echo "[ERROR] 没有找到 $SYNC/event_trigger_D.csv"
  exit 1
fi

L1=$(wc -l "$SYNC/event_trigger_D.csv" | awk '{print $1}')
echo "[INFO] 第一次行数：$L1"

sleep 5

L2=$(wc -l "$SYNC/event_trigger_D.csv" | awk '{print $1}')
echo "[INFO] 第二次行数：$L2"

if [ "$L2" -le "$L1" ]; then
  echo "[WARN] event_trigger_D.csv 行数没有增长，trigger 可能异常。"
else
  echo "[OK] D 路 trigger 正在增长。"
fi

echo
echo "========== [4/8] 检查 D 路人体检测 =========="
sleep 2

if [ -f "$SYNC/human_result_D.jsonl" ]; then
  ls -lh "$SYNC/human_result_D.jsonl"
  echo "[INFO] human_result_D 行数："
  wc -l "$SYNC/human_result_D.jsonl"
  echo
  echo "[INFO] human_result_D 最后 3 行："
  tail -n 3 "$SYNC/human_result_D.jsonl"
else
  echo "[WARN] 没有找到 human_result_D.jsonl"
fi

echo
echo "========== [5/8] 检查 HIT 日志 =========="
echo "[INFO] launcher_logs："
ls -lh "$SYNC/launcher_logs" 2>/dev/null || true

echo
echo "[INFO] HIT 日志末尾："
for f in "$SYNC"/launcher_logs/*HIT*.log "$SYNC"/launcher_logs/*hit*.log; do
  if [ -f "$f" ]; then
    echo
    echo "----- $f -----"
    tail -n 100 "$f"
  fi
done

echo
echo "========== [6/8] 请现在开始 D 路打枪测试 =========="
echo
echo "测试要求："
echo "  1. 人物只站在 D 路画面里。"
echo "  2. 确认 D 路画面里人体完整可见。"
echo "  3. 每枪间隔 5 秒以上。"
echo "  4. 这次建议只打 5 枪，全部明确打到人体。"
echo
echo "推荐节奏："
echo "  第 1 枪：打人体中心区域"
echo "  等 5 秒"
echo "  第 2 枪：打人体中心区域"
echo "  等 5 秒"
echo "  第 3 枪：打人体中心区域"
echo "  等 5 秒"
echo "  第 4 枪：打人体中心区域"
echo "  等 5 秒"
echo "  第 5 枪：打人体中心区域"
echo
read -p "打完 5 枪后，按 Enter 继续统计和打包..."

echo
echo "========== [7/8] 统计 D 路事件 =========="

echo
echo "[INFO] D 路关键文件："
ls -lh \
  "$SYNC"/event_trigger_D.csv \
  "$SYNC"/human_result_D.jsonl \
  "$SYNC"/overlay_bullet_event_D.jsonl \
  "$SYNC"/overlay_bullet_point_D.jsonl \
  "$SYNC"/hit_judge_debug_D.jsonl \
  "$SYNC"/hit_candidate_D.jsonl \
  "$SYNC"/hit_judge_debug_all.jsonl \
  "$SYNC"/hit_candidate_all.jsonl \
  2>/dev/null || true

echo
echo "[INFO] D 路行数统计："
wc -l "$SYNC"/event_trigger_D.csv 2>/dev/null || true
wc -l "$SYNC"/human_result_D.jsonl 2>/dev/null || true
wc -l "$SYNC"/overlay_bullet_event_D.jsonl 2>/dev/null || true
wc -l "$SYNC"/overlay_bullet_point_D.jsonl 2>/dev/null || true
wc -l "$SYNC"/hit_judge_debug_D.jsonl 2>/dev/null || true
wc -l "$SYNC"/hit_candidate_D.jsonl 2>/dev/null || true
wc -l "$SYNC"/hit_judge_debug_all.jsonl 2>/dev/null || true
wc -l "$SYNC"/hit_candidate_all.jsonl 2>/dev/null || true

echo
echo "[INFO] D 路 HIT debug 最后 20 行："
tail -n 20 "$SYNC"/hit_judge_debug_D.jsonl 2>/dev/null || true

echo
echo "[INFO] D 路 hit_candidate 内容："
cat "$SYNC"/hit_candidate_D.jsonl 2>/dev/null || true

echo
echo "========== [8/8] 打包 D 路数据 =========="

ls -lh /dev/shm/mvs_latest_* > "$SYNC/dev_shm_mvs_latest_ls.txt" 2>/dev/null || true

tar -czf "$PACK_NAME" \
  ids_8cam_fusion_config.json \
  "$SYNC"/_runtime_config \
  "$SYNC"/launcher_logs \
  "$SYNC"/launcher_run_D_*.log \
  "$SYNC"/event_trigger_D.csv \
  "$SYNC"/human_result_D.jsonl \
  "$SYNC"/overlay_bullet_event_D.jsonl \
  "$SYNC"/overlay_bullet_point_D.jsonl \
  "$SYNC"/hit_judge_debug_D.jsonl \
  "$SYNC"/hit_candidate_D.jsonl \
  "$SYNC"/hit_judge_debug_all.jsonl \
  "$SYNC"/hit_candidate_all.jsonl \
  "$SYNC"/mvs_latest_D.json \
  "$SYNC"/mvs_frame_audit_D.csv \
  "$SYNC"/frame_bundle_audit_D.csv \
  "$SYNC"/event_human_mask_stats_D.jsonl \
  "$SYNC"/dev_shm_mvs_latest_ls.txt \
  2>/dev/null

echo
echo "[DONE] 已生成："
ls -lh "$PACK_NAME"

echo
echo "现在可以回到第一个终端按 Ctrl+C 停止程序。"
echo "然后把这个压缩包发给我：$PACK_NAME"
