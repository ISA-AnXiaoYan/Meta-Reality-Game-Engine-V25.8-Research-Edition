#!/usr/bin/env bash
set -u

SYNC="sync_ipc"

echo "========== 等待 sync_start.flag =========="
for i in $(seq 1 120); do
  if [ -f "$SYNC/sync_start.flag" ]; then
    echo "[OK] sync_start.flag 已出现"
    break
  fi
  echo "等待同步启动 ${i}s ..."
  sleep 1
done

if [ ! -f "$SYNC/sync_start.flag" ]; then
  echo "[ERROR] 没有等到 sync_start.flag"
  exit 1
fi

echo
echo "========== 等待 event_trigger_D.csv =========="
for i in $(seq 1 30); do
  if [ -f "$SYNC/event_trigger_D.csv" ]; then
    echo "[OK] event_trigger_D.csv 已出现"
    break
  fi
  echo "等待 event_trigger_D.csv ${i}s ..."
  sleep 1
done

if [ ! -f "$SYNC/event_trigger_D.csv" ]; then
  echo "[ERROR] 没有找到 event_trigger_D.csv"
  exit 1
fi

echo
echo "========== 统计 5 秒增长 =========="
L1=$(wc -l "$SYNC/event_trigger_D.csv" | awk '{print $1}')
T1=$(date +%s.%N)

sleep 5

L2=$(wc -l "$SYNC/event_trigger_D.csv" | awk '{print $1}')
T2=$(date +%s.%N)

DIFF=$((L2 - L1))

echo "5 秒前行数: $L1"
echo "5 秒后行数: $L2"
echo "5 秒增长:   $DIFF 行"

echo
echo "判断："
if [ "$DIFF" -lt 150 ]; then
  echo "[异常] trigger 行数太少，可能 trigger 没跑稳。"
elif [ "$DIFF" -le 260 ]; then
  echo "[正常] 接近 40Hz。5 秒理论约 200 行。"
else
  echo "[异常] 明显高于 40Hz。不是旧文件堆积，而是当前运行中 trigger 被重复计数。"
  echo "       如果接近 3000 多行/5秒，就是大约 600Hz，和上次数据一致。"
fi

echo
echo "========== 同时检查 MVS 帧增长 =========="
if [ -f "$SYNC/mvs_frame_audit_D.csv" ]; then
  M1=$(wc -l "$SYNC/mvs_frame_audit_D.csv" | awk '{print $1}')
  sleep 5
  M2=$(wc -l "$SYNC/mvs_frame_audit_D.csv" | awk '{print $1}')
  echo "MVS 5 秒增长: $((M2 - M1)) 行"
  echo "理论也应该接近 200 行。"
else
  echo "[WARN] 没有找到 mvs_frame_audit_D.csv"
fi

echo
echo "========== 当前关键文件行数 =========="
wc -l "$SYNC/event_trigger_D.csv" 2>/dev/null || true
wc -l "$SYNC/human_result_D.jsonl" 2>/dev/null || true
wc -l "$SYNC/mvs_frame_audit_D.csv" 2>/dev/null || true
