#!/usr/bin/env bash
set -u
cd ~/PycharmProjects/Ids_Test_3.9/test607 || exit 1

OUT="/tmp/ysxq_E_hit_raw_$(date +%Y%m%d_%H%M%S).txt"

{
  echo "========== TIME =========="
  date

  echo
  echo "========== VERIFY E =========="
  python3 ysxq_verify_hit_sync.py \
    --config hik_rgb_yolo_seg_multicam_same_model_stableid_v5.json \
    --root ./sync_ipc \
    --cams E \
    --strict

  echo
  echo "========== E HIT DEBUG RAW LAST 120 =========="
  tail -n 120 sync_ipc/hit_judge_debug_E.jsonl 2>/dev/null || echo "NO hit_judge_debug_E.jsonl"

  echo
  echo "========== E HIT CANDIDATE RAW LAST 120 =========="
  tail -n 120 sync_ipc/hit_candidate_E.jsonl 2>/dev/null || echo "NO hit_candidate_E.jsonl"

  echo
  echo "========== E BULLET EFFECT RAW LAST 160 =========="
  tail -n 160 sync_ipc/bullet_effect_E.csv 2>/dev/null || echo "NO bullet_effect_E.csv"

  echo
  echo "========== E SUMMARY FROM LAST DEBUG/CANDIDATE =========="
  python3 - <<'PY'
import json
from pathlib import Path
from collections import Counter, defaultdict

def load_jsonl(path):
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows

debug = load_jsonl("sync_ipc/hit_judge_debug_E.jsonl")[-120:]
cand = load_jsonl("sync_ipc/hit_candidate_E.jsonl")[-120:]

print("debug_rows_last120 =", len(debug))
print("candidate_rows_last120 =", len(cand))
print("debug_status =", dict(Counter(r.get("status") for r in debug)))
print("debug_reject =", dict(Counter(r.get("reject_reason") for r in debug)))

debug_bullets = [r.get("bullet_id") for r in debug if r.get("camera") == "E"]
cand_bullets = [r.get("bullet_id") for r in cand if (r.get("camera") == "E" or r.get("camera") is None)]

print("debug_unique_bullet_ids =", sorted(set(debug_bullets), key=lambda x: str(x)))
print("candidate_unique_bullet_ids =", sorted(set(cand_bullets), key=lambda x: str(x)))
print("candidate_unique_count =", len(set(cand_bullets)))

print()
print("---- DEBUG TABLE LAST 40 ----")
for r in debug[-40:]:
    print(
        "bullet=", r.get("bullet_id"),
        "status=", r.get("status"),
        "reject=", r.get("reject_reason"),
        "sync=", r.get("impact_sync_index"),
        "delta=", r.get("person_frame_delta_sync"),
        "fallback=", r.get("human_fallback_used"),
        "event_xy=", r.get("event_xy"),
        "mvs=", r.get("event_point_mvs"),
    )

print()
print("---- CANDIDATE TABLE LAST 40 ----")
for r in cand[-40:]:
    print(
        "bullet=", r.get("bullet_id"),
        "score=", r.get("score") or r.get("total_score"),
        "mode=", r.get("geometry_mode") or r.get("match_method") or r.get("geometry_best_method"),
        "sync=", r.get("impact_sync_index"),
        "event_xy=", r.get("event_xy"),
        "mvs=", r.get("event_point_mvs") or r.get("event_point"),
        "person=", r.get("person_id") or r.get("stable_id"),
    )
PY
} | tee "$OUT"

echo
echo "saved: $OUT"
