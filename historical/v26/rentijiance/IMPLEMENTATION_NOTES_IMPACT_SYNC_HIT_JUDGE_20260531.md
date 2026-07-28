# Impact-sync hit judgment integration (2026-05-31)

This build upgrades `hit_judge_server.py` from a loose nearest-frame bridge into an impact-timestamp-driven hit matcher.

## What changed

The event-camera bullet detector remains unchanged.  Bullet extraction, turn/terminate generation, `BulletEvent`, UDP sending, bullet-id stitching, and all original event-side logic are still the source of truth.

The new logic is only inside the hit-judge bridge:

```text
BulletEvent turn/terminate timestamp_us
    -> event_trigger_{camera}.csv
    -> MVS trigger period / sync_index
    -> matching human_result_{camera}.jsonl frame
    -> event-to-MVS homography
    -> body mask / bbox hit candidate
```

It no longer treats UDP receive time or the latest YOLO result as timing truth.

## Important JSON options

```json
"hit_judge": {
  "impact_sync_enable": true,
  "impact_event_types": ["turn", "terminate"],
  "impact_person_strategy": "nearest",
  "impact_anchor_period_ms": 40.0,
  "impact_midpoint_ms": 20.0,
  "impact_pending_timeout_ms": 1500.0,
  "impact_max_offset_ms": 45.0,
  "impact_use_mask": true
}
```

`impact_person_strategy="nearest"` means:

```text
impact offset < 20 ms   -> use MVS frame k
impact offset >= 20 ms  -> use MVS frame k+1
```

This is usually better for running people than always using the start-of-period MVS frame.

Other supported strategies:

```text
period  -> always use frame k where T_k <= impact < T_{k+1}
union   -> test frame k and k+1 when available
nearest -> default, choose the closest synchronized MVS frame
```

## Pending queue

The server keeps BulletEvents briefly if the required MVS human frame or trigger CSV row has not arrived yet.  This prevents early UDP arrivals from being incorrectly discarded while YOLO is still processing.

Default wait:

```json
"impact_pending_timeout_ms": 1500.0
```

## Output

The output JSONL still uses:

```text
sync_ipc/hit_candidate_{camera}.jsonl
sync_ipc/hit_candidate_all.jsonl
```

Additional fields are now included:

```json
"judge_mode": "impact_sync",
"impact_sync_index": 123,
"impact_anchor_ts_us": 123456789,
"impact_next_sync_index": 124,
"impact_offset_ms": 17.2,
"person_frame_sync_index": 123,
"human_match_strategy": "nearest_current"
```

## Running

From the project root:

```bash
python3 hit_judge_server.py --config hik_rgb_yolo_seg_multicam_same_model_stableid_v5.json --port 5003 --cams A,B,C,D,E,F
```

or from `rentijiance/`:

```bash
./run_hit_judge_server.sh
```

## What did not change

The original bullet detection / event-camera judgment code was not changed.  This is a stricter bridge from existing BulletEvents into MVS human segmentation.
