# YOLO Shard Optimization

This package adds a multi-process YOLO/MVS_AGG mode intended to reduce the single-process CPU bottleneck in `hik_rgb_yolo_seg_multicam_same_model_stableid_v5.py`.

## What changed

- `launch_fusion_system.py`
  - Adds `--mvs-yolo-shard-workers-enable`.
  - Adds `--mvs-yolo-shards A,C:B,D:E,G:F,H`.
  - Starts one YOLO worker process per shard instead of one monolithic `MVS_AGG`.
  - Writes per-shard runtime configs under `sync_ipc/_runtime_config/`.
  - Limits BLAS/OpenMP CPU threads per YOLO worker.

- `hik_rgb_yolo_seg_multicam_same_model_stableid_v5.py`
  - Supports `--cams`, `--yolo-worker-id`, external-MVS-only worker mode.
  - Supports config-driven camera subset filtering.
  - Adds async `human_result_{camera}.jsonl` writer.
  - Adds per-worker `yolo_worker_<id>_status.json` stage metrics.
  - Adds stale input/result counters.
  - Adds optional external postprocess process per YOLO shard for StableID/ReID/JSONL.

## Recommended first test

Use four YOLO workers:

```bash
--mvs-yolo-shard-workers-enable \
--mvs-yolo-shards A,C:B,D:E,G:F,H \
--yolo-worker-cpus 20-21,22-23,24-25,26-27 \
--yolo-worker-omp-num-threads 1 \
--yolo-async-jsonl-writer-enable \
--yolo-external-postprocess-enable \
--yolo-external-postprocess-cpus 10-11,12-13,14-15,16-17 \
--yolo-external-postprocess-queue-size 4 \
--yolo-drop-stale-input-ms 120 \
--yolo-drop-stale-result-ms 250
```

## Expected result

- Old mode: one `MVS_AGG`, around 25 batch/s for 8-camera batches.
- New mode: four workers, each processing 2-camera batches: `YOLO_AC`, `YOLO_BD`, `YOLO_EG`, `YOLO_FH`.
- Status window should show `yolo_worker_count=4` and higher total `yolo_infer_fps`.

## Output files

Each worker writes:

- `sync_ipc/yolo_worker_yolo_ac_status.json`
- `sync_ipc/yolo_worker_yolo_bd_status.json`
- `sync_ipc/yolo_worker_yolo_eg_status.json`
- `sync_ipc/yolo_worker_yolo_fh_status.json`
- `sync_ipc/yolo_postprocess_yolo_ac_status.json`
- `sync_ipc/yolo_postprocess_yolo_bd_status.json`
- `sync_ipc/yolo_postprocess_yolo_eg_status.json`
- `sync_ipc/yolo_postprocess_yolo_fh_status.json`
- `sync_ipc/yolo_infer_perf_yolo_ac.csv`
- `sync_ipc/yolo_infer_perf_yolo_bd.csv`
- `sync_ipc/yolo_infer_perf_yolo_eg.csv`
- `sync_ipc/yolo_infer_perf_yolo_fh.csv`

`human_result_{camera}.jsonl` schema is unchanged, so hit_judge and preview renderer remain compatible.
Each worker filters routes by `yolo_worker_cams`, so it only creates `/dev/shm/mvs_latest_{camera}` readers for its assigned cameras.
