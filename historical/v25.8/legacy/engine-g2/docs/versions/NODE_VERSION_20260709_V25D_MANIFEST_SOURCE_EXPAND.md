# Node Version 20260709 V25-D/E Manifest Source Expansion

## Scope

This node extends the V25 replay toolchain so a single dataset folder can drive
Event replay, MVS replay, or full Event+MVS replay startup.

## Changes

- `remote_ops/check_v25_dataset_manifest.py`
  - Supports same-session split bundles:
    - `recordings/<session>` for MVS AVI and frame index files.
    - `raw_records/<session>` for Event RAW files.
  - Reads and merges manifests from all discovered sibling roots.
  - Records `search_roots`, `manifest_sources`, and `source_roots` in the report.
  - Generates launch args for replay modes with `--print-launch-extra-args`.
  - For full replay, passes the real MVS source root to
    `--mvs-file-frame-broker-dataset-dir`, even if the operator selected the
    raw-records folder.

- `tools/remote_test.ps1`
  - Uses manifest-generated launch args when `-DatasetDir` is provided for
    `smoke-event-replay`, `smoke-mvs-replay`, or `smoke-full-replay`.
  - Keeps the old explicit Event RAW mapping path only for manual no-dataset
    replay startup.

- Documentation
  - `REMOTE_TEST_WORKFLOW_CN.md` records the split-bundle rule and official
    External Trigger requirement.
  - `MVS_CAMERA_CHAIN_CN.md` records the MVS source-root rule.

## Validation

Local validation:

- Python compile passed for `remote_ops/check_v25_dataset_manifest.py`.
- PowerShell parser check passed for `tools/remote_test.ps1`.
- Synthetic split-bundle check passed:
  - selected root: `recordings/s1`
  - discovered Event root: `raw_records/s1`
  - discovered MVS root: `recordings/s1`
  - merged `event_external_trigger_valid=true`

Remote validation:

- `check-dataset-manifest` on `recordings/20260708_181936_full` now discovers
  the sibling Event RAW root `raw_records/20260708_181936_full`.
- `check-event-raw-exttrig -DatasetDir raw_records/20260708_181936_full
  -UpdateManifest` verified A-H RAW External Trigger tracks and updated the
  Event RAW manifest.
- First full replay smoke exposed launcher argument leakage: MVS file broker
  args were passed to Event workers. `multi_camera_launcher.py` now filters
  `mvs_source_mode` and `mvs_file_frame_broker_*` from Event worker args.
- Second full replay smoke exposed short Event RAW exhaustion. The Event file
  FIFO broker now supports `--loop`; manifest-generated full replay args enable
  `--event-file-fifo-broker-loop`.
- Third full replay smoke exposed missing replay sync-start: MVS file replay
  did not create `sync_start.flag`, so Event workers discarded all trigger
  edges while waiting for formal sync. `mvs_file_frame_broker.py` now writes
  `sync_start.flag` and `sync_start_latest.json` on the first aligned batch.
- Fourth full replay smoke confirmed Event workers receive and map triggers,
  but BEV Event/MVS alignment still does not pass for this historical split
  dataset. The root cause is dataset-level timing ambiguity: Event RAW replay
  and MVS AVI replay advance at different effective rates, and the current
  manifest lacks a recorded common trigger anchor that maps Event trigger index
  to MVS `program_sync_index`.
- Direct-index replay mapping was added as a guarded experiment for
  `source_mode=mvs_file_replay`, but this dataset still cannot be accepted for
  BEV Event/MVS sync because Event RAW loops/decoding progress do not match the
  MVS replay target window.
- `remote_ops/check_fullsys_run.py` now reports `event_worker_sync` in live,
  summary, and compact reports. It exposes BEV target sync, per-camera Event
  mapped sync, direct-index state, trigger counts, and mapped-vs-BEV deltas so
  future full-replay failures can be diagnosed without manually opening eight
  Event worker status JSON files.
- `remote_ops/check_v25_dataset_manifest.py` now separates full replay startup
  readiness from strict sync acceptance:
  - `usable_for_full_replay_startup`: Event/MVS replay sources can launch and be
    consumed by the main chain.
  - `usable_for_full_replay_sync`: startup is usable and the dataset also has a
    recorded Event-trigger-to-MVS-sync anchor.
  The historical `20260708_181936_full` split dataset is intentionally marked
  startup-usable but sync-not-accepted with `full_replay_sync_anchor_missing`.
- `remote_ops/run_fullsys_test.sh` now copies runtime replay-sync anchors into
  future full-recording datasets when a recording session exists:
  `event_trigger_map_*.jsonl`, `event_sync_map_ring_*.json`,
  `global_trigger_timeline*.json*`, and `sync_start.*` are copied into
  `raw_records/<session>/replay_sync/` and `recordings/<session>/replay_sync/`.
  This is non-blocking finalize work and is intended to make newly recorded
  datasets eligible for `usable_for_full_replay_sync`.

Current acceptance boundary:

- V25-D manifest sibling discovery: accepted.
- V25-B Event RAW ExtTrigger verification and FIFO loop: accepted for startup
  and worker consumption.
- V25-C MVS file broker startup and Global YOLO consumption: accepted for MVS
  replay smoke.
- V25-E full Event+MVS replay: partially integrated, not accepted for BEV
  Event/MVS sync until the dataset manifest records an original sync anchor,
  for example `event_trigger_zero_to_mvs_sync` per camera or a recorded
  `event_trigger_map_{camera}.jsonl` bundle.

## Risk

- Historical manifests may be partial or stale. The checker now prefers derived
  file presence plus merged External Trigger validity over a single manifest's
  replay booleans.
- Full replay must not mix different sessions. The automatic merge is limited to
  same-name sibling folders under `recordings/` and `raw_records/`.
- `-AllowMissingExtTrigger` is debug-only and cannot be used as formal sync
  acceptance evidence.
- A full replay dataset needs more than "Event RAW has External Trigger" and
  "MVS AVI has frame indices". It must also preserve the original common sync
  anchor between Event trigger edges and MVS `program_sync_index`; otherwise
  wall-time replay and direct-index replay can both drift.
