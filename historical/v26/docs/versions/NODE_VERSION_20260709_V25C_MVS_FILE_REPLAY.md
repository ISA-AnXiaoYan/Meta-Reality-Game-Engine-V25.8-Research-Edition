# Node Version: 2026-07-09 V25-C MVS File Replay

## Scope

This node implements the V25-C visible-light offline source path:

- `mvs_source_mode=live_camera|file_replay` startup selection.
- `remote_ops/mvs_file_frame_broker.py` as the MVS file replay source.
- Full-system launcher integration through `launch_fusion_system.py`.
- Remote workflow support through `tools/remote_test.ps1 smoke-mvs-replay`.
- Full-system archive/summary support for `mvs_file_frame_broker_status/stats`.

Production default remains `mvs_source_mode=live_camera`.

## Implementation Notes

`mvs_source_mode=file_replay` starts `MVS_FILE_BROKER` and does not start live MVS split workers or the live trigger coordinator, avoiding double writes to:

- `/dev/shm/mvs_latest_{camera}.mmap`
- `/dev/shm/mvs_raw_latest_{camera}.mmap`
- `sync_ipc/mvs_latest_{camera}.json`

The broker publishes replay trigger evidence:

- `sync_ipc/trigger_status.json`
- `sync_ipc/global_trigger_timeline.jsonl`
- `sync_ipc/global_trigger_timeline_latest.json`

The broker publishes A-H by aligned `program_sync_index/sync_index` batches. It does not publish per-camera latest independently.

Replay sidecar timing is rebased to the current replay wall clock:

- `capture_ts_us`
- `timestamp_us`
- `mvs_soft_trigger_est_wall_us`
- `soft_trigger_cmd_wall_us`

Original recording time is preserved in `record_capture_ts_us_original`.

The first implementation uses a zero-length exposure window for replay:

- `rgb_exposure_time_us=0`
- `rgb_exposure_start_offset_us=0`
- `rgb_exposure_mid_offset_us=0`
- `rgb_exposure_end_offset_us=0`
- `exposure_source=mvs_file_replay_zero_window`

This is intentional for V25-C so Global YOLO input buffering can validate replay sync without waiting for missing exposure-end evidence.

## Validation

Dataset used:

```text
recordings/20260708_181936_full
```

Known bad dataset:

```text
recordings/v25b_exttrig_smoke_20260709_154731_display1_keepstdin_full
```

The bad dataset has only one MVS frame on H and must not be used for MVS replay acceptance.

### Standalone Smoke

Command:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools\remote_test.ps1 `
  smoke-mvs-broker `
  -DatasetDir recordings/20260708_181936_full `
  -MaxFrames 80 `
  -ReplayTiming fast
```

Result:

- A-H published 10 aligned batches.
- A-H each published 10 frames.
- `error_count=0`.
- A-F skipped 5 initial frames to align with G/H starting sync.

### Full-System Smoke

Final validation run:

```text
sync_ipc/full_system_tests/v25c_mvs_replay_timefix_20260709_185122_display1_keepstdin
```

Key result:

- `launcher_rc=124`, normal timeout.
- `mvs_file_broker_stats.source_mode=file_replay`.
- A-H each published 1898 frames.
- `mvs_file_broker_stats.error_count=0`.
- Global YOLO:
  - `input_alignment_full_batch_ratio=0.9981`
  - `input_alignment_same_sync_batch_ratio=1.0`
  - `input_alignment_current_aligned_full_batch_ratio=0.9981`
  - `input_buffer_aligned_full_ratio=0.9981`
  - `input_buffer_fallback_ratio=0.0019`
  - `visual_mid_aligned_batch_ratio=1.0`
  - `capture_ts_spread_ms_p95=0.0`
- Global BEV:
  - `render_fps=30.32`
  - MVS alignment selected 8/8 cameras.
  - MVS offset A-H = 0ms.

The run still reports `runtime_overall_state=BROKEN` because this was an MVS-only replay smoke while Event still ran in live mode and produced no valid sparse/trigger output. That is not a V25-C MVS replay failure. Event file replay/full replay acceptance remains V25-B/V25-E work.

## Remaining Risks

- Full Event+MVS replay still needs V25-D/E validation.
- Replay exposure is currently a zero-window simulation. If later evaluations need real exposure timing, full recording must preserve per-frame exposure readback in the dataset manifest.
- MVS raw-preview replay is grayscale/Bayer8-compatible. It validates reader contracts but is not yet high-fidelity visual replay.
- `smoke-mvs-replay` summary currently includes live Event health. In no-event scenes it may report Event broken even when MVS replay is healthy.

## Recommended Next Step

Proceed to V25-D/E:

1. Make manifest-driven source expansion the default for replay.
2. Add a full replay smoke using both `event_source_mode=file_replay` and `mvs_source_mode=file_replay`.
3. Add mode-aware summary verdicts so MVS-only replay can pass without being polluted by optional live Event status.
