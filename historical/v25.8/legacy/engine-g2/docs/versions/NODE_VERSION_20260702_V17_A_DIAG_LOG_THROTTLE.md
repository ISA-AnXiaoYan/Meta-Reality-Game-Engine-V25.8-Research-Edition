# V17-A Diagnostic Log Throttle

Date: 2026-07-02

## Scope

This node version fixes the long-run diagnostic I/O hotspot found in:

- `fullsys_10min_20260702_195947_display1_keepstdin`

The direct hotspot was `sync_ipc/event_human_mask_stats_{camera}.jsonl`, which grew to about 4-6 GB per camera in a 14.6 minute run.

## Changes

- Disabled `event_human_mask_stats_jsonl` by default in the 8-camera fusion config and launch profile.
- Added launcher-level runtime override so generated runtime config also keeps human-mask stats JSONL disabled unless explicitly requested.
- Added event-worker safeguards:
  - `--event-human-mask-stats-max-fps`, default `2.0`
  - `--event-human-mask-stats-flush-ms`, default `1000.0`
  - status fields: `stats_jsonl_enabled`, `stats_jsonl_path`, `stats_jsonl_max_fps`, `stats_jsonl_write_count`, `stats_jsonl_skip_count`
- Kept in-memory/event status counters available through `event_human_mask_stats`; only high-volume per-slice JSONL is disabled by default.

## Expected Effect

- Stop GB-level growth of `event_human_mask_stats_*.jsonl`.
- Reduce event-worker disk I/O pressure and JSON serialization overhead.
- Make the next long-run performance sample cleaner before optimizing BEV overlay, Preview render cadence, or Event worker CPU paths.

## Verification Points

- Event worker command lines should either omit `--event-human-mask-stats-jsonl` or pass it as an empty value.
- `sync_ipc/event_human_mask_stats_*.jsonl` should not be created in normal launch.
- `event_worker_{camera}_status.json` should still expose current `human_mask` stats with `stats_jsonl_enabled=false`.
- CPU hotspot comparison should be done after at least 5-10 minutes of runtime.

## Risks

- Offline tools that expect full per-slice `event_human_mask_stats_*.jsonl` will no longer have that file in normal runs.
- For targeted mask-delay diagnosis, explicitly enable `event_human_mask_stats_jsonl` and keep `event_human_mask_stats_max_fps` low unless full-resolution diagnosis is absolutely required.
