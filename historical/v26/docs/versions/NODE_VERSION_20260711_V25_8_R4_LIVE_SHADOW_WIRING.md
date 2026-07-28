# V25.8-R4 Live Shadow Wiring Node

## Scope

This node adds three launcher-managed V25 processes behind the explicit
`v25_live_shadow_enable` switch:

- `mrg_visualization.live_shadow`
- `mrg_combat_engine.live_shadow`
- `mrg_game_bridge.live_shadow`

The default remains disabled. The accepted remote run enabled the services only
in a disposable profile and restored every candidate Legacy file afterward.

## Data Flow

```text
Legacy global_person_latest.json
  -> V25 Visualization Shadow
  -> namespaced visualization latest/status

Legacy pseudo_hit_all.jsonl + hit_judge_status.json
  -> V25 Combat Shadow
  -> namespaced combat decisions/status
  -> V25 Game Bridge dry-run
  -> namespaced local JSONL/status only
```

Legacy pseudo-hit rows are compatibility observations unless they contain the
full V25 trajectory/world/mask/trigger contract. Compatibility decisions carry
`algorithm_evaluated=false`; they are not presented as a new Combat verdict.

## Authority Contract

- Visualization and Combat: `authority=shadow_only`.
- Game Bridge: `authority=dry_run_only`.
- All three: `official_result=false`, `official_writer_count=0`,
  `writes_legacy_ipc=false` and `production_network_enabled=false`.
- The existing Legacy Game Event Bridge and final-hit authority are unchanged.
- `system_status_latest.json.categories.v25_shadow_services` reports every
  service, input freshness, session consistency and exact broken link.

## Remote Acceptance

Passed on candidate 1:

`sync_ipc/desktop_runs/v25_8_r4_20260711_131628_display1_keepstdin`

- Each V25 service was observed in 116 process samples.
- Each service published 116 running status samples.
- Combat and Game Bridge inputs were connected in 115 samples.
- Unified V25 Shadow status was valid in 115 samples.
- Visualization performed 2335 valid reads from live Legacy world snapshots.
- Game Bridge network connection attempts: 0.
- R3 regression remained passed: BEV render/output 30.333/12.745fps,
  Sync CPU 2.871%, diagnostic writes 0.456MiB/min, polygons zero-missing.
- Patch restore: `restored_verified`; residual managed processes: 0.

## Remaining Limit

The unattended run contained no pseudo-hit records. Combat and Game Bridge were
therefore live-connected but idle. Their deterministic event path is covered by
local tests, but remote live hit functionality and field effectiveness are not
claimed. R5 must exercise the event path with deterministic replay and fault
injection before RC1.

