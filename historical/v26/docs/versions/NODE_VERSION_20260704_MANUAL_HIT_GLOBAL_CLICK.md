# Node Version - 2026-07-04 Manual Hit Preview And Global BEV Click

## Scope

This node closes the A-test manual-hit control path used for operator confirmed
hit output to the game process.

- Branch: `a-test-20260703-v20-hit-candidate-attribution-split`
- Remote project: `ysxq@192.168.31.115:~/PycharmProjects/Ids_Test_3.9/test622yolo_gpu_627`
- Launch profile: `launch_profiles/627_event_overlay_bev.json`
- Active desktop run after closure deploy:
  `sync_ipc/desktop_runs/desktop_latest_20260704_174556_display1_keepstdin`
- Previous live-click validation run:
  `sync_ipc/desktop_runs/desktop_latest_20260704_173951_display1_keepstdin`
- Previous live-click validation archive:
  `sync_ipc/desktop_runs/desktop_latest_20260704_173951_display1_keepstdin.tar.gz`

## What Changed

### Manual hit service

- Added `manual_hit_service.py` as an independent sidecar.
- The sidecar reads `sync_ipc/manual_hit_commands.jsonl`.
- It resolves the clicked target against Global BEV display IDs first, then
  falls back to runtime Global Person ID evidence when needed.
- It emits publishable `damage_attribution` events into
  `sync_ipc/damage_attribution_events.jsonl`.
- Damage source remains the virtual global ID `99`.
- Target ID source is the Global BEV display ID when available.
- Duplicate clicks on the same target are rate limited by `manual_hit_dedup_ms`
  in the launch profile.

### Main preview click path

- The PyOpenGL CUDA preview window can write manual-hit commands directly.
- Status Diagnostics exposes controls for:
  - manual click hit mode
  - main preview hit display
  - trajectory-only display
  - human label detail vs ID-only mode
- Enabling manual click mode forces preview hit display on and trajectory-only
  mode off so clickable person targets are not cleared.

### Global BEV click path

- The Global BEV Fusion window can now write manual-hit commands on left click.
- Click target selection uses the same visible Global BEV person overlay points
  that the operator sees.
- The command prefers `display_slot_id` / `display_player_id` from the Global
  BEV overlay stabilizer and preserves source runtime `global_id` as evidence.
- Global BEV exposes `manual_hit` status:
  - `click_enable`
  - `command_jsonl`
  - `target_cache_count`
  - `target_cache_age_ms`
  - `command_seq`
  - `last_command`
- The Status Diagnostics manual-hit toggle now writes both:
  - `sync_ipc/preview_control.json`
  - `sync_ipc/global_bev_control.json`

### Game event bridge

- Added `game_event_bridge_control.json` and runtime `hit_source_mode`.
- Supported modes:
  - `all`
  - `manual_only`
  - `auto_only`
- The launch profile defaults to `manual_only`.
- Non-manual automatic hit events are suppressed while manual-only mode is
  active.
- Outbound hit events include source diagnostics:
  - `event_source`
  - `manual`
  - `manual_authority`
  - `attribution_source`
- Manual Global BEV clicks are labelled as `manual_global_bev_click`.

### Remote test workflow

- `tools/dev_check.ps1` now compiles the manual-hit and bridge services.
- `tools/remote_test.ps1` and `remote_ops/run_fullsys_test.sh` can pass common
  hit-logic hot-update parameters through environment variables.
- Remote validation continues to use `tools/remote_test.ps1`; avoid handwritten
  PowerShell/SSH quoting for complex commands.

## Runtime Contract

Operator-facing hit output is now separated from automatic hit judgement:

1. Preview window click or Global BEV window click creates a manual command.
2. `manual_hit_service.py` resolves the target and writes a publishable
   attribution event.
3. `game_event_bridge_service.py` is the only process that publishes to the game
   manager.
4. With `hit_source_mode=manual_only`, automatic hit candidates are not allowed
   to publish game damage.

Expected game event behavior:

- Source: virtual global ID `99`
- Target: clicked Global BEV display player ID when available
- Event source:
  - `manual_preview_click` for main preview clicks
  - `manual_global_bev_click` for Global BEV clicks

## Validation

### Local validation

Command:

```powershell
powershell -ExecutionPolicy Bypass -File tools\dev_check.ps1
```

Result:

- `py_compile=ok`
- `json_ok=launch_profiles/627_event_overlay_bev.json`
- `dev_check=ok`

### Remote validation

Deploy and start command:

```powershell
powershell -ExecutionPolicy Bypass -File tools\remote_test.ps1 start-latest -DeployWorktree -ReadyWait 180
```

Current active run:

- Run dir:
  `sync_ipc/desktop_runs/desktop_latest_20260704_174556_display1_keepstdin`
- Final ready state: `OK`
- Event ready: `8/8`
- Global BEV render FPS: about `30.17fps`
- `global_bev_status.json.manual_hit.click_enable=true`
- `global_bev_status.json.manual_hit.target_cache_count=11`
- `manual_hit_status.json.enabled=true`
- `game_event_bridge_status.json.hit_source_mode=manual_only`
- `game_event_bridge_status.json.send_hits_enabled=true`
- Log scan after restart:
  - `global_bev.log`: no `Traceback` / `ERROR`
  - `manual_hit.log`: no `Traceback` / `ERROR`
  - `game_event_bridge.log`: no `Traceback` / `ERROR`

Previous live-click run evidence:

- Run dir:
  `sync_ipc/desktop_runs/desktop_latest_20260704_173951_display1_keepstdin`
- Global BEV click command path was active:
  `/home/ysxq/PycharmProjects/Ids_Test_3.9/test622yolo_gpu_627/sync_ipc/manual_hit_commands.jsonl`
- Global BEV click status before restart:
  - `click_enable=true`
  - `target_cache_count=8`
- Observed Global BEV click flow:
  - 18 Global BEV click commands written
  - 16 accepted by `manual_hit_service.py`
  - 2 rejected by target dedup cooldown
  - 16 sent by `game_event_bridge_service.py`
- Bridge mode during that run:
  - `hit_source_mode=manual_only`
  - `sent_hit=16`
  - non-manual hit candidates were suppressed by source mode

## Known Risks And Open Work

- The post-restart run verified that the Global BEV click path is armed and
  healthy. The source-label rename to `manual_global_bev_click` should be
  confirmed on the next operator Global BEV click.
- Manual click mode intentionally bypasses automatic hit judgement. This is for
  operator confirmed output and dataset collection, not automatic hit accuracy
  validation.
- Manual hit target stability depends on Global BEV display ID stability. Known
  Global ID work remains open around handoff areas and close interaction cases.
- Dedup is target-based and currently defaults to `800ms`. Fast repeated manual
  clicks on the same player may be intentionally rejected.
- `manual_only` suppresses automatic hit events from publishing to the game
  bridge. Remember to switch mode back when automatic hit judgement is the
  intended authority.
- OBS-SRT ffmpeg restart/backoff noise still appears in runner tails. This node
  does not address streaming output.
- SMB reads of large JSONL tails can time out. Prefer compact status JSON and
  `tools/remote_test.ps1 monitor/summary` for live checks.

## Handoff Rules

1. Use this Git node as the source of truth, not the dirty remote worktree.
2. Start and stop remote tests through `tools/remote_test.ps1`.
3. Keep heavy runtime data, datasets, archives, `sync_ipc/`, and caches out of
   Git.
4. For the next validation click, check:
   - `global_bev_status.json.manual_hit.last_command`
   - `manual_hit_latest.json.latest_event.source`
   - `game_event_bridge_sent.jsonl` latest hit payload
5. For automatic hit work, explicitly set the authority and bridge source mode
   before testing so manual and automatic paths do not contaminate each other.
