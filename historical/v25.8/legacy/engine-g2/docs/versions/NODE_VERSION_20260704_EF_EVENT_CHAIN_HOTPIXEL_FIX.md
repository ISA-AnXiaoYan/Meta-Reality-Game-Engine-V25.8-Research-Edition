# Node Version - 2026-07-04 EF Event Chain Hot-Pixel Fix

## Scope

This node closes the 2026-07-04 EF event-camera tracking diagnosis and repair.
It is a source-code and workflow handoff point for the A-test branch:

- Branch: `a-test-20260703-v20-hit-candidate-attribution-split`
- Remote project: `ysxq@192.168.31.115:~/PycharmProjects/Ids_Test_3.9/test622yolo_gpu_627`
- Launch profile: `launch_profiles/627_event_overlay_bev.json`
- Verified run label: `ef_hotpixel_cluster_fix_v1`
- Verified run dir: `sync_ipc/desktop_runs/ef_hotpixel_cluster_fix_v1_20260704_153515_display1_keepstdin`
- Verified archive: `sync_ipc/desktop_runs/ef_hotpixel_cluster_fix_v1_20260704_153515_display1_keepstdin.tar.gz`

## What Changed

### Event worker EF tracking repair

- Added tracking-only hot-pixel suppression before STC / Spatter / LineFilter.
- Kept Event overlay / raw visualization untouched so hardware data visibility is preserved.
- Added static hot-pixel map in the launch profile:
  - E: `(228, 19)`
  - F: `(202, 272)`
- Added cluster sanitize before `line_filter.update()` to remove invalid/out-of-range clusters.
- Added F-only shadow LineFilter diagnostics with relaxed parameters. Shadow mode does not emit bullet points or change final behavior.
- Rebuilds the shadow LineFilter when realtime detection state is reset.

### Status and diagnosis

Event worker status and unified system status now expose:

- `track_event_count_before_hot_pixel`
- `event_hot_pixel_filter`
- `event_cluster_sanitize`
- `line_filter_shadow_diag`
- `event_block_stage_reason`
- `event_spatial_collapse_suspect`

The Sync Diagnostics status window now shows per-camera hot-pixel removal,
cluster sanitize, and shadow LineFilter counters in the Event chain row.

### Toolchain and workflow

- Remote launch/stop/monitor continues to go through `tools/remote_test.ps1`.
- Complex Linux logic remains in `remote_ops/`.
- For a mixed local worktree, do not use broad `deploy-worktree` blindly. Either commit first, use snapshot deploy, or copy the exact intended files.
- Remote smoke evidence is recorded by run label and archived before Git sync.

## Validation

### Local validation

Used bundled Codex Python because this Windows environment did not expose
`python` or `py` on PATH:

```powershell
$env:PYTHONPYCACHEPREFIX="$env:TEMP\codex_pycache_ef_fix"
& "C:\Users\axyis\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m py_compile `
  metavision_spatter_tracking_linefilter_tsf1_autosender_backfill.py `
  launch_fusion_system.py `
  pyopengl_cuda_preview_renderer.py
```

Also verified the launch profile is accepted:

```powershell
& "C:\Users\axyis\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" `
  launch_fusion_system.py --launch-profile launch_profiles/627_event_overlay_bev.json --help
```

### Remote validation

Remote conda validation passed:

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache_ef_fix \
  /home/ysxq/miniconda3/envs/ids_recv/bin/python3 -m py_compile \
  metavision_spatter_tracking_linefilter_tsf1_autosender_backfill.py \
  launch_fusion_system.py \
  pyopengl_cuda_preview_renderer.py

/home/ysxq/miniconda3/envs/ids_recv/bin/python3 \
  launch_fusion_system.py --launch-profile launch_profiles/627_event_overlay_bev.json --help
```

Remote run result:

- `launcher_rc=0`
- `runtime_overall_state=OK`
- `runtime_broken_links=[]`
- `runtime_degraded_links=[]`
- `status_key_ok=true`
- `mask_stats_jsonl_no_growth=true`
- Global BEV render FPS: about `30.2fps`
- Global BEV event layer: `dilate_backend=cv2_dilate`, `dilate_px=16`

## EF Findings After Fix

Before this node, E/F could show overlay activity while bullet tracking did not
produce display tracks. Diagnosis showed:

- E was dominated by hot pixel `(228,19)`.
- F had a persistent hot pixel `(202,272)` plus invalid cluster pollution.

After this node:

- E/F hot-pixel filtering is active and visible in status.
- E/F overlay remains drawn, so visual evidence of hardware data is preserved.
- The block stage has moved downstream to LineFilter / cluster-shape rejection.

Observed examples during the verified run:

- E: hot-pixel filter removed hundreds of events per slice; block stage became `spatter_no_cluster` or `line_filter_rejected_all_clusters`.
- F: hot-pixel filter removed the `(202,272)` hotspot; cluster sanitize removed invalid clusters when present; official and shadow LineFilter still did not form draw paths in the sampled windows.

This means the previous "hardware/overlay has data but no EF bullet display"
case is no longer a black box. The current remaining problem is not hardware
silence. It is the shape/cluster/LineFilter acceptance of the remaining events.

## Known Risks And Open Work

- E/F still may not show bullet trajectories until the remaining event clusters
  satisfy LineFilter or LineFilter is further tuned.
- F shadow LineFilter is diagnostic only. Do not treat it as production output.
- Hot-pixel learning is intentionally disabled in the profile for this node to
  avoid adding runtime uncertainty. Static pixels are the only enabled filter.
- OBS-SRT ffmpeg still shows restart/backoff noise in the runner tail. This was
  present outside the EF tracking fix and is not closed by this node.
- Game bridge and hit-judge work remains separate from this EF tracking node.
- Status JSON can be read mid-write through SMB. Use retrying JSON readers for
  analysis scripts instead of single-shot PowerShell `ConvertFrom-Json`.

## Handoff Rules

1. Start from the Git commit/tag for this node, not from an ad hoc remote state.
2. Use `tools/remote_test.ps1` for remote start/stop/monitor.
3. Keep runtime data, archives, datasets, `sync_ipc/`, and caches out of Git.
4. When the worktree is mixed, deploy only the intended files or commit first.
5. For the next EF iteration, compare:
   - `event_hot_pixel_filter.output_count`
   - `event_cluster_sanitize.removed_count`
   - `line_filter_reject_diag.reject_reason_counts`
   - `line_filter_shadow_diag.accepted_count`
   - `bullet_display_diag.draw_paths_count`

