# V15 noise/DEFH test report

Date: 2026-07-01

## Scope

V15 focuses on suppressing false ballistic hits caused by human-motion event noise, while preserving enough diagnostics to investigate DEFH event-camera detection loss.

Changes tested:

- `visual_point_hit_min_track_points = 3`
- `visual_point_hit_min_track_path_len_px = 20.0`
- `visual_point_hit_body_motion_reject_enable = true`
- V15 test archive now includes hit candidates, hit rejects, event worker status, event loop profiles, event human-mask stats, event sync health, event trigger CSV, overlay bullet point JSONL, and effective config snapshots.

## Stage Backup And Archive

Stage node recorded before V15:

- `../versions/NODE_VERSION_20260701_V14_E_HIT_NO_MISS_STAGE.md`

Remote pre-change backups:

- `sync_ipc/code_backups/pre_v15_noise_defh_20260701_121840`
- `sync_ipc/code_backups/pre_v15_effective_20260701_123617`

Invalid first V15 run, caused by profile using `ids_8cam_fusion_config_627_runtime.json` instead of the uploaded `ids_8cam_fusion_config.json`:

- `sync_ipc/full_system_tests/fullsys_v15_noise_defh_10min_20260701_122021_display1_keepstdin.tar.gz`

Valid effective V15 run:

- `sync_ipc/full_system_tests/fullsys_v15_noise_defh_10min_20260701_123630_display1_keepstdin.tar.gz`
- `sync_ipc/full_system_tests/fullsys_v15_noise_defh_10min_20260701_123630_display1_keepstdin/analysis_v15_noise_defh.json`

## Key Result

The effective V15 run had no real hit event. It can only be used as a no-real-shot noise suppression run, not as a miss-rate acceptance run.

Under that boundary, the effective V15 run reduced hit candidates from V14-E's 116 candidates to 1 candidate in a 10-minute run.

- Candidate reduction: about 99.1%.
- Remaining candidate: camera A, `segment_intersect_mask`, path `24.958px`, `8` track points, duration `4ms`.
- Short candidates below `20px`: `0`.
- Main new reject reason: `point_track_path_too_short = 686`.

## Effective V15 Metrics

Launcher:

- `launcher_rc = 124`, normal timeout stop.
- `Traceback = 0`
- `unrecognized_args = 0`
- `restart_backoff = 0`
- `Address already in use = 0`
- `Broken pipe = 0`
- `drop_late = 1`

Hit candidates:

- Total: `1`
- By camera: `A: 1`
- By method: `segment_intersect_mask: 1`

Rejects:

- Total: `927`
- `point_track_path_too_short: 686`
- `point_no_human_timeout: 154`
- `point_no_candidate: 39`
- `point_track_points_too_small: 26`
- `point_bad_trigger: 18`
- `point_no_trigger: 4`

Rejects by camera:

- `A: 215`
- `B: 167`
- `C: 32`
- `D: 192`
- `E: 14`
- `F: 86`
- `G: 110`
- `H: 111`

Runtime health:

- Global YOLO state: `running`
- Global YOLO workers: `running, running`
- Legacy YOLO shards: skipped by Global YOLO primary path.
- Global BEV render FPS: about `30.14`.
- Global BEV output FPS: about `9.72`.
- During mid-run observation, all A-H event workers refreshed and reached `LOCKED`; the initial stale H status was an old status-file window, not a persistent process failure.

## Late Bullet Render Review

User observation after the run:

- There were no real hit events in this test.
- Bullet trajectories were not rendered during the late part of the test.

Post-run evidence:

- Run window: `12:36:30` to `12:46:50`.
- `overlay_bullet_point_all.jsonl` had `973` rows.
- First bullet point: `12:37:05.396`.
- Last bullet point: `12:42:32.856`.
- There were no new bullet point records for the final `257.143s`.
- Global BEV final state had `recent_records=0`, `projected_points=0`, `visible_points=0`, `draw_points=0`, `latest_age_ms=-1.0`, and `file_age_ms=247085.478`.
- PyOpenGL CUDA Preview final state had `trajectory_reader.open=True`, `trajectory_drawn_vtx=0`, and `trajectory_reader.life_ms=2500.0`.

Last bullet point by camera:

- `A: 12:41:51.907`
- `B: 12:41:56.500`
- `C: 12:41:42.374`
- `D: 12:41:24.920`
- `E: 12:41:59.560`
- `F: 12:42:32.856`
- `G: 12:42:09.477`
- `H: 12:41:59.927`

Direct conclusion:

- Late bullet rendering disappeared because the real-time bullet point source stopped producing new `overlay_bullet_point` records after `12:42:32.856`.
- Global BEV and PyOpenGL Preview did not show late trajectories because all existing trajectory records exceeded their visible TTL.
- This is not evidence of a Global BEV draw crash or Preview CUDA/PBO failure.

Upstream chain:

- Event trigger CSVs continued until the test end and were `LOCKED`, so the sync trigger chain was still alive.
- Event Overlay Overview still had fresh event overlay frames near the end, so event frame visualization was still alive.
- `BulletEventSender` closed normally on all 8 event workers with `dropped=0`; there was no queue overflow.
- `process_display_paths failed=0`; the display-path send function did not report exceptions.
- Therefore the likely missing stage is `line_filter.get_draw_paths(False)` not producing confirmed/display trajectory points late in the test.

Risk:

- Since V15 uses the display-layer path as the trajectory source, late-stage absence of draw paths makes both Preview and Global BEV bullet rendering disappear even though event overlays and triggers continue.
- The current status window reports render/read state but does not explicitly report `draw_paths` count or last `overlay_bullet_point` age as a broken link.

## Direct Bug Found

The first V15 run did not exercise the intended thresholds because the formal launch profile reads:

- `launch_profiles/627_event_overlay_bev.json`
- `config = ids_8cam_fusion_config_627_runtime.json`

The local upload initially changed `ids_8cam_fusion_config.json`, but the formal remote profile used `ids_8cam_fusion_config_627_runtime.json`. As a result, the generated runtime file had:

- `visual_point_hit_min_track_points = null`
- `visual_point_hit_min_track_path_len_px = null`
- `visual_point_hit_body_motion_reject_enable = null`

Fix applied:

- Patched `ids_8cam_fusion_config_627_runtime.json`.
- Updated V15 test script to patch and archive the formal profile config before launch.
- Verified generated `sync_ipc/_runtime_config/mvs_runtime_from_unified.json` contains the three V15 keys before the effective run.

## Current Assessment

V15-A false-positive suppression is effective in the no-real-shot/noise run, but this run does not validate real-hit recall.

Late bullet rendering has a separate upstream data-source issue: the confirmed display-path bullet point stream stopped several minutes before the test ended.

It should not yet be accepted as final hit-judgement tuning until a real-shot run confirms no new misses. The remaining single candidate has only `4ms` duration but path above `20px`, so the next tuning option is either:

- raise visual point path threshold from `20px` to `35px`, or
- add a conservative visual point duration threshold, for example `>=8ms`.

The path threshold is simpler but has a higher miss risk for short visible tracks. The duration threshold is more directly aimed at human-motion bursts, but needs real-shot evidence before enabling.
