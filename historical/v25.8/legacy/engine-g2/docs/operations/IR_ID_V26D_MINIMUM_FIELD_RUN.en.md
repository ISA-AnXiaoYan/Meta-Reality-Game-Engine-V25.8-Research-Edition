# IR-ID V26-D Minimum Field Run

Status: code, offline regression, and a real field RAW sidecar replay have passed; the current live primary workspace has not been enabled.

## Boundary

- Runtime: the legacy Event runtime on `patch/v26-d-ir-id-legacy-sidecar`.
- Hook: after the Event worker receives `raw_evs` from its existing EVB FIFO and before human mask/STC/spatter/line-filter/bullet processing.
- Behavior: crop only the selected ROI, submit it non-blockingly to a background decoder, and write observation/status/perf sidecars.
- No HAL broker change, no second FIFO reader, and no write to Global Person ID, `identity_evidence`, hit judgement, or any formal authority output.
- Default off. Enablement requires all of `--active-led-marker-enable`, one camera alias, an ROI, and a run ID.

## Verified Facts

- Ten IR-ID regressions pass on this branch, covering 1/4/8/12 ms slice boundaries, EVP replay, default-off, one raw tap, no HAL marker logic, the async queue, and the field validator.
- The synthetic ID=7 self-test passes with the schema v2 `ir-id-blink-v1` observation contract.
- The historical G recording `recordings/20260710_163327/EVENT_G_4110035688/event_slices.evp` contains 14,966 4 ms slices and 1,941 events. Full-frame and candidate ROI decoding both produced 0 pulses and 0 observations; it is not hardware-detection evidence.
- The user-provided real field RAW `recording_2026-07-10_17-22-32.raw` was replayed on the field host with `/usr/bin/python3`. In ROI `648,480,48,48`, the V26-D sidecar produced 139 schema-v2 observations across 39.95 seconds and 9,987 4 ms slices; every observation was ID=7. Median confidence was 0.935, minimum confidence was 0.822, and the location remained near `(680.65, 513.86)`.
- The 64-deep replay queue dropped 291 slices. The bounded 512-deep queue processed all 9,987 slices with 0 drops and 0 sidecar errors; the maximum decoder `process_ms_p95` across 40 perf samples was 0.152 ms. The field command therefore uses 512.
- Historical `IRID_20260710_171351_G_B01_H150_B16_FULL` still contains only its manifest, so its `brightness=16/full8x8` configuration cannot be audit-linked to the RAW above. However, the separate objective of proving reliable EventCD recognition of a high-brightness pure-colour signal is now met by the real RAW and its duplicate experiment can be cancelled. full8x8 remains a short visibility diagnostic only, never the final wearable solution.
- The field host on the 192.168.31 network cannot reach the ESP SoftAP default address `http://10.10.10.1`. An operator connected to `ESP32-S3-Matrix-LEDID-62F2A0` must first confirm that the board is online and emitting.

## Field Preconditions

1. On a terminal connected to the ESP SoftAP, run:

```bash
python tools/active_led_marker_poc/esp32_marker_control.py --base-url http://10.10.10.1 status
python tools/active_led_marker_poc/esp32_marker_control.py --base-url http://10.10.10.1 set player_id=7 alpha_ms=20 pulse_width_ms=10 brightness=16 mask=center3x3
python tools/active_led_marker_poc/esp32_marker_control.py --base-url http://10.10.10.1 start
```

2. Keep hit judgement safe-off for the minimum run. IR-ID does not change judgement, but this run must not carry production decisions.
3. Build an ROI heatmap from a fresh G event-slice recording. Do not treat the July 10 candidate ROIs as final; if no clear local hotspot is visible within 10 seconds, stop and recover hardware visibility first.
4. The ROI must cover only the expected LED pixels. Start at 48x48; never enable the sidecar with the full frame.

## Launch

Run this in a test snapshot that has this branch applied and is compatible with the field runtime. `G`, ROI, and run ID below are examples and must be replaced by the actual run values.

```bash
RUN_ID=IRID_V26D_G_$(date +%Y%m%d_%H%M%S)
LAUNCH_EXTRA_ARGS="--active-led-marker-enable --active-led-marker-cameras G --active-led-marker-roi 648,480,48,48 --active-led-marker-run-id ${RUN_ID} --active-led-marker-alpha-ms 20 --active-led-marker-bit-count 8 --active-led-marker-bin-us 1000 --active-led-marker-min-events-per-bin 8 --active-led-marker-min-segment-events 20 --active-led-marker-queue-size 512" \
PROFILE=launch_profiles/627_event_overlay_bev.json \
bash remote_ops/run_fullsys_test.sh "${RUN_ID}" 120
```

The current field workspace has many uncommitted files. Do not overwrite its `launch_fusion_system.py` or Event worker. Apply this branch in a matching test snapshot, or have the runtime owner perform a three-way merge. The passed result is a recorded-field replay, not a hot-overlay validation of the current live process.

## Acceptance And Rollback

After the run, verify the three G artifacts:

```bash
python validate_ir_id_field_run.py \
  --observation sync_ipc/active_marker_observation_G.jsonl \
  --status sync_ipc/active_marker_status_G.json \
  --perf sync_ipc/active_marker_perf_G.csv \
  --expected-id 7 --run-id "${RUN_ID}" \
  --min-observations 3 --max-false-ids 0 --max-queue-dropped 0 --max-p95-ms 2.0 \
  --output sync_ipc/ir_id_runs/${RUN_ID}/v26d_field_validation.json
```

Pass criteria: at least three schema-v2 ID=7 observations with the requested run ID; no false IDs, no queue drops, no status error, and every decoder `process_ms_p95` at or below 2 ms. Any failure is a field failure and must not progress to `identity_evidence`.

Rollback is to stop the run and omit `--active-led-marker-enable` on the next start. The sidecar does not persist in the Event worker authority path. Preserve observations, status, perf, EVP/RAW, and the validation JSON as failure evidence.
