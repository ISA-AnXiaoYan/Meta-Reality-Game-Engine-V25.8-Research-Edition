# Handoff - Event Overlay / Global BEV Dilation Control Closure

## Read This First

This handoff closes the current A-test worktree at node:

`A-TEST-20260702-EVENT-OVERLAY-BEV-DILATE-CONTROL`

The node is intended as a safe continuation point for the next workflow, not as a final product release.

## Environment

Remote host:

```text
ysxq@192.168.31.115
```

Remote project:

```text
~/PycharmProjects/Ids_Test_3.9/test622yolo_gpu_627
```

Runtime environment:

```bash
conda activate ids_recv
export DISPLAY=:1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __NV_PRIME_RENDER_OFFLOAD=1
export __GL_SYNC_TO_VBLANK=0
export CUDA_VISIBLE_DEVICES=0
```

Launcher profile:

```text
launch_profiles/627_event_overlay_bev.json
```

Keep stdin alive when launching the full system:

```bash
tail -f /dev/null | /home/ysxq/miniconda3/envs/ids_recv/bin/python3 launch_fusion_system.py --launch-profile launch_profiles/627_event_overlay_bev.json
```

## What Changed

### Sync Diagnostics Controls

The status window now exposes:

- `Event独立窗`: controls the standalone Event Overlay Overview preview window.
- `BEV膨胀px`: controls event-layer dilation in Global BEV.

English keys remain stable in JSON for scripts and later regression analysis.

### Standalone Event Overlay Overview

The process stays alive when the window is hidden. The window is controlled by:

```text
sync_ipc/event_overlay_overview_control.json
```

Useful manual toggle:

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path("sync_ipc/event_overlay_overview_control.json")
p.write_text(json.dumps({"event_overlay_overview_visible": False}, ensure_ascii=False), encoding="utf-8")
PY
```

Restore:

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path("sync_ipc/event_overlay_overview_control.json")
p.write_text(json.dumps({"event_overlay_overview_visible": True}, ensure_ascii=False), encoding="utf-8")
PY
```

### Global BEV Event Layer Dilation

The Global BEV event layer uses:

```text
sync_ipc/global_bev_control.json
```

Manual example:

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path("sync_ipc/global_bev_control.json")
data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
data["show_event_layer"] = True
data["event_layer_dilate_px"] = 16
p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
PY
```

## Status Checks

Check Global BEV:

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path("sync_ipc/global_bev_status.json")
d = json.loads(p.read_text(encoding="utf-8"))
e = d.get("event_layer", {})
print({
    "render_fps": d.get("render_fps"),
    "output_fps": d.get("output_fps"),
    "last_error": d.get("last_error"),
    "event_enabled": e.get("enabled"),
    "dilate_px": e.get("dilate_px"),
    "dilate_backend": e.get("dilate_backend"),
    "skipped_dense": e.get("dilate_skipped_dense_cam_count"),
})
PY
```

Check Event Overview:

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path("sync_ipc/event_overlay_overview_status.json")
d = json.loads(p.read_text(encoding="utf-8"))
c = d.get("control", {})
print({
    "visible": c.get("window_visible"),
    "requested": c.get("visible_requested"),
    "error": c.get("error"),
})
PY
```

## Artifacts To Preserve

Remote:

- `sync_ipc/full_system_tests/pre_event_overlay_window_bev_dilate_20260702_223131.tar.gz`
- `sync_ipc/full_system_tests/node_event_overlay_bev_dilate_closure_20260702_224945.tar.gz`
- `sync_ipc/full_system_tests/fullsys_20min_bev_event_layer_20260702_224502_display1_keepstdin`

Local:

- `docs/versions/NODE_VERSION_20260702_A_TEST_EVENT_OVERLAY_BEV_DILATE_CONTROL.md`
- `docs/reports/HANDOFF_20260702_EVENT_OVERLAY_BEV_DILATE_CONTROL.md`
- `docs/versions/VERSION_MANIFEST_20260702_EVENT_OVERLAY_BEV_DILATE_CONTROL.json`
- `A_TEST_NODE_20260702_EVENT_OVERLAY_BEV_DILATE_CONTROL_WORKTREE.zip`
- `A_TEST_NODE_20260702_EVENT_OVERLAY_BEV_DILATE_CONTROL_WORKTREE.zip.sha256`

## Next Operator Checklist

1. Start from this worktree and read `WORKTREE_START_HERE_20260702.md`.
2. Confirm status window controls still write the expected JSON keys.
3. Confirm Global BEV `render_fps` stays near `30 fps`.
4. Confirm Event Overlay Overview can hide/show without process restart.
5. Do not treat this node as a hit-detection accuracy change; it is a visualization/control and operations-status node.
