# V25.8-R3 BEV, Polygon and Diagnostic-Disk Node

## Scope

This node closes the V25.8-R3 unattended work package. It changes only the
candidate Global YOLO polygon contract, Global BEV output scheduling, unified
status propagation, and V25 Sync diagnostic publication policy.

Production authority remains in Legacy. The remote runner installs these files
only for the candidate run, verifies their SHA restoration afterward, and does
not modify the persistent production launch profile.

## Code Changes

- `global_yolo_polygon_contract.py` defines usable polygon, `valid_empty`,
  `valid_complete`, disabled and invalid states.
- Global YOLO worker and merge status now expose
  `polygon_contract_state`, `polygon_output_enabled`, and
  `published_persons_without_polygon`.
- A worker with no detected people is a valid empty producer when polygon
  output is enabled; any published person without a usable polygon is invalid.
- `monotonic_rate_deadline.py` provides accumulated monotonic deadlines for
  Global BEV publication cadence.
- OpenGL upload and drawing remain owned by the GUI thread. CPU preparation
  remains outside the GUI thread, and duplicate MVS texture uploads stay
  suppressed.
- V25 Sync journal defaults to off; latest compatibility rings are bounded and
  published at 2 Hz. The R3 candidate places them on tmpfs.

## Remote Acceptance

Passed run:

`sync_ipc/desktop_runs/v25_8_r3_20260711_120643_display1_keepstdin`

- BEV render median: 30.329 fps.
- BEV output median: 12.762 fps.
- Polygon merge: `valid_complete`, missing people: 0.
- Sync CPU: 2.794%.
- Diagnostic filesystem writes: 0.446 MiB/min.
- MVS late-tick drops: 0.
- Candidate restore: `restored_verified`.
- Residual managed processes: 0.

## Important Boundary

The passing disposable profile requested `global_bev_output_fps=13` to satisfy
an observed output hard gate of 12 fps. The persistent production profile still
contains 12 fps and was not changed. That profile decision belongs to the R5
freeze, not this node.

No visible sparse Event points were present, so pixel-level Event/MVS visual
alignment remains unvalidated. Global YOLO throughput is monitor-only under the
current V25.8 decision; polygon correctness remains a hard gate.

