# V25 Startup Control Panel Desktop Entry And Managed Start

Date: 2026-07-10

Branch: `v24-a-bev-event-sync-truth`

## Scope

This node productizes the V25 startup control panel on the remote desktop and
adds a managed one-click full-system start path.

The desktop entry opens the panel only. It does not start the system until the
operator presses `一键启动整套系统`.

## Runtime contract

- Desktop entry: `/home/ysxq/桌面/MRG-Startup-Control-Panel.desktop`
- GUI entry: `remote_ops/open_startup_control_panel.sh`
- Panel: `remote_ops/v25_replay_control_panel.py`
- Managed start bridge: `remote_ops/start_system_from_panel.py`
- Managed launcher: `remote_ops/start_latest_system.sh`
- Control input: `sync_ipc/v25_replay_status.json`
- Start result: `sync_ipc/v25_panel_launch_status.json`
- Panel log: `sync_ipc/v25_startup_control_panel.log`
- Managed run label: `desktop_latest`

The panel writes a structured `launch_extra_argv` array. The managed launcher
passes that array to `launch_fusion_system.py` without PowerShell/SSH or shell
string re-parsing. The legacy whitespace-delimited launch contract remains for
existing remote-test callers.

## Operator behavior

- Default source mode is `live`.
- Live mode explicitly requests `event_source_mode=live_hal` and
  `mvs_source_mode=live_camera`.
- Event, MVS, and full replay modes use the selected dataset and replay timing.
- A degraded replay configuration cannot start unless the operator explicitly
  enables the missing-EXTTRIG debug downgrade.
- Repeated clicks are protected by the existing managed launcher lock and PID
  contract.
- The GUI stays responsive while the start bridge waits up to 120 seconds for
  the real full-system status to become ready.

## Validation

Local:

- `tools/remote_test.ps1 check`: passed.
- Python compile for the panel and start bridge: passed.
- PowerShell parser and `git diff --check`: passed.

Remote:

- Desktop entry installed under `/home/ysxq/桌面`.
- Panel process restarted successfully on `DISPLAY=:1`.
- Active GUI PID after deployment: `3742547`.
- `tools/remote_test.ps1 check-startup-panel`: passed.
- Dry-run state: `dry_run_ok`.
- Dry-run launch argv: `--event-source-mode live_hal --mvs-source-mode live_camera`.

## Validation boundary

The GUI, structured argument contract, shell syntax, desktop launch, and
managed-start dry-run are verified. This node did not automatically press the
start button or launch the physical full system, to avoid changing the remote
runtime without an operator decision. The first operator-triggered live start
must confirm `v25_panel_launch_status.state=ready`, a nonzero `launcher_pid`,
and a valid `run_dir` before this path is treated as field-proven.

Replay-mode one-click startup still relies on the dataset manifest capability
and retains the existing V25 full-replay synchronization limitations.

## Rollback

Remove `MRG-Startup-Control-Panel.desktop`, stop only the panel GUI process,
and restore the previous `remote_ops/start_latest_system.sh` and V25 control
panel files. The existing `MRG-OneClick-Start.desktop` and
`MRG-OneClick-Stop.desktop` paths remain independent fallbacks.
