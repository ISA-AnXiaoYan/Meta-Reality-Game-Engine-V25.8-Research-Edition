# G0-G2 Repository Governance Node

Date: 2026-07-10

Branch: `v24-a-bev-event-sync-truth`

## Objective

Establish a controlled repository baseline without changing production runtime
entrypoints or hardware behavior.

## G0 baseline

- Startup-control-panel feature closed in commit `825dc0d`.
- Remote desktop GUI and managed-start dry-run passed.
- Physical full-system start from the new GUI remains an explicit field
  validation boundary.

## G1 guardrails

- Added `governance/entrypoints.json` with 22 canonical runtime/operations roles.
- Added `governance/file_ownership.json`.
- Added an existing-debt-only baseline in `governance/hygiene_baseline.json`.
- Added `tools/check_repo_hygiene.py` to `tools/dev_check.ps1`.
- Added GitHub Actions repository hygiene check.
- Expanded `.gitignore` for runtime output, recordings, archives, models,
  profiler output, local config, and credentials.
- Expanded `.gitattributes` to enforce LF for Linux-facing source and CRLF for
  PowerShell.

G1 commit: `cd82358`.

## G2 low-risk organization

- Moved node/stage documents to `docs/versions/`.
- Moved test reports and historical comparisons to `docs/reports/`.
- Moved stage architecture and operations documents into dedicated folders.
- Moved explicit `before` configuration/source snapshots to `archive/`.
- Removed 16 zero-byte snapshot PNG files.
- Removed eight zero-byte, unreferenced placeholder files from the root.
- Updated known living-document and manifest references.

No canonical production Python entrypoint, launch profile, `remote_ops` script,
native source, or active configuration was moved in G2.

## Metrics

| Metric | Before G1/G2 | After G2 |
| --- | ---: | ---: |
| Root tracked files | 154 | 99 |
| Root Python files | 66 | 60 |
| Root Markdown files | 67 | 23 |
| Zero-byte tracked files | 28 | 4 |
| Exact duplicate groups | 11 | 11 |
| Flat fallback migration files | 56 | 50 |
| `/home/ysxq/` absolute path occurrences | 180 | 180 |

The unchanged duplicate and absolute-path counts are intentional G3/G4 debt;
they were not mixed into the low-risk G2 move.

## Validation

- `tools/check_repo_hygiene.py`: required, expected state `ok`.
- `tools/dev_check.ps1`: required.
- JSON/profile parsing: required.
- Python compile of selected production entrypoints: required.
- `git diff --check`: required.
- Remote startup-panel dry-run: passed on 2026-07-10 with `state=dry_run_ok`,
  `mode=live`, `event_source_mode=live_hal`, and
  `mvs_source_mode=live_camera`. The check did not start hardware or the full
  system.

## Risks and boundaries

- 11 exact duplicate groups still exist and can still diverge until G3.
- 50 root/fallback files remain migration debt.
- Four zero-byte legacy files remain under `rentijiance/`; they are baseline
  debt and cannot increase.
- 180 machine-specific absolute-path occurrences remain configuration debt.
- The launcher still has multi-candidate compatibility paths. G3 must not
  delete aliases before wrappers and replay/live regression evidence exist.
