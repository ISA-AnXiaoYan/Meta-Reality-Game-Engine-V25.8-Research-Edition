# Inward four-sensor geometry variant

<!-- SPDX-License-Identifier: AGPL-3.0-only -->

- Variant ID: `geometry.inward_four_sensor.footpoint.v1`
- Source/provenance: clean-room profile and transform implementation in `src/mrge/geometry/`
- Hypothesis: four sensors look inward and player location is represented by a foot-point anchor
- Geometry profile: `research/geometry_profiles/inward_four_sensor_footpoint_v1.json`
- Inputs: synthetic sensor poses and profile-bound local points
- Outputs: transformed research coordinates and candidate preview facts
- Deterministic checks: `tests/test_geometry_profiles.py`
- Evidence: local CPU contract checks
- Failure modes: missing calibration, wrong handedness, stale profile, inconsistent world frame
- Applicable conditions: only when the declared profile matches the physical layout
- Not qualified for: real-camera calibration, hit authority, production game state
- Authority: `research-only`
- Owner: MRGE research maintainers
- Last reviewed: 2026-07-28
