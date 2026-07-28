#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/cpp_bullet_core"
python3 setup.py build_ext --inplace
python3 - <<'PY'
import _cpp_bullet_full_port_stage06 as m
cfg = m.TrackConfig()
assert hasattr(cfg, 'bullet_id_bounce_link_enabled'), 'C++ extension is old: missing bullet_id_bounce_link_enabled'
print('[OK] C++ extension rebuilt with bullet_id_bounce_link_enabled')
PY
