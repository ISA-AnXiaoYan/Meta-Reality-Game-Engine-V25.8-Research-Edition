#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 launch_fusion_system.py --config ids_5cam_fusion_config.json --cams A,B,C,D,E --skip-event --skip-mvs --skip-gl --preview-fps 60 "$@"
