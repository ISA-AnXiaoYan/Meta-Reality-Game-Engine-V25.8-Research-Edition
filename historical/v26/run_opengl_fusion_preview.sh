#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Start only OpenGL through the unified launcher so the generated runtime MVS config is used.
python3 launch_fusion_system.py --config ids_5cam_fusion_config.json --cams A,B,C,D,E --skip-event --skip-mvs --skip-hit --preview-fps 60 "$@"
