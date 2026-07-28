#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python3 fusion_renderer_gl.py --config hik_rgb_yolo_seg_multicam_same_model_stableid_v5.json --cams A,B,C,D,E,F --preview-fps 120
