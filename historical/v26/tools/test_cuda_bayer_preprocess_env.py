#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from hik_rgb_yolo_seg_multicam_same_model_stableid_v5 import RawBayerFrame
from cuda_bayer_preprocess import preprocess_bayer_batch_gpu


def main():
    import torch
    print('torch:', torch.__version__)
    print('cuda available:', torch.cuda.is_available())
    print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')
    h, w = 480, 640
    # synthetic Bayer-like image
    raw = np.zeros((h, w), dtype=np.uint8)
    raw[:, :] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    item = RawBayerFrame(raw=raw, ts_us=0, fid=1, mvs_frame_num=1, width=w, height=h, pixel_type=0, bayer_pattern='BG', mvs_meta={})
    y, metas = preprocess_bayer_batch_gpu([item], imgsz=640, bayer_pattern='BG', device='cuda:0', dtype='fp16')
    print('output:', y.shape, y.dtype, y.device, 'minmax=', float(y.min()), float(y.max()))
    print('meta:', metas[0])

if __name__ == '__main__':
    main()
