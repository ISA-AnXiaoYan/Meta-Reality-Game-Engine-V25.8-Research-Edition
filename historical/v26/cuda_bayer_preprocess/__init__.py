# -*- coding: utf-8 -*-
"""Bayer8 -> YOLO tensor CUDA preprocessing.

This module builds a small PyTorch CUDA extension on first import.  It keeps the
YOLO path on GPU: Bayer8 batch -> RGB letterbox -> normalized NCHW tensor.
"""
from .preprocess import preprocess_bayer_batch_gpu

__all__ = ["preprocess_bayer_batch_gpu"]
