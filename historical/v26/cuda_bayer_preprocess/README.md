# cuda_bayer_preprocess

YOLO raw Bayer GPU preprocessing module.

Runtime path:

```text
Bayer8 raw [B,H,W] uint8 on CPU
-> upload to CUDA
-> custom kernel: Bayer demosaic + letterbox + normalize + RGB NCHW
-> YOLO model.predict(source=cuda_tensor)
```

The extension is JIT-built on first import through `torch.utils.cpp_extension.load`.
Make sure `CUDA_HOME`, `PATH` and `LD_LIBRARY_PATH` point to a CUDA Toolkit with `nvcc`.
