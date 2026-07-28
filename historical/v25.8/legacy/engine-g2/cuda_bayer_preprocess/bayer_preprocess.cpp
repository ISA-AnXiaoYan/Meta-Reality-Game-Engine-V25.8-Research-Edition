#include <torch/extension.h>

torch::Tensor bayer_preprocess_cuda(torch::Tensor raw, int64_t out_h, int64_t out_w, int64_t pattern, bool use_half, double pad_norm);

torch::Tensor bayer_preprocess(torch::Tensor raw, int64_t out_h, int64_t out_w, int64_t pattern, bool use_half, double pad_norm) {
    TORCH_CHECK(raw.is_cuda(), "raw must be a CUDA tensor");
    TORCH_CHECK(raw.dtype() == torch::kUInt8, "raw must be uint8");
    TORCH_CHECK(raw.dim() == 3, "raw must have shape [B,H,W]");
    return bayer_preprocess_cuda(raw.contiguous(), out_h, out_w, pattern, use_half, pad_norm);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bayer_preprocess", &bayer_preprocess, "Bayer8 to RGB letterbox normalized NCHW (CUDA)");
}
