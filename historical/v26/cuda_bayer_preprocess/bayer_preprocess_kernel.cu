#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// pattern: 0=RG, 1=BG, 2=GR, 3=GB. color: 0=R,1=G,2=B.
__device__ __forceinline__ int bayer_color(int y, int x, int pattern) {
    int py = y & 1;
    int px = x & 1;
    if (pattern == 0) { // RGGB
        if (py == 0 && px == 0) return 0;
        if (py == 1 && px == 1) return 2;
        return 1;
    }
    if (pattern == 1) { // BGGR
        if (py == 0 && px == 0) return 2;
        if (py == 1 && px == 1) return 0;
        return 1;
    }
    if (pattern == 2) { // GRBG
        if (py == 0 && px == 1) return 0;
        if (py == 1 && px == 0) return 2;
        return 1;
    }
    // GBRG
    if (py == 1 && px == 0) return 0;
    if (py == 0 && px == 1) return 2;
    return 1;
}

__device__ __forceinline__ unsigned char raw_at(const unsigned char* raw, int b, int h, int w, int y, int x) {
    y = max(0, min(h - 1, y));
    x = max(0, min(w - 1, x));
    return raw[(b * h + y) * w + x];
}

__device__ __forceinline__ void demosaic_pixel(const unsigned char* raw, int b, int h, int w, int y, int x, int pattern, float& r, float& g, float& bl) {
    int c = bayer_color(y, x, pattern);
    float v = (float)raw_at(raw, b, h, w, y, x);
    if (c == 0) { // R site
        r = v;
        g = 0.25f * ((float)raw_at(raw,b,h,w,y-1,x) + raw_at(raw,b,h,w,y+1,x) + raw_at(raw,b,h,w,y,x-1) + raw_at(raw,b,h,w,y,x+1));
        bl = 0.25f * ((float)raw_at(raw,b,h,w,y-1,x-1) + raw_at(raw,b,h,w,y-1,x+1) + raw_at(raw,b,h,w,y+1,x-1) + raw_at(raw,b,h,w,y+1,x+1));
    } else if (c == 2) { // B site
        bl = v;
        g = 0.25f * ((float)raw_at(raw,b,h,w,y-1,x) + raw_at(raw,b,h,w,y+1,x) + raw_at(raw,b,h,w,y,x-1) + raw_at(raw,b,h,w,y,x+1));
        r = 0.25f * ((float)raw_at(raw,b,h,w,y-1,x-1) + raw_at(raw,b,h,w,y-1,x+1) + raw_at(raw,b,h,w,y+1,x-1) + raw_at(raw,b,h,w,y+1,x+1));
    } else { // G site
        g = v;
        int left_c = bayer_color(y, max(0, x-1), pattern);
        int right_c = bayer_color(y, min(w-1, x+1), pattern);
        if (left_c == 0 || right_c == 0) { // R horizontally, B vertically
            r = 0.5f * ((float)raw_at(raw,b,h,w,y,x-1) + raw_at(raw,b,h,w,y,x+1));
            bl = 0.5f * ((float)raw_at(raw,b,h,w,y-1,x) + raw_at(raw,b,h,w,y+1,x));
        } else {
            bl = 0.5f * ((float)raw_at(raw,b,h,w,y,x-1) + raw_at(raw,b,h,w,y,x+1));
            r = 0.5f * ((float)raw_at(raw,b,h,w,y-1,x) + raw_at(raw,b,h,w,y+1,x));
        }
    }
}

__device__ __forceinline__ void sample_rgb_bilinear(const unsigned char* raw, int b, int h, int w, float fy, float fx, int pattern, float& r, float& g, float& bl) {
    fx = fminf(fmaxf(fx, 0.0f), (float)(w - 1));
    fy = fminf(fmaxf(fy, 0.0f), (float)(h - 1));
    int x0 = (int)floorf(fx);
    int y0 = (int)floorf(fy);
    int x1 = min(w - 1, x0 + 1);
    int y1 = min(h - 1, y0 + 1);
    float ax = fx - x0;
    float ay = fy - y0;
    float r00,g00,b00,r01,g01,b01,r10,g10,b10,r11,g11,b11;
    demosaic_pixel(raw,b,h,w,y0,x0,pattern,r00,g00,b00);
    demosaic_pixel(raw,b,h,w,y0,x1,pattern,r01,g01,b01);
    demosaic_pixel(raw,b,h,w,y1,x0,pattern,r10,g10,b10);
    demosaic_pixel(raw,b,h,w,y1,x1,pattern,r11,g11,b11);
    float w00=(1-ax)*(1-ay), w01=ax*(1-ay), w10=(1-ax)*ay, w11=ax*ay;
    r = r00*w00 + r01*w01 + r10*w10 + r11*w11;
    g = g00*w00 + g01*w01 + g10*w10 + g11*w11;
    bl = b00*w00 + b01*w01 + b10*w10 + b11*w11;
}

template <typename scalar_t>
__global__ void bayer_kernel(const unsigned char* __restrict__ raw, scalar_t* __restrict__ out, int B, int H, int W, int OH, int OW, int pattern, float pad_norm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * OH * OW;
    if (idx >= total) return;
    int ox = idx % OW;
    int oy = (idx / OW) % OH;
    int b = idx / (OH * OW);
    float scale = fminf((float)OW / (float)W, (float)OH / (float)H);
    int resized_w = (int)roundf((float)W * scale);
    int resized_h = (int)roundf((float)H * scale);
    float pad_x = 0.5f * ((float)OW - (float)resized_w);
    float pad_y = 0.5f * ((float)OH - (float)resized_h);
    float r = pad_norm, g = pad_norm, bl = pad_norm;
    if ((float)ox >= pad_x && (float)ox < pad_x + resized_w && (float)oy >= pad_y && (float)oy < pad_y + resized_h) {
        float sx = ((float)ox - pad_x + 0.5f) / scale - 0.5f;
        float sy = ((float)oy - pad_y + 0.5f) / scale - 0.5f;
        float rr, gg, bb;
        sample_rgb_bilinear(raw, b, H, W, sy, sx, pattern, rr, gg, bb);
        r = rr / 255.0f;
        g = gg / 255.0f;
        bl = bb / 255.0f;
    }
    int plane = OH * OW;
    int base = b * 3 * plane + oy * OW + ox;
    out[base] = (scalar_t)r;
    out[base + plane] = (scalar_t)g;
    out[base + 2 * plane] = (scalar_t)bl;
}

torch::Tensor bayer_preprocess_cuda(torch::Tensor raw, int64_t out_h, int64_t out_w, int64_t pattern, bool use_half, double pad_norm) {
    int B = (int)raw.size(0);
    int H = (int)raw.size(1);
    int W = (int)raw.size(2);
    auto options = raw.options().dtype(use_half ? torch::kFloat16 : torch::kFloat32);
    auto out = torch::empty({B, 3, (int)out_h, (int)out_w}, options);
    int total = B * (int)out_h * (int)out_w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (use_half) {
        bayer_kernel<at::Half><<<blocks, threads, 0, stream>>>(raw.data_ptr<unsigned char>(), out.data_ptr<at::Half>(), B, H, W, (int)out_h, (int)out_w, (int)pattern, (float)pad_norm);
    } else {
        bayer_kernel<float><<<blocks, threads, 0, stream>>>(raw.data_ptr<unsigned char>(), out.data_ptr<float>(), B, H, W, (int)out_h, (int)out_w, (int)pattern, (float)pad_norm);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
