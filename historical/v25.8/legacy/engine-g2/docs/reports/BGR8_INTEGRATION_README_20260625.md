# 625_old 接入 BGR8_Packed 直通链路说明

此补丁以 `625_old.zip` 为基线，仅接入已经在 `625_bgr8.zip` 中验证通过的 MVS BGR8 直通采集链路。

## 修改范围

只修改 MVS 图像格式采集与 `_convert_to_bgr()` 相关代码：

- `hik_rgb_yolo_seg_multicam_same_model_stableid_v5.py`
- `mvs_softsync_capture_core.py`
- `cpp_bullet_core/hik_rgb_yolo_seg_multicam_same_model_stableid_v5.py`
- `cpp_bullet_core/mvs_softsync_capture_core.py`
- `cpp_bullet_core/rentijiance/hik_rgb_yolo_seg_multicam_same_model_stableid_v5.py`
- `rentijiance/hik_rgb_yolo_seg_multicam_same_model_stableid_v5.py`
- `ids_8cam_fusion_config.json`
- `hik_rgb_yolo_seg_multicam_same_model_stableid_v5.json`
- `rentijiance/hik_rgb_yolo_seg_multicam_same_model_stableid_v5.json`

没有修改：事件相机同步、softsync 时序、子弹轨迹检测、hit_judge、8765/8766 协议、UE5 overlay 数据结构。

## 当前链路

配置里所有 `prefer_bgr8` 已设置为 `true`，程序在 `StartGrabbing` 前请求海康相机输出 `PixelType_Gvsp_BGR8_Packed`。

每帧取到后：

```text
如果 frame_out.stFrameInfo.enPixelType == PixelType_Gvsp_BGR8_Packed:
    直接 ctypes.cast 指针安全读取 SDK buffer
    reshape(h, w, 3).copy()
    跳过 MV_CC_ConvertPixelType
    跳过 cv2.cvtColor
否则:
    保留原 MV_CC_ConvertPixelType 兜底
```

## 生效日志

运行后看到下面日志，说明正在走 BGR8 直通：

```text
[HIK][BGR8_FAST] frame is already BGR8_Packed; skip MV_CC_ConvertPixelType/cvtColor
```

如果看到 `[SDK_CONVERT]`，说明该路实际没有返回 BGR8，代码会自动走旧转换兜底。

## 复制方式

在项目根目录执行：

```bash
rsync -av --backup --suffix=.before_bgr8_integrated /path/to/625_old_bgr8_integrated_changed/625_old/ ./
```
