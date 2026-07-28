# 8 路 BGR8 代码层面强制版说明

本版本在 Python 采集入口里直接请求海康 MVS 相机输出 `PixelType_Gvsp_BGR8_Packed`，不再只依赖 JSON 的 `prefer_bgr8`。

## 没有改动的链路

- 没有改 C++ 版本
- 没有改同步逻辑、soft trigger、strobe、frame bundle、event trigger CSV
- 没有改子弹轨迹检测、hit_judge_server、8765/8766 协议
- 没有改 YOLO/OSNet 的检测和 ID 逻辑

## 新链路

```text
MV_CC_OpenDevice
→ 设置 ROI/FPS/曝光/同步原有逻辑
→ 代码层面强制 MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_BGR8_Packed)
→ MV_CC_StartGrabbing
→ 每帧检查 stFrameInfo.enPixelType
   - 如果实际是 BGR8_Packed：直接 numpy reshape + copy，不 ConvertPixelType，不 cvtColor
   - 如果不是 BGR8_Packed：保留旧 MV_CC_ConvertPixelType 兜底
```

## 关键日志

设置相机端格式成功时会看到：

```text
[BGR8_FORCE] set PixelFormat=PixelType_Gvsp_BGR8_Packed before StartGrabbing
```

真正每帧走到零转换路径时会看到：

```text
[BGR8_FAST] direct BGR8_Packed path enabled ... skip MV_CC_ConvertPixelType/cvtColor
```

如果看到：

```text
[SDK_CONVERT] ... using MV_CC_ConvertPixelType -> BGR8 fallback
```

说明某一路相机没有实际返回 BGR8_Packed，代码会自动走旧转换，不会崩。

## 备份

被修改文件旁边保留 `.bak_before_force_bgr8_codelevel` 备份。
