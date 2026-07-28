# BGR8 safe recovery patch

这个补丁针对 8 路 BGR8 代码层面强制后出现 `GetImageBuffer timeout/fail in receiver` 的情况。

改动：
- 保留 `_convert_to_bgr()` 的 BGR8 fast path：如果实际帧是 `PixelType_Gvsp_BGR8_Packed`，仍跳过 `MV_CC_ConvertPixelType/cvtColor`。
- 取消所有相机无条件强制 `PixelFormat=BGR8_Packed`。
- 只有当某一路 JSON 显式 `prefer_bgr8=true` 时，才在 `StartGrabbing` 前请求 BGR8。
- `ids_5cam_fusion_config.json` 恢复到 `active_profile=hik_6cam` 且 `prefer_bgr8=false`，方便先恢复出图。
- 新增 `ids_8cam_fusion_config_bgr8_safe_recovery.json`，默认 8 路布局但不强制 BGR8。

建议先确认恢复出图，再逐路打开 prefer_bgr8。
