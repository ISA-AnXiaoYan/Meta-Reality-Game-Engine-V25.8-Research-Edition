# V1.1 节点：事件侧 strict mask gate + TSF1 12ms

定义时间：2026-07-01 01:55 CST

## 节点结论

V1.1 定义为当前稳定节点：事件 worker 侧人体 mask 同步缓冲采用 strict same-sync gate，TSF1 周期为 12ms，并修复 V11 中同 sync mask 到达后绕过 hard timeout 的问题。

本节点可作为下一阶段“事件相机像素膨胀层按场地/物件坐标融合到八路 MVS 拼接视图”的开发基线。

## 修改范围

- `metavision_spatter_tracking_linefilter_tsf1_autosender_backfill.py`
  - `HumanMaskSyncEventBuffer.pop_ready()` 中 strict 模式先判断 `due_timeout`。
  - 超过 `human_mask_buffer_timeout_ms` 的 bucket 直接 `mask_timeout_drop`，不再因为同 sync mask 后到而释放旧事件切片。

保留上一轮 V11 配置：

- `tsf1_pub_fps = 83.3333333333`
- `event_tsf1_pub_max_fps = 83.3333333333`
- `tsf1_pub_accumulation_time_us = 12000`
- `event_viz_build_max_fps = 80.0`
- `event_overlay_pub_max_fps = 40.0`
- `strict_same_sync_drop`
- `raw_fallback_enabled = false`

## 备份

- 本地修改前备份：`C:\Users\axyis\Documents\判定开发\项目代码\_local_backup_before_v11_1_20260701_012543`
- 远端修改前备份：`/home/ysxq/PycharmProjects/Ids_Test_3.9/test622yolo_gpu_627/sync_ipc/code_backups/pre_v11_1_strict_timeout_20260701_012547`

## 远端实测

测试归档：

- `sync_ipc/full_system_tests/fullsys_5min_v11_1_strict_timeout_tsf12_20260701_012757_display1_keepstdin.tar.gz`
- `cpu_hotspots_20260701_012821.tar.gz`

本地归档副本：

- `C:\Users\axyis\Documents\判定开发\remote_tests\fullsys_5min_v11_1_strict_timeout_tsf12_20260701_012757_display1_keepstdin.tar.gz`
- `C:\Users\axyis\Documents\判定开发\remote_tests\cpu_hotspots_20260701_012821.tar.gz`

测试结果：

- `launcher_rc = 124`，timeout 正常结束。
- OBS/SRT 稳定：`start ffmpeg = 8`，`restart_backoff = 0`，`Address already in use = 0`，`Broken pipe = 0`。
- 8 路事件 worker `raw_fallback_enabled = false`。
- 本次 run 窗口内 `buffer_slice_age_ms`：
  - A：p50 146.0ms，p95 233.9ms，max 250.0ms
  - B：p50 111.1ms，p95 234.6ms，max 249.9ms
  - C：p50 178.4ms，p95 238.8ms，max 250.0ms
  - D：p50 176.7ms，p95 239.5ms，max 249.9ms
  - E：p50 228.4ms，p95 244.1ms，max 250.0ms
  - F：p50 231.9ms，p95 244.1ms，max 250.0ms
  - G：p50 174.5ms，p95 240.1ms，max 249.9ms
  - H：p50 184.0ms，p95 241.6ms，max 250.0ms
- 本轮未再出现 V11 的 E/F 秒级旧切片释放。
- `system_status_latest.json` 已能显示真实链路分类：MVS、Event、Global YOLO、Global Person ID、Global BEV、Bullet reprocess、Preview。

## 已知风险

- hard timeout 将延迟上限压住，但会以 `mask_timeout_drop` 形式丢弃无法等到同 sync mask 的事件切片；这是当前方案用命中率换确定性延迟的代价。
- TSF1 12ms/约 83Hz 增加事件 worker CPU 压力，热点主要仍在 Global YOLO、Global BEV、Preview 和高事件量 worker。
- 本次远端测试 `hit_candidate_all.jsonl = 0`，说明没有完成真实弹道命中样本验证；弹道识别/命中准确性仍需带弹道输入的专项测试。
- 当前人体 mask filter 统计中多数路为 `empty_current_mask`，只有 G 路观察到少量实际 polygon 过滤；mask 有效性仍依赖现场人体位置、YOLO polygon 输出和相机视场。

