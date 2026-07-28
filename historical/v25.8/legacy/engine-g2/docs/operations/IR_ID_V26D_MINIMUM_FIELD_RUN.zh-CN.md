# IR-ID V26-D 最小现场实测

状态：代码、离线回归和真实场地 RAW sidecar 回放已通过；尚未在当前在线主工作区启用。

## 边界

- 运行位置：`patch/v26-d-ir-id-legacy-sidecar` 的 legacy Event runtime。
- Hook 位置：Event worker 从同一路 EVB FIFO 取到 `raw_evs` 后、human mask/STC/spatter/line filter/bullet 之前。
- 行为：只截取指定 ROI，非阻塞投递给后台 decoder，并写 observation/status/perf sidecar。
- 不修改 HAL broker；不增加第二个 FIFO reader；不向 Global Person ID、`identity_evidence`、命中裁决或正式权威输出写入任何数据。
- 默认关闭。只有同时指定 `--active-led-marker-enable`、唯一相机 alias、ROI、run ID 时才允许启用。

## 已验证事实

- 本分支 IR-ID 回归：10 项通过，包括任意 1/4/8/12 ms 切片边界、EVP 回放、default-off、单一 raw tap、无 HAL marker 逻辑、异步队列与现场验收器。
- 协议 synthetic ID=7 自检通过；`ir-id-blink-v1` observation schema 为 2。
- 现场历史 G 路录制 `recordings/20260710_163327/EVENT_G_4110035688/event_slices.evp` 共 14,966 个 4 ms slice、1,941 个事件；全画幅及候选 ROI 都得到 0 pulse、0 observation。它不能作为硬件可检测证据。
- 用户提供的真实场地 RAW `recording_2026-07-10_17-22-32.raw` 已在现场机 `/usr/bin/python3` 回放。ROI `648,480,48,48` 的 V26-D sidecar 在 39.95 秒、9,987 个 4 ms slice 中输出 139 条 schema v2 observation，全部为 ID=7；置信度中位数 0.935，最低 0.822，位置稳定在约 `(680.65, 513.86)`。
- 该 RAW 的 64 深度队列回放有 291 个 drop；512 深度有界队列处理全部 9,987 个 slice、0 drop、0 sidecar error，40 个 perf 样本的最大 `process_ms_p95` 为 0.152 ms。现场命令因此使用 512。
- 历史 `IRID_20260710_171351_G_B01_H150_B16_FULL` 仍只有 manifest，不能把其 `brightness=16/full8x8` 与上述 RAW 做配置审计绑定。但“是否可被 EventCD 可靠识别”的独立高亮纯色验证目标已由真实 RAW 满足，可取消重复实验。full8x8 仅可用作短时可见性诊断，不能成为最终头戴方案。
- 现场机当前不能从 192.168.31 网段访问 ESP SoftAP 默认地址 `http://10.10.10.1`。必须由接入 `ESP32-S3-Matrix-LEDID-62F2A0` 的操作终端先确认模块在线并启动发码。

## 现场前置条件

1. 使用接入 ESP SoftAP 的终端执行：

```bash
python tools/active_led_marker_poc/esp32_marker_control.py --base-url http://10.10.10.1 status
python tools/active_led_marker_poc/esp32_marker_control.py --base-url http://10.10.10.1 set player_id=7 alpha_ms=20 pulse_width_ms=10 brightness=16 mask=center3x3
python tools/active_led_marker_poc/esp32_marker_control.py --base-url http://10.10.10.1 start
```

2. 在启动 IR-ID 前，由当班人员把命中裁决保持 safe-off；IR-ID 本身不会改变裁决，但现场最小测试不应承担生产判定。
3. 从新的 G 路 event-slice 录制生成 ROI 热图。不得复用 7 月 10 日的候选 ROI 作为最终 ROI；若 10 秒内没有明显局部热点，停止本轮并先处理硬件可见性。
4. ROI 必须只覆盖 LED 预期像素区域，建议从 48x48 起；不得用全画幅开启 sidecar。

## 启动命令

在已应用本分支且与现场 runtime 基线一致的测试快照中执行。`G`、ROI 和 run ID 是示例，必须替换为本轮实测值。

```bash
RUN_ID=IRID_V26D_G_$(date +%Y%m%d_%H%M%S)
LAUNCH_EXTRA_ARGS="--active-led-marker-enable --active-led-marker-cameras G --active-led-marker-roi 648,480,48,48 --active-led-marker-run-id ${RUN_ID} --active-led-marker-alpha-ms 20 --active-led-marker-bit-count 8 --active-led-marker-bin-us 1000 --active-led-marker-min-events-per-bin 8 --active-led-marker-min-segment-events 20 --active-led-marker-queue-size 512" \
PROFILE=launch_profiles/627_event_overlay_bev.json \
bash remote_ops/run_fullsys_test.sh "${RUN_ID}" 120
```

现场机的当前主工作区含大量未提交文件，不能直接覆盖 `launch_fusion_system.py` 或 Event worker。先在与该主机版本一致的测试快照应用本分支，或由 runtime owner 做三方合并；禁止用覆盖拷贝部署。已通过的是 recorded-field replay，不是对当前在线主进程的热覆盖验证。

## 验收与回滚

运行结束后，检查以下三份 G 路产物：

```bash
python validate_ir_id_field_run.py \
  --observation sync_ipc/active_marker_observation_G.jsonl \
  --status sync_ipc/active_marker_status_G.json \
  --perf sync_ipc/active_marker_perf_G.csv \
  --expected-id 7 --run-id "${RUN_ID}" \
  --min-observations 3 --max-false-ids 0 --max-queue-dropped 0 --max-p95-ms 2.0 \
  --output sync_ipc/ir_id_runs/${RUN_ID}/v26d_field_validation.json
```

通过条件：至少 3 条 schema v2、同 run ID 的 ID=7 observation；无错误 ID、无 queue drop、status 无错误，并且每个 perf 样本的 decoder `process_ms_p95` 不超过 2 ms。任一条件不满足即为“不通过”，不得接入 `identity_evidence`。

回滚只需停止本轮测试并在下次启动时去掉 `--active-led-marker-enable`；sidecar 不会保留到 Event worker 主链。保留 observation、status、perf、EVP/RAW 与本验证 JSON 作为失败证据。
