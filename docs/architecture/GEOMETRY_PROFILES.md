# Geometry profiles

<!-- SPDX-License-Identifier: AGPL-3.0-only -->

几何算法不从 `camera_id`、数组顺序或默认矩阵推断传感器朝向。所有投影都必须绑定 `GeometryProfile`、`SensorPose`、`world_frame_id`、`anchor_mode` 和可选 `calibration_id`。

当前提供：

- `foot_point`：以玩家脚底/接触点作为定位锚点；
- `sensor_center`：以传感器中心作为几何参考；
- `custom_anchor`：由外部 profile 声明锚点；
- `four_sides_inward`：四周传感器朝中心观察；
- `parallel_wall`：平行墙面传感器布局。

传感器模组改为“四周往中心看”时，应新增或选择对应 profile，并提供坐标变换回放和对照测试；不得直接改写默认 profile 或将北/东/南/西的数组下标当作语义。
