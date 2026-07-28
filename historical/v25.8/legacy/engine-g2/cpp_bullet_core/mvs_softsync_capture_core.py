#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mvs_softsync_capture_core.py

海康机器人 MVS 可见光相机软同步采集核心模块。

用途：
  1) 统一打开多台海康 MVS 相机；
  2) 将相机配置为 Software Trigger；
  3) 每个 trigger_index 先调用外部 IO/采集卡脉冲回调，再并行软触发多台 MVS；
  4) 并行取回每台相机的 BGR 帧和同步 metadata；
  5) 后续由 publisher 模块把帧写入 shared memory、把 metadata 写入 json/csv。

重要边界：
  - 本文件不直接依赖具体采集卡/IO SDK。
  - 如果需要“曝光前给事件相机发脉冲”，请在上层传入 event_pulse_callback。
  - 本文件只负责 MVS 相机控制和同步触发/取帧，不做 YOLO、不做 OBS、不做事件相机处理。

建议运行架构：
  mvs_softsync_frame_publisher.py
      -> import 本文件
      -> MvsSoftSyncGroup.open_all()
      -> 循环 group.trigger_once(trigger_index, event_pulse_callback=...)
      -> 将返回的 SyncedMvsFrame 写入 /dev/shm 和 meta json

注意：软件触发不是硬件级完全同时；本模块会记录每台相机软件触发命令开始/结束时间，
      便于你评估 A/B 软触发抖动。
"""

from __future__ import annotations

import ctypes
import importlib
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# =============================================================================
# 时间工具
# =============================================================================


def now_us() -> int:
    """Unix wall time, microseconds."""
    return int(time.time() * 1_000_000)


def perf_us() -> int:
    """Monotonic performance timer, microseconds. 只用于间隔统计，不用于跨进程时间戳。"""
    return time.perf_counter_ns() // 1000


def sleep_us(us: int) -> None:
    """尽量精确地等待若干微秒。

    Python/Windows/Linux 都不是实时系统；这里采用“大段 sleep + 最后短暂自旋”的方式，
    只用于减少 event pre-pulse 到 software trigger 之间的粗略误差。
    """
    us = int(us)
    if us <= 0:
        return
    end = time.perf_counter_ns() + us * 1000
    # 大于 2ms 的部分用 sleep，剩余约 0.3ms 自旋。
    while True:
        remain_ns = end - time.perf_counter_ns()
        if remain_ns <= 0:
            return
        if remain_ns > 2_000_000:
            time.sleep((remain_ns - 300_000) / 1e9)
        else:
            # short spin
            pass


# =============================================================================
# 数据结构
# =============================================================================


@dataclass
class HikCameraConfig:
    """单台海康 MVS 相机配置。"""

    camera_id: str
    serial: str
    mvs_sdk_path: str = "/opt/MVS/Samples/64/Python/MvImport"
    timeout_ms: int = 1000

    # 图像/曝光参数
    exposure_us: Optional[float] = 8000.0
    gain: Optional[float] = None
    exposure_auto_off: bool = True
    gain_auto_off: bool = True

    # ROI/传输参数
    force_full_sensor: bool = True
    capture_width: int = 0
    capture_height: int = 0
    reset_native_sampling: bool = True
    prefer_bgr8: bool = False
    optimize_gige_packet_size: bool = True

    # soft trigger 参数
    trigger_source: str = "Software"
    trigger_activation: str = "RisingEdge"

    # 可选：让某台 MVS 在曝光开始时从 Line 输出脉冲。
    # 当前 A 路径中，事件相机脉冲由采集卡/IO 发出，因此默认关闭。
    enable_strobe_output: bool = False
    strobe_line_selector: str = "Line1"
    strobe_line_mode: str = "Strobe"
    strobe_line_source: str = "ExposureStartActive"
    strobe_enable_feature: str = "StrobeEnable"
    strobe_duration_us: Optional[float] = 1000.0
    strobe_delay_us: Optional[float] = 0.0

    # 若某些节点在个别型号上不存在，是否将其视为致命错误。
    strict_feature_set: bool = False


@dataclass
class SoftSyncConfig:
    """多相机软同步组配置。"""

    fps: float = 25.0
    trigger_timeout_ms: int = 1000

    # 事件相机预脉冲：由上层传入 event_pulse_callback 实际实现。
    event_pre_pulse_enable: bool = True
    event_pre_pulse_lead_us: int = 2000
    event_pulse_width_us: int = 500

    # 并行触发/并行取帧
    parallel_trigger: bool = True
    parallel_grab: bool = True
    max_workers: int = 8

    # 警告阈值：同一次 trigger_index 内，多台相机 TriggerSoftware 命令开始时间跨度。
    trigger_spread_warn_us: int = 1500


@dataclass
class TriggerCommandResult:
    camera_id: str
    serial: str
    ret: int
    command_start_wall_us: int
    command_end_wall_us: int
    command_start_perf_us: int
    command_end_perf_us: int
    error: str = ""

    @property
    def duration_us(self) -> int:
        return int(self.command_end_perf_us - self.command_start_perf_us)


@dataclass
class FrameGrabResult:
    camera_id: str
    serial: str
    ok: bool
    ret: int
    frame: Optional[np.ndarray]
    frame_info: Dict[str, Any]
    grab_start_wall_us: int
    grab_end_wall_us: int
    grab_start_perf_us: int
    grab_end_perf_us: int
    error: str = ""

    @property
    def grab_duration_us(self) -> int:
        return int(self.grab_end_perf_us - self.grab_start_perf_us)


@dataclass
class SyncedMvsFrame:
    """一次 trigger_index 下某台相机对应的同步帧。"""

    camera_id: str
    camera_sn: str
    trigger_index: int
    sync_mode: str

    frame: Optional[np.ndarray]
    ok: bool

    # MVS 帧信息
    mvs_frame_num: Optional[int]
    width: Optional[int]
    height: Optional[int]
    pixel_type: Optional[int]
    frame_len: Optional[int]
    raw_frame_info: Dict[str, Any] = field(default_factory=dict)

    # 事件预脉冲信息
    event_pulse_enable: bool = False
    event_pulse_wall_us: Optional[int] = None
    event_pulse_end_wall_us: Optional[int] = None
    event_pulse_ret: Any = None
    event_pulse_error: str = ""
    event_pre_pulse_lead_us: int = 0
    event_pulse_width_us: int = 0

    # 软件触发命令信息
    trigger_cmd_wall_us: Optional[int] = None
    trigger_cmd_end_wall_us: Optional[int] = None
    trigger_ret: Optional[int] = None
    trigger_duration_us: Optional[int] = None
    trigger_error: str = ""

    # 取帧信息
    grab_wall_us: Optional[int] = None
    grab_end_wall_us: Optional[int] = None
    grab_ret: Optional[int] = None
    grab_duration_us: Optional[int] = None
    grab_error: str = ""

    # 事件相机时间戳暂由上层根据 event_trigger.csv 配对后填写。
    event_ts_us: Optional[int] = None
    sync_status: str = "softsync_no_event_ts"

    def meta_dict(self) -> Dict[str, Any]:
        """用于写入 meta json/csv 的轻量 metadata，不包含图像数组。"""
        return {
            "camera_id": self.camera_id,
            "camera_sn": self.camera_sn,
            "trigger_index": int(self.trigger_index),
            "sync_mode": self.sync_mode,
            "sync_status": self.sync_status,
            "ok": bool(self.ok),
            "mvs_frame_num": self.mvs_frame_num,
            "width": self.width,
            "height": self.height,
            "pixel_type": self.pixel_type,
            "frame_len": self.frame_len,
            "event_pulse_enable": self.event_pulse_enable,
            "event_pulse_wall_us": self.event_pulse_wall_us,
            "event_pulse_end_wall_us": self.event_pulse_end_wall_us,
            "event_pulse_ret": self.event_pulse_ret,
            "event_pulse_error": self.event_pulse_error,
            "event_pre_pulse_lead_us": self.event_pre_pulse_lead_us,
            "event_pulse_width_us": self.event_pulse_width_us,
            "trigger_cmd_wall_us": self.trigger_cmd_wall_us,
            "trigger_cmd_end_wall_us": self.trigger_cmd_end_wall_us,
            "trigger_ret": self.trigger_ret,
            "trigger_duration_us": self.trigger_duration_us,
            "trigger_error": self.trigger_error,
            "grab_wall_us": self.grab_wall_us,
            "grab_end_wall_us": self.grab_end_wall_us,
            "grab_ret": self.grab_ret,
            "grab_duration_us": self.grab_duration_us,
            "grab_error": self.grab_error,
            "event_ts_us": self.event_ts_us,
            "raw_frame_info": self.raw_frame_info,
        }


# event_pulse_callback(trigger_index, pulse_width_us) -> Any
EventPulseCallback = Callable[[int, int], Any]


# =============================================================================
# 海康 MVS 单相机封装
# =============================================================================


class MvsFeatureError(RuntimeError):
    pass


class HikMvsSoftTriggerCamera:
    """单台海康 MVS 相机：打开、配置软触发、软件触发、取 BGR 帧。"""

    def __init__(self, cfg: HikCameraConfig):
        self.cfg = cfg
        self.mvs: Any = None
        self.cam: Any = None
        self.camera_sn: str = str(cfg.serial or "").strip()
        self.opened = False
        self.grabbing = False

    # ------------------------------------------------------------------
    # 打开/关闭
    # ------------------------------------------------------------------

    def open(self) -> None:
        self._import_mvs()
        self._open_device_no_start()
        self._configure_before_start()
        self.start_grabbing()
        self.opened = True

    def close(self) -> None:
        if self.cam is None:
            return
        for fn in ["MV_CC_StopGrabbing", "MV_CC_CloseDevice", "MV_CC_DestroyHandle"]:
            try:
                getattr(self.cam, fn)()
            except Exception:
                pass
        self.grabbing = False
        self.opened = False
        self.cam = None

    def __enter__(self) -> "HikMvsSoftTriggerCamera":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _import_mvs(self) -> None:
        p = str(self.cfg.mvs_sdk_path or "").strip()
        if p and p not in sys.path:
            sys.path.append(p)
        try:
            self.mvs = importlib.import_module("MvCameraControl_class")
        except Exception as e:
            raise RuntimeError(
                f"导入海康 MVS Python wrapper 失败：{p}/MvCameraControl_class.py\n"
                f"原始错误: {type(e).__name__}: {e}"
            )

    @staticmethod
    def _decode_c_array(v: Any) -> str:
        try:
            b = bytes(v)
        except Exception:
            try:
                b = bytes(bytearray(v))
            except Exception:
                return ""
        return b.split(b"\x00", 1)[0].decode("utf-8", errors="ignore").strip()

    def _device_serial(self, info: Any) -> str:
        mvs = self.mvs
        try:
            if info.nTLayerType == getattr(mvs, "MV_GIGE_DEVICE"):
                return self._decode_c_array(info.SpecialInfo.stGigEInfo.chSerialNumber)
            if info.nTLayerType == getattr(mvs, "MV_USB_DEVICE"):
                return self._decode_c_array(info.SpecialInfo.stUsb3VInfo.chSerialNumber)
        except Exception:
            pass
        return ""

    def _open_device_no_start(self) -> None:
        mvs = self.mvs
        device_list = mvs.MV_CC_DEVICE_INFO_LIST()
        tlayer_type = getattr(mvs, "MV_GIGE_DEVICE") | getattr(mvs, "MV_USB_DEVICE")
        ret = mvs.MvCamera.MV_CC_EnumDevices(tlayer_type, device_list)
        if ret != 0 or int(device_list.nDeviceNum) <= 0:
            raise RuntimeError(f"未枚举到海康工业相机设备，ret=0x{ret:x}")

        target_serial = str(self.cfg.serial or "").strip()
        selected_idx = -1
        found: List[Tuple[int, str]] = []
        for i in range(int(device_list.nDeviceNum)):
            info = ctypes.cast(device_list.pDeviceInfo[i], ctypes.POINTER(mvs.MV_CC_DEVICE_INFO)).contents
            sn = self._device_serial(info)
            found.append((i, sn))
            if target_serial and sn == target_serial:
                selected_idx = i

        if selected_idx < 0:
            if target_serial:
                raise RuntimeError(f"没有找到指定序列号相机: {target_serial}，当前枚举到: {found}")
            selected_idx = 0

        info = ctypes.cast(device_list.pDeviceInfo[selected_idx], ctypes.POINTER(mvs.MV_CC_DEVICE_INFO)).contents
        sn = self._device_serial(info)
        self.camera_sn = sn or target_serial or str(selected_idx)
        print(f"[MVS][{self.cfg.camera_id}] selected index={selected_idx}, serial={self.camera_sn}")

        self.cam = mvs.MvCamera()
        ret = self.cam.MV_CC_CreateHandle(info)
        if ret != 0:
            raise RuntimeError(f"MV_CC_CreateHandle failed: 0x{ret:x}")
        access = getattr(mvs, "MV_ACCESS_Exclusive", 1)
        ret = self.cam.MV_CC_OpenDevice(access, 0)
        if ret != 0:
            try:
                self.cam.MV_CC_DestroyHandle()
            except Exception:
                pass
            raise RuntimeError(f"MV_CC_OpenDevice failed: 0x{ret:x}")

        # GigE 相机尽量设置最优包长，降低丢包概率。
        if self.cfg.optimize_gige_packet_size:
            self._try_optimize_gige_packet_size()

    def start_grabbing(self) -> None:
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise RuntimeError(f"MV_CC_StartGrabbing failed: 0x{ret:x}")
        self.grabbing = True

    # ------------------------------------------------------------------
    # Feature 设置工具
    # ------------------------------------------------------------------

    def _feature_warn_or_raise(self, msg: str, required: bool = False) -> None:
        if required or self.cfg.strict_feature_set:
            raise MvsFeatureError(msg)
        print(f"[MVS][{self.cfg.camera_id}][WARN] {msg}")

    def set_enum(self, name: str, value: str, required: bool = False) -> bool:
        """优先使用 MV_CC_SetEnumValueByString；失败时尝试 SDK 常量 fallback。"""
        # 1) ByString 是 GenICam 最清晰的写法。
        fn = getattr(self.cam, "MV_CC_SetEnumValueByString", None)
        if fn is not None:
            try:
                ret = fn(name, str(value))
                if ret == 0:
                    return True
                # 继续 fallback
                last_ret = ret
            except Exception as e:
                last_ret = None
                last_err = f"{type(e).__name__}: {e}"
            else:
                last_err = f"ret=0x{last_ret:x}"
        else:
            last_ret = None
            last_err = "MV_CC_SetEnumValueByString not found"

        # 2) 常用节点 fallback：优先 SDK 常量，其次少量通用枚举值。
        fallback = self._enum_fallback_value(name, value)
        if fallback is not None:
            try:
                ret = self.cam.MV_CC_SetEnumValue(name, int(fallback))
                if ret == 0:
                    print(f"[MVS][{self.cfg.camera_id}] set {name}={value} by numeric fallback {fallback}")
                    return True
                last_err = f"ByString failed, numeric fallback ret=0x{ret:x}"
            except Exception as e:
                last_err = f"ByString failed, numeric fallback error={type(e).__name__}: {e}"

        self._feature_warn_or_raise(f"set enum {name}={value} failed: {last_err}", required=required)
        return False

    def _enum_fallback_value(self, name: str, value: str) -> Optional[int]:
        mvs = self.mvs
        key = f"{name}:{value}".lower()

        # TriggerMode 常量在 MVS Python wrapper 里通常存在。
        if key == "triggermode:on":
            return getattr(mvs, "MV_TRIGGER_MODE_ON", 1)
        if key == "triggermode:off":
            return getattr(mvs, "MV_TRIGGER_MODE_OFF", 0)

        # TriggerSource 常量在部分 SDK 中存在；软件触发常用值通常为 7。
        if key == "triggersource:software":
            return getattr(mvs, "MV_TRIGGER_SOURCE_SOFTWARE", 7)
        if key == "triggersource:line0":
            return getattr(mvs, "MV_TRIGGER_SOURCE_LINE0", 0)
        if key == "triggersource:line1":
            return getattr(mvs, "MV_TRIGGER_SOURCE_LINE1", 1)
        if key == "triggersource:line2":
            return getattr(mvs, "MV_TRIGGER_SOURCE_LINE2", 2)
        if key == "triggersource:line3":
            return getattr(mvs, "MV_TRIGGER_SOURCE_LINE3", 3)

        # 常见开关型枚举。
        if value.lower() == "off":
            return 0
        if value.lower() == "on":
            return 1

        return None

    def set_bool(self, name: str, value: bool, required: bool = False) -> bool:
        try:
            ret = self.cam.MV_CC_SetBoolValue(name, bool(value))
            if ret == 0:
                return True
            self._feature_warn_or_raise(f"set bool {name}={value} failed: ret=0x{ret:x}", required=required)
        except Exception as e:
            self._feature_warn_or_raise(f"set bool {name}={value} failed: {type(e).__name__}: {e}", required=required)
        return False

    def set_float(self, name: str, value: float, required: bool = False) -> bool:
        try:
            ret = self.cam.MV_CC_SetFloatValue(name, float(value))
            if ret == 0:
                return True
            self._feature_warn_or_raise(f"set float {name}={value} failed: ret=0x{ret:x}", required=required)
        except Exception as e:
            self._feature_warn_or_raise(f"set float {name}={value} failed: {type(e).__name__}: {e}", required=required)
        return False

    def set_int(self, name: str, value: int, required: bool = False) -> bool:
        try:
            ret = self.cam.MV_CC_SetIntValue(name, int(value))
            if ret == 0:
                return True
            self._feature_warn_or_raise(f"set int {name}={value} failed: ret=0x{ret:x}", required=required)
        except Exception as e:
            self._feature_warn_or_raise(f"set int {name}={value} failed: {type(e).__name__}: {e}", required=required)
        return False

    def set_command(self, name: str, required: bool = False) -> int:
        try:
            ret = self.cam.MV_CC_SetCommandValue(name)
            if ret != 0:
                self._feature_warn_or_raise(f"set command {name} failed: ret=0x{ret:x}", required=required)
            return int(ret)
        except Exception as e:
            self._feature_warn_or_raise(f"set command {name} failed: {type(e).__name__}: {e}", required=required)
            return -1

    def _get_int_param(self, name: str) -> Optional[Any]:
        mvs = self.mvs
        for cls_name, fn_name in [
            ("MVCC_INTVALUE_EX", "MV_CC_GetIntValueEx"),
            ("MVCC_INTVALUE", "MV_CC_GetIntValue"),
        ]:
            cls = getattr(mvs, cls_name, None)
            fn = getattr(self.cam, fn_name, None)
            if cls is None or fn is None:
                continue
            try:
                param = cls()
                ctypes.memset(ctypes.byref(param), 0, ctypes.sizeof(param))
                ret = fn(name, param)
                if ret == 0:
                    return param
            except Exception:
                continue
        return None

    @staticmethod
    def _param_value(param: Any, names: Sequence[str], default: Optional[int] = None) -> Optional[int]:
        for n in names:
            if hasattr(param, n):
                try:
                    return int(getattr(param, n))
                except Exception:
                    pass
        return default

    def _set_int_min(self, name: str) -> None:
        p = self._get_int_param(name)
        if p is None:
            self.set_int(name, 0, required=False)
            return
        v = self._param_value(p, ["nMin", "nMinValue"], 0)
        if v is not None:
            self.set_int(name, int(v), required=False)

    def _set_int_max(self, name: str) -> None:
        p = self._get_int_param(name)
        if p is None:
            print(f"[MVS][{self.cfg.camera_id}][WARN] read {name}.Max failed, skip")
            return
        v = self._param_value(p, ["nMax", "nMaxValue"], None)
        if v is not None and int(v) > 0:
            self.set_int(name, int(v), required=False)

    def _try_optimize_gige_packet_size(self) -> None:
        fn = getattr(self.cam, "MV_CC_GetOptimalPacketSize", None)
        if fn is None:
            return
        try:
            opt = int(fn())
            if opt > 0:
                self.set_int("GevSCPSPacketSize", opt, required=False)
                print(f"[MVS][{self.cfg.camera_id}] GevSCPSPacketSize={opt}")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 相机配置
    # ------------------------------------------------------------------

    def _configure_before_start(self) -> None:
        cfg = self.cfg

        if cfg.reset_native_sampling:
            self.reset_native_sampling()

        if cfg.force_full_sensor and cfg.capture_width <= 0 and cfg.capture_height <= 0:
            self.restore_full_roi()
        else:
            if cfg.capture_width > 0:
                self.set_int("Width", cfg.capture_width, required=False)
            if cfg.capture_height > 0:
                self.set_int("Height", cfg.capture_height, required=False)

        self.configure_soft_trigger()

        if cfg.enable_strobe_output:
            self.configure_strobe_output()

        if cfg.prefer_bgr8:
            self.try_set_bgr8_pixel_format()

    def reset_native_sampling(self) -> None:
        for name in ["DecimationHorizontal", "DecimationVertical", "BinningHorizontal", "BinningVertical"]:
            # 有些型号是 Int，有些型号是 Enum；这里先 Int，失败不作为致命错误。
            ok = self.set_int(name, 1, required=False)
            if not ok:
                try:
                    self.cam.MV_CC_SetEnumValue(name, 1)
                except Exception:
                    pass

    def restore_full_roi(self) -> None:
        # 先 Offset 置最小，否则 Width/Height 可能无法设到最大。
        self._set_int_min("OffsetX")
        self._set_int_min("OffsetY")
        self._set_int_max("Width")
        self._set_int_max("Height")
        self._set_int_min("OffsetX")
        self._set_int_min("OffsetY")

    def configure_soft_trigger(self) -> None:
        cfg = self.cfg

        # 固定曝光和增益，减少每帧曝光时间变化。
        if cfg.exposure_auto_off:
            self.set_enum("ExposureAuto", "Off", required=False)
        if cfg.exposure_us is not None:
            self.set_float("ExposureTime", float(cfg.exposure_us), required=False)

        if cfg.gain_auto_off:
            self.set_enum("GainAuto", "Off", required=False)
        if cfg.gain is not None:
            self.set_float("Gain", float(cfg.gain), required=False)

        # 连续采集 + 等待软件触发。
        self.set_enum("AcquisitionMode", "Continuous", required=False)
        self.set_enum("TriggerMode", "On", required=True)
        self.set_enum("TriggerSource", cfg.trigger_source, required=True)
        # Software trigger 下 TriggerActivation 可能不存在/无效，失败不致命。
        self.set_enum("TriggerActivation", cfg.trigger_activation, required=False)

        # 触发模式下不依赖 AcquisitionFrameRate；有些型号不允许设置，失败忽略。
        self.set_bool("AcquisitionFrameRateEnable", False, required=False)

        print(
            f"[MVS][{cfg.camera_id}] configured soft trigger: "
            f"TriggerMode=On TriggerSource={cfg.trigger_source} "
            f"ExposureTime={cfg.exposure_us}us Gain={cfg.gain}"
        )

    def configure_strobe_output(self) -> None:
        cfg = self.cfg
        self.set_enum("LineSelector", cfg.strobe_line_selector, required=False)
        self.set_enum("LineMode", cfg.strobe_line_mode, required=False)
        self.set_enum("LineSource", cfg.strobe_line_source, required=False)
        self.set_bool(cfg.strobe_enable_feature, True, required=False)
        if cfg.strobe_duration_us is not None:
            # 不同型号可能不支持该节点，失败只报警，不做错误 fallback。
            self.set_float("StrobeLineDuration", float(cfg.strobe_duration_us), required=False)
        if cfg.strobe_delay_us is not None:
            self.set_float("StrobeLineDelay", float(cfg.strobe_delay_us), required=False)
        print(
            f"[MVS][{cfg.camera_id}] configured strobe output: "
            f"{cfg.strobe_line_selector} {cfg.strobe_line_source}"
        )

    def try_set_bgr8_pixel_format(self) -> None:
        mvs = self.mvs
        for attr_name in ["PixelType_Gvsp_BGR8_Packed", "PixelType_Gvsp_RGB8_Packed"]:
            pix = getattr(mvs, attr_name, None)
            if pix is None:
                continue
            try:
                ret = self.cam.MV_CC_SetEnumValue("PixelFormat", int(pix))
                if ret == 0:
                    print(f"[MVS][{self.cfg.camera_id}] set PixelFormat={attr_name}")
                    return
            except Exception:
                pass
        print(f"[MVS][{self.cfg.camera_id}][WARN] prefer_bgr8 requested but PixelFormat set failed")

    # ------------------------------------------------------------------
    # 触发与取帧
    # ------------------------------------------------------------------

    def trigger_software(self) -> TriggerCommandResult:
        start_wall = now_us()
        start_perf = perf_us()
        ret = 0
        err = ""
        try:
            ret = self.set_command("TriggerSoftware", required=False)
        except Exception as e:
            ret = -1
            err = f"{type(e).__name__}: {e}"
        end_perf = perf_us()
        end_wall = now_us()
        return TriggerCommandResult(
            camera_id=self.cfg.camera_id,
            serial=self.camera_sn,
            ret=int(ret),
            command_start_wall_us=start_wall,
            command_end_wall_us=end_wall,
            command_start_perf_us=start_perf,
            command_end_perf_us=end_perf,
            error=err,
        )

    def grab_one(self, timeout_ms: Optional[int] = None) -> FrameGrabResult:
        timeout = int(timeout_ms if timeout_ms is not None else self.cfg.timeout_ms)
        mvs = self.mvs
        start_wall = now_us()
        start_perf = perf_us()
        frame_out = mvs.MV_FRAME_OUT()
        ctypes.memset(ctypes.byref(frame_out), 0, ctypes.sizeof(frame_out))
        ret = self.cam.MV_CC_GetImageBuffer(frame_out, timeout)
        if ret != 0:
            end_perf = perf_us()
            end_wall = now_us()
            return FrameGrabResult(
                camera_id=self.cfg.camera_id,
                serial=self.camera_sn,
                ok=False,
                ret=int(ret),
                frame=None,
                frame_info={},
                grab_start_wall_us=start_wall,
                grab_end_wall_us=end_wall,
                grab_start_perf_us=start_perf,
                grab_end_perf_us=end_perf,
                error=f"MV_CC_GetImageBuffer ret=0x{ret:x}",
            )

        try:
            info = self._extract_frame_info(frame_out.stFrameInfo)
            frame = self._convert_to_bgr(frame_out)
            end_perf = perf_us()
            end_wall = now_us()
            return FrameGrabResult(
                camera_id=self.cfg.camera_id,
                serial=self.camera_sn,
                ok=True,
                ret=0,
                frame=frame,
                frame_info=info,
                grab_start_wall_us=start_wall,
                grab_end_wall_us=end_wall,
                grab_start_perf_us=start_perf,
                grab_end_perf_us=end_perf,
                error="",
            )
        except Exception as e:
            end_perf = perf_us()
            end_wall = now_us()
            return FrameGrabResult(
                camera_id=self.cfg.camera_id,
                serial=self.camera_sn,
                ok=False,
                ret=-1,
                frame=None,
                frame_info={},
                grab_start_wall_us=start_wall,
                grab_end_wall_us=end_wall,
                grab_start_perf_us=start_perf,
                grab_end_perf_us=end_perf,
                error=f"convert failed: {type(e).__name__}: {e}",
            )
        finally:
            try:
                self.cam.MV_CC_FreeImageBuffer(frame_out)
            except Exception:
                pass

    def _extract_frame_info(self, info: Any) -> Dict[str, Any]:
        names = [
            "nWidth", "nHeight", "nFrameLen", "enPixelType", "nFrameNum",
            "nDevTimeStampHigh", "nDevTimeStampLow", "nHostTimeStamp",
            "nSecondCount", "nCycleCount", "nCycleOffset",
        ]
        out: Dict[str, Any] = {}
        for n in names:
            if hasattr(info, n):
                try:
                    out[n] = int(getattr(info, n))
                except Exception:
                    pass
        return out

    def _convert_to_bgr(self, frame_out: Any) -> np.ndarray:
        mvs = self.mvs
        info = frame_out.stFrameInfo
        w = int(info.nWidth)
        h = int(info.nHeight)
        src_len = int(info.nFrameLen)
        src_pixel_type = int(info.enPixelType)

        dst_pixel = getattr(mvs, "PixelType_Gvsp_BGR8_Packed", None)
        need_rgb_to_bgr = False
        if dst_pixel is None:
            dst_pixel = getattr(mvs, "PixelType_Gvsp_RGB8_Packed")
            need_rgb_to_bgr = True

        dst_size = w * h * 3
        dst_buf = (ctypes.c_ubyte * dst_size)()
        param = mvs.MV_CC_PIXEL_CONVERT_PARAM()
        ctypes.memset(ctypes.byref(param), 0, ctypes.sizeof(param))
        param.nWidth = w
        param.nHeight = h
        param.pSrcData = frame_out.pBufAddr
        param.nSrcDataLen = src_len
        param.enSrcPixelType = src_pixel_type
        param.enDstPixelType = dst_pixel
        param.pDstBuffer = dst_buf
        param.nDstBufferSize = dst_size

        ret = self.cam.MV_CC_ConvertPixelType(param)
        if ret != 0:
            # fallback：Mono8 或已经是 3 通道 packed。
            raw = np.ctypeslib.as_array((ctypes.c_ubyte * src_len).from_address(frame_out.pBufAddr))
            if src_len == w * h:
                gray = raw.reshape(h, w).copy()
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if src_len >= w * h * 3:
                return raw[: w * h * 3].reshape(h, w, 3).copy()
            raise RuntimeError(f"MV_CC_ConvertPixelType failed: 0x{ret:x}, pixel_type={src_pixel_type}, len={src_len}")

        img = np.ctypeslib.as_array(dst_buf).reshape(h, w, 3).copy()
        if need_rgb_to_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img


# =============================================================================
# 多相机软同步组
# =============================================================================


class MvsSoftSyncGroup:
    """多台 MVS 相机软同步触发组。"""

    def __init__(self, camera_configs: Sequence[HikCameraConfig], sync_cfg: Optional[SoftSyncConfig] = None):
        if not camera_configs:
            raise ValueError("camera_configs must not be empty")
        self.sync_cfg = sync_cfg or SoftSyncConfig()
        self.cameras: List[HikMvsSoftTriggerCamera] = [HikMvsSoftTriggerCamera(c) for c in camera_configs]
        self._executor: Optional[ThreadPoolExecutor] = None
        self._last_trigger_perf_us: Optional[int] = None

    def open_all(self) -> None:
        for cam in self.cameras:
            cam.open()
        workers = max(1, min(int(self.sync_cfg.max_workers), max(2, len(self.cameras) * 2)))
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mvs_softsync")
        print(f"[SoftSync] opened {len(self.cameras)} cameras, executor_workers={workers}")

    def close_all(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None
        for cam in self.cameras:
            cam.close()
        print("[SoftSync] closed")

    def __enter__(self) -> "MvsSoftSyncGroup":
        self.open_all()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close_all()

    def trigger_once(
        self,
        trigger_index: int,
        event_pulse_callback: Optional[EventPulseCallback] = None,
        event_ts_us: Optional[int] = None,
    ) -> List[SyncedMvsFrame]:
        """执行一次同步触发并取回所有相机图像。

        顺序：
          1. 如启用 event pre-pulse，则调用 event_pulse_callback(trigger_index, pulse_width_us)。
          2. 等待 event_pre_pulse_lead_us。
          3. 并行对所有 MVS 调 TriggerSoftware。
          4. 并行/顺序取回每台相机一帧。
          5. 返回每台相机的 SyncedMvsFrame。

        event_pulse_callback 由上层根据采集卡/IO SDK 实现。本文件不直接控制具体 IO 硬件。
        """
        if self._executor is None:
            raise RuntimeError("MvsSoftSyncGroup is not opened. Call open_all() first.")

        cfg = self.sync_cfg
        event_pulse_wall: Optional[int] = None
        event_pulse_end_wall: Optional[int] = None
        event_pulse_ret: Any = None
        event_pulse_error = ""

        if cfg.event_pre_pulse_enable:
            event_pulse_wall = now_us()
            try:
                if event_pulse_callback is None:
                    # 不视为致命错误：允许先只测双 MVS 软同步。
                    event_pulse_ret = None
                    event_pulse_error = "event_pulse_callback is None"
                else:
                    event_pulse_ret = event_pulse_callback(int(trigger_index), int(cfg.event_pulse_width_us))
            except Exception as e:
                event_pulse_ret = None
                event_pulse_error = f"{type(e).__name__}: {e}"
            event_pulse_end_wall = now_us()
            sleep_us(int(cfg.event_pre_pulse_lead_us))

        # 触发所有相机。
        trigger_results = self._trigger_all()
        self._warn_trigger_spread(trigger_results)

        # 取回所有帧。
        grab_results = self._grab_all(timeout_ms=int(cfg.trigger_timeout_ms))

        trig_by_id = {r.camera_id: r for r in trigger_results}
        frames: List[SyncedMvsFrame] = []
        for gr in grab_results:
            tr = trig_by_id.get(gr.camera_id)
            info = gr.frame_info or {}
            frame = SyncedMvsFrame(
                camera_id=gr.camera_id,
                camera_sn=gr.serial,
                trigger_index=int(trigger_index),
                sync_mode="software_trigger_with_event_prepulse" if cfg.event_pre_pulse_enable else "software_trigger_only",
                frame=gr.frame,
                ok=bool(gr.ok and tr is not None and tr.ret == 0),
                mvs_frame_num=info.get("nFrameNum"),
                width=info.get("nWidth"),
                height=info.get("nHeight"),
                pixel_type=info.get("enPixelType"),
                frame_len=info.get("nFrameLen"),
                raw_frame_info=info,
                event_pulse_enable=bool(cfg.event_pre_pulse_enable),
                event_pulse_wall_us=event_pulse_wall,
                event_pulse_end_wall_us=event_pulse_end_wall,
                event_pulse_ret=event_pulse_ret,
                event_pulse_error=event_pulse_error,
                event_pre_pulse_lead_us=int(cfg.event_pre_pulse_lead_us),
                event_pulse_width_us=int(cfg.event_pulse_width_us),
                trigger_cmd_wall_us=tr.command_start_wall_us if tr else None,
                trigger_cmd_end_wall_us=tr.command_end_wall_us if tr else None,
                trigger_ret=tr.ret if tr else None,
                trigger_duration_us=tr.duration_us if tr else None,
                trigger_error=tr.error if tr else "missing trigger result",
                grab_wall_us=gr.grab_start_wall_us,
                grab_end_wall_us=gr.grab_end_wall_us,
                grab_ret=gr.ret,
                grab_duration_us=gr.grab_duration_us,
                grab_error=gr.error,
                event_ts_us=event_ts_us,
                sync_status="paired_event_trigger" if event_ts_us is not None else "softsync_no_event_ts",
            )
            if frame.ok is False and not frame.grab_error and not frame.trigger_error:
                frame.sync_status = "softsync_error"
            frames.append(frame)

        frames.sort(key=lambda x: x.camera_id)
        return frames

    def _trigger_all(self) -> List[TriggerCommandResult]:
        if self._executor is None:
            raise RuntimeError("executor is None")
        if not self.sync_cfg.parallel_trigger or len(self.cameras) == 1:
            return [cam.trigger_software() for cam in self.cameras]

        futures = [self._executor.submit(cam.trigger_software) for cam in self.cameras]
        out: List[TriggerCommandResult] = []
        for f in as_completed(futures):
            try:
                out.append(f.result())
            except Exception as e:
                out.append(TriggerCommandResult(
                    camera_id="unknown", serial="", ret=-1,
                    command_start_wall_us=now_us(), command_end_wall_us=now_us(),
                    command_start_perf_us=perf_us(), command_end_perf_us=perf_us(),
                    error=f"trigger future error: {type(e).__name__}: {e}\n{traceback.format_exc()}",
                ))
        return out

    def _grab_all(self, timeout_ms: int) -> List[FrameGrabResult]:
        if self._executor is None:
            raise RuntimeError("executor is None")
        if not self.sync_cfg.parallel_grab or len(self.cameras) == 1:
            return [cam.grab_one(timeout_ms=timeout_ms) for cam in self.cameras]

        futures = [self._executor.submit(cam.grab_one, timeout_ms) for cam in self.cameras]
        out: List[FrameGrabResult] = []
        for f in as_completed(futures):
            try:
                out.append(f.result())
            except Exception as e:
                out.append(FrameGrabResult(
                    camera_id="unknown", serial="", ok=False, ret=-1, frame=None, frame_info={},
                    grab_start_wall_us=now_us(), grab_end_wall_us=now_us(),
                    grab_start_perf_us=perf_us(), grab_end_perf_us=perf_us(),
                    error=f"grab future error: {type(e).__name__}: {e}\n{traceback.format_exc()}",
                ))
        return out

    def _warn_trigger_spread(self, trigger_results: List[TriggerCommandResult]) -> None:
        starts = [r.command_start_perf_us for r in trigger_results if r.ret == 0]
        if len(starts) < 2:
            return
        spread = max(starts) - min(starts)
        if spread > int(self.sync_cfg.trigger_spread_warn_us):
            msg = " ".join([f"{r.camera_id}:ret=0x{int(r.ret) & 0xffffffff:x},start={r.command_start_perf_us}" for r in trigger_results])
            print(f"[SoftSync][WARN] trigger command spread {spread} us > {self.sync_cfg.trigger_spread_warn_us} us | {msg}")

    def wait_until_next_period(self) -> None:
        """按 sync_cfg.fps 控制周期。

        publisher 的主循环可以这样用：
            group.wait_until_next_period()
            frames = group.trigger_once(i, event_pulse_callback=...)
        """
        period_us = int(round(1_000_000.0 / max(1e-6, float(self.sync_cfg.fps))))
        t = perf_us()
        if self._last_trigger_perf_us is None:
            self._last_trigger_perf_us = t
            return
        target = self._last_trigger_perf_us + period_us
        remain = target - t
        if remain > 0:
            sleep_us(remain)
        self._last_trigger_perf_us = target


# =============================================================================
# 简单工具：将 frames 转成日志行
# =============================================================================


def synced_frames_to_csv_rows(frames: Sequence[SyncedMvsFrame]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for f in frames:
        rows.append({
            "trigger_index": f.trigger_index,
            "camera_id": f.camera_id,
            "camera_sn": f.camera_sn,
            "ok": int(f.ok),
            "sync_mode": f.sync_mode,
            "sync_status": f.sync_status,
            "mvs_frame_num": f.mvs_frame_num,
            "width": f.width,
            "height": f.height,
            "event_pulse_wall_us": f.event_pulse_wall_us,
            "event_pulse_end_wall_us": f.event_pulse_end_wall_us,
            "event_pulse_ret": f.event_pulse_ret,
            "event_pulse_error": f.event_pulse_error,
            "event_pre_pulse_lead_us": f.event_pre_pulse_lead_us,
            "event_pulse_width_us": f.event_pulse_width_us,
            "trigger_cmd_wall_us": f.trigger_cmd_wall_us,
            "trigger_cmd_end_wall_us": f.trigger_cmd_end_wall_us,
            "trigger_ret": f.trigger_ret,
            "trigger_duration_us": f.trigger_duration_us,
            "trigger_error": f.trigger_error,
            "grab_wall_us": f.grab_wall_us,
            "grab_end_wall_us": f.grab_end_wall_us,
            "grab_ret": f.grab_ret,
            "grab_duration_us": f.grab_duration_us,
            "grab_error": f.grab_error,
            "event_ts_us": f.event_ts_us,
        })
    return rows


if __name__ == "__main__":
    print(
        "This is a core module. Import it from mvs_softsync_frame_publisher.py.\n"
        "Example:\n"
        "  from mvs_softsync_capture_core import HikCameraConfig, SoftSyncConfig, MvsSoftSyncGroup\n"
    )
