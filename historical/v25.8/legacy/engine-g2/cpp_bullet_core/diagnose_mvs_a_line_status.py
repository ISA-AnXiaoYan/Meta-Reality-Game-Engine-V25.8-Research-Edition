#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
from ctypes import *
from pathlib import Path

MVIMPORT = "/opt/MVS/Samples/64/Python/MvImport"
if MVIMPORT not in sys.path:
    sys.path.insert(0, MVIMPORT)

from MvCameraControl_class import *  # noqa

SERIAL = "DB0664027"
LINE = "Line1"
FPS = 25.0
TEST_SECONDS = 10.0

def bytes_from_c_array(arr):
    return bytes(arr).split(b"\x00", 1)[0]

def get_serial(dev_info):
    try:
        if dev_info.nTLayerType == MV_USB_DEVICE:
            return bytes_from_c_array(dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber).decode(errors="ignore")
        if dev_info.nTLayerType == MV_GIGE_DEVICE:
            return bytes_from_c_array(dev_info.SpecialInfo.stGigEInfo.chSerialNumber).decode(errors="ignore")
    except Exception:
        pass
    return ""

def set_enum_str(cam, name, value):
    if hasattr(cam, "MV_CC_SetEnumValueByString"):
        ret = cam.MV_CC_SetEnumValueByString(name, str(value))
        print(f"[SET] {name}={value} ret=0x{ret:x}")
        return ret
    print(f"[WARN] no MV_CC_SetEnumValueByString for {name}")
    return -1

def set_bool(cam, name, value):
    ret = cam.MV_CC_SetBoolValue(name, bool(value))
    print(f"[SET] {name}={value} ret=0x{ret:x}")
    return ret

def set_float(cam, name, value):
    ret = cam.MV_CC_SetFloatValue(name, float(value))
    print(f"[SET] {name}={value} ret=0x{ret:x}")
    return ret

def get_bool(cam, name):
    try:
        v = c_bool(False)
        # 注意：海康 Python wrapper 内部通常已经会 byref(pBoolValue)，
        # 这里应该传 c_bool 实例本身，而不是 byref(v)。
        ret = cam.MV_CC_GetBoolValue(name, v)
        if ret == 0:
            return int(bool(v.value)), ret
        return None, ret
    except Exception as e:
        print(f"[GET][EXC] {name}: {e}")
        return None, -1

def main():
    dev_list = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, dev_list)
    if ret != 0:
        raise RuntimeError(f"EnumDevices failed ret=0x{ret:x}")

    found = None
    print("[MVS] detected devices:")
    for i in range(int(dev_list.nDeviceNum)):
        dev = cast(dev_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        sn = get_serial(dev)
        print(f"  [{i}] serial={sn}")
        if sn == SERIAL:
            found = dev

    if found is None:
        raise RuntimeError(f"没有找到 MVS_A serial={SERIAL}")

    cam = MvCamera()
    ret = cam.MV_CC_CreateHandle(found)
    if ret != 0:
        raise RuntimeError(f"CreateHandle failed ret=0x{ret:x}")

    try:
        ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            raise RuntimeError(f"OpenDevice failed ret=0x{ret:x}")

        print(f"[DIAG] configure {SERIAL} {LINE} as Strobe / ExposureStartActive")
        cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        set_bool(cam, "AcquisitionFrameRateEnable", True)
        set_float(cam, "AcquisitionFrameRate", FPS)

        set_enum_str(cam, "LineSelector", LINE)
        set_enum_str(cam, "LineMode", "Strobe")
        set_enum_str(cam, "LineSource", "ExposureStartActive")
        set_bool(cam, "LineInverter", False)
        set_bool(cam, "StrobeEnable", True)

        # 尽量加宽脉冲，方便软件轮询 LineStatus 捕捉。若相机不支持会失败，不影响测试继续。
        set_float(cam, "StrobeLineDuration", 20000.0)
        set_float(cam, "StrobeLineDelay", 0.0)
        set_float(cam, "StrobeLinePreDelay", 0.0)

        # 先读一次 LineStatus，看 SDK 是否支持该节点
        v, ret = get_bool(cam, "LineStatus")
        print(f"[GET] LineStatus first={v} ret=0x{ret:x}")
        if ret != 0:
            print("[RESULT] 当前 SDK/相机没有读到 LineStatus，不能用这个方法判断物理输出。")
            return

        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise RuntimeError(f"StartGrabbing failed ret=0x{ret:x}")

        print(f"[DIAG] polling LineStatus for {TEST_SECONDS:.1f}s ...")
        last = None
        changes = 0
        ones = 0
        zeros = 0
        samples = 0
        t_end = time.time() + TEST_SECONDS

        while time.time() < t_end:
            v, ret = get_bool(cam, "LineStatus")
            if ret == 0 and v is not None:
                samples += 1
                if v:
                    ones += 1
                else:
                    zeros += 1
                if last is not None and v != last:
                    changes += 1
                    print(f"[TOGGLE] LineStatus {last} -> {v} at {time.time():.6f}", flush=True)
                last = v
            time.sleep(0.001)

        print(f"[RESULT] samples={samples}, ones={ones}, zeros={zeros}, changes={changes}")

        if changes > 0 and ones > 0 and zeros > 0:
            print("[JUDGE] MVS_A 的 LineStatus 有高低变化：MVS_A 侧很可能有同步脉冲输出。")
            print("[JUDGE] IDS_A 仍然 event_trigger=0，则更像 IDS_A 输入通道/接口/线缆路径没有收到。")
        else:
            print("[JUDGE] 没有观察到 LineStatus 变化：优先怀疑 MVS_A Line1 没有真实脉冲，或 LineStatus 不能反映输出脉冲。")
            print("[JUDGE] 下一步建议做 LineSelector 软件扫描，不换线。")

    finally:
        try:
            cam.MV_CC_StopGrabbing()
        except Exception:
            pass
        try:
            cam.MV_CC_CloseDevice()
        except Exception:
            pass
        try:
            cam.MV_CC_DestroyHandle()
        except Exception:
            pass

if __name__ == "__main__":
    main()
