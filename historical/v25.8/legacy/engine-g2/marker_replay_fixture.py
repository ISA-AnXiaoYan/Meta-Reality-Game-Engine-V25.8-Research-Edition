from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

try:
    from .marker_blink_decoder import EVENT_DTYPE, EVP_HEADER, EVP_MAGIC, synthetic_events
except ImportError:
    from marker_blink_decoder import EVENT_DTYPE, EVP_HEADER, EVP_MAGIC, synthetic_events  # type: ignore


def synthetic_marker_stream(
    marker_ids: Sequence[int],
    *,
    repeats: int = 1,
    alpha_us: int = 20000,
    pulse_width_us: int = 10000,
    checksum: bool = False,
    inter_frame_gap_us: int = 80000,
) -> np.ndarray:
    arrays: List[np.ndarray] = []
    cursor = 100000
    for _ in range(max(1, int(repeats))):
        for marker_id in marker_ids:
            arr = synthetic_events(
                int(marker_id),
                alpha_us=int(alpha_us),
                pulse_width_us=int(pulse_width_us),
                checksum=bool(checksum),
            ).copy()
            first = int(arr["t"].min())
            arr["t"] += int(cursor - first)
            cursor = int(arr["t"].max()) + max(int(inter_frame_gap_us), 4 * int(alpha_us))
            arrays.append(arr)
    if not arrays:
        return np.asarray([], dtype=EVENT_DTYPE)
    out = np.concatenate(arrays)
    out.sort(order="t")
    return out


def partition_events(evs: np.ndarray, *, slice_us: int, phase_us: int = 0) -> Iterable[np.ndarray]:
    arr = np.asarray(evs)
    if len(arr) <= 0:
        return
    width = max(1, int(slice_us))
    phase = int(phase_us)
    first_idx = int(np.floor((int(arr["t"].min()) - phase) / width))
    last_idx = int(np.floor((int(arr["t"].max()) - phase) / width))
    for idx in range(first_idx, last_idx + 1):
        start = phase + idx * width
        end = start + width
        mask = (arr["t"] >= start) & (arr["t"] < end)
        yield np.ascontiguousarray(arr[mask])


def write_evp_fixture(
    output_dir: Path,
    evs: np.ndarray,
    *,
    slice_us: int = 4000,
    phase_us: int = 0,
    camera: str = "G",
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "event_slices.evp"
    manifest_path = output_dir / "event_slices_manifest.json"
    index_path = output_dir / "event_slices_index.csv"
    rows = ["seq,slice_ts_us,event_count,file_offset,payload_bytes"]
    total_events = 0
    slices = 0
    with data_path.open("wb") as fp:
        for seq, batch in enumerate(partition_events(evs, slice_us=slice_us, phase_us=phase_us)):
            batch = np.asarray(batch, dtype=EVENT_DTYPE)
            slice_ts_us = int(batch["t"].max()) if len(batch) else int(phase_us + (seq + 1) * slice_us)
            offset = int(fp.tell())
            payload = batch.tobytes(order="C")
            fp.write(EVP_HEADER.pack(
                EVP_MAGIC,
                1,
                EVP_HEADER.size,
                int(seq),
                int(slice_ts_us),
                0,
                int(len(batch)),
                int(EVENT_DTYPE.itemsize),
            ))
            fp.write(payload)
            rows.append(f"{seq},{slice_ts_us},{len(batch)},{offset},{len(payload)}")
            total_events += int(len(batch))
            slices += 1
    index_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "kind": "event_slice_fixture",
        "camera": str(camera),
        "data_file": data_path.name,
        "index_file": index_path.name,
        "event_dtype_descr": [[str(name), str(fmt)] for name, fmt in EVENT_DTYPE.descr],
        "header_format": "<4sHHQQQII",
        "header_size": int(EVP_HEADER.size),
        "record_magic": "EVP1",
        "record_version": 1,
        "slice_us": int(slice_us),
        "phase_us": int(phase_us),
        "slices_written": int(slices),
        "events_written": int(total_events),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _parse_ids(value: str) -> List[int]:
    out = [int(v.strip()) for v in str(value).split(",") if v.strip()]
    if not out or any(v < 0 or v > 255 for v in out):
        raise ValueError("--ids must contain comma-separated values in 0..255")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate deterministic IR-ID EVP replay fixtures.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--ids", default="7,13")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--alpha-ms", type=float, default=20.0)
    ap.add_argument("--pulse-width-ms", type=float, default=10.0)
    ap.add_argument("--checksum", action="store_true")
    ap.add_argument("--slice-us", type=int, default=4000)
    ap.add_argument("--phase-us", type=int, default=0)
    ap.add_argument("--camera", default="G")
    args = ap.parse_args()
    ids = _parse_ids(args.ids)
    evs = synthetic_marker_stream(
        ids,
        repeats=int(args.repeats),
        alpha_us=int(round(float(args.alpha_ms) * 1000.0)),
        pulse_width_us=int(round(float(args.pulse_width_ms) * 1000.0)),
        checksum=bool(args.checksum),
    )
    manifest = write_evp_fixture(
        Path(args.output_dir),
        evs,
        slice_us=int(args.slice_us),
        phase_us=int(args.phase_us),
        camera=str(args.camera),
    )
    print(json.dumps({"ids": ids, "events": int(len(evs)), "manifest": manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
