#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path

DEFAULT_ROOT = Path('/home/ysxq/PycharmProjects/Ids_Test_3.9/test607')

def human_size(n):
    for unit in ['B','KB','MB','GB','TB']:
        if n < 1024:
            return f'{n:.1f}{unit}' if unit != 'B' else f'{n}B'
        n /= 1024.0
    return f'{n:.1f}PB'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(DEFAULT_ROOT))
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    rec = root / 'recordings'
    print(f'[check] recordings root: {rec}')
    if not rec.exists():
        print('[check] recordings directory does not exist')
        return 1
    sessions = [p for p in rec.iterdir() if p.is_dir() and p.name != 'event_raw']
    sessions.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    print(f'[check] sessions: {len(sessions)}')
    for sess in sessions[:5]:
        print(f'\n=== {sess.name} ===')
        for name in ['manifest_mvs.json', 'manifest_mvs_stop.json']:
            fp = sess / name
            if fp.exists():
                try:
                    data = json.loads(fp.read_text(encoding='utf-8'))
                    print(f'{name}: ok keys={list(data.keys())[:8]}')
                except Exception as e:
                    print(f'{name}: read error {e}')
        for sub in sorted(sess.glob('MVS_*')):
            avi = sub / 'raw_mvs.avi'
            csv = sub / 'frame_index.csv'
            n_lines = 0
            if csv.exists():
                try:
                    with open(csv, 'r', encoding='utf-8', errors='ignore') as f:
                        n_lines = sum(1 for _ in f) - 1
                except Exception:
                    n_lines = -1
            print(f'{sub.name}: avi={avi.exists()} {human_size(avi.stat().st_size) if avi.exists() else "-"}, frame_index_rows={n_lines}')
    event_dir = rec / 'event_raw'
    raws = sorted(event_dir.glob('*.raw'), key=lambda p: p.stat().st_mtime, reverse=True) if event_dir.exists() else []
    print(f'\n[check] event raw files: {len(raws)}')
    for p in raws[:10]:
        print(f'  {p.name}  {human_size(p.stat().st_size)}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
