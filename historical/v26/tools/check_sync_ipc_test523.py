#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, csv, json, statistics
from pathlib import Path

DEFAULT_ROOT = Path('/home/ysxq/PycharmProjects/Ids_Test_3.9/test607')

def read_triggers(path: Path):
    if not path.exists():
        return []
    rows=[]
    with path.open('r',encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            try:
                si=int(r.get('sync_index',-1))
                if si < 0:
                    continue
                rows.append({k:int(v) if str(v).lstrip('-').isdigit() else v for k,v in r.items()})
            except Exception:
                pass
    return rows

def read_human(path: Path, max_lines=200000):
    if not path.exists():
        return []
    rows=[]
    with path.open('r',encoding='utf-8') as f:
        for i,line in enumerate(f):
            if i>=max_lines: break
            line=line.strip()
            if not line: continue
            try: rows.append(json.loads(line))
            except Exception: pass
    return rows

def summarize(cam, root):
    sync=root/'sync_ipc'
    tr=read_triggers(sync/f'event_trigger_{cam}.csv')
    hu=read_human(sync/f'human_result_{cam}.jsonl')
    print(f'\n[{cam}]')
    print(f'event triggers(selected) = {len(tr)} file={sync/f"event_trigger_{cam}.csv"}')
    if tr:
        print(f'  sync_index: {tr[0].get("sync_index")} -> {tr[-1].get("sync_index")}, event_ts_us: {tr[0].get("event_ts_us")} -> {tr[-1].get("event_ts_us")}')
    print(f'human frames = {len(hu)} file={sync/f"human_result_{cam}.jsonl"}')
    if hu:
        print(f'  mvs_frame_num: {hu[0].get("mvs_frame_num")} -> {hu[-1].get("mvs_frame_num")}, sync_index_hint: {hu[0].get("sync_index_hint")} -> {hu[-1].get("sync_index_hint")}')
        offs=[]
        for r in hu[:min(len(hu),2000)]:
            try: offs.append(int(r.get('sync_index_hint'))-int(r.get('mvs_frame_num')))
            except Exception: pass
        if offs:
            print(f'  sync_index_hint - mvs_frame_num: median={statistics.median(offs)} unique={sorted(set(offs))[:10]}')
    if tr and hu:
        # 粗略比较最后编号是否同速增长；正式 offset 仍应由同步程序自动估计。
        print(f'  last_gap_hint_vs_trigger = {int(hu[-1].get("sync_index_hint",0))-int(tr[-1].get("sync_index",0))}')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default=str(DEFAULT_ROOT))
    ap.add_argument('--cams',nargs='+',default=['A','D'])
    a=ap.parse_args()
    root=Path(a.root).expanduser().resolve()
    print(f'root={root}')
    for cam in a.cams:
        summarize(cam,root)
if __name__=='__main__': main()
