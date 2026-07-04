#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from numba import njit
STATE_KEY='observation.state'; ARM=np.array([0,1,2,3,4,5,6,8,9,10,11,12,13,14],dtype=np.int64); GRIP=np.array([7,15],dtype=np.int64)
@njit(cache=True)
def keep_count_hybrid(x, jd, gd, stride, keep_first_static):
    n=x.shape[0]
    if n<=0: return 0
    kept=1; anchor=0; static_run=0
    for i in range(1,n):
        arm=0.0
        for k in range(len(ARM)):
            d=abs(x[i,ARM[k]]-x[anchor,ARM[k]])
            if d>arm: arm=d
        grip=0.0
        for k in range(len(GRIP)):
            d=abs(x[i,GRIP[k]]-x[anchor,GRIP[k]])
            if d>grip: grip=d
        if arm>jd or grip>gd:
            kept+=1; anchor=i; static_run=0
        else:
            static_run+=1
            if keep_first_static:
                if (static_run-1)%stride==0: kept+=1
            else:
                if static_run%stride==0: kept+=1
    return kept

def read_json(p): return json.loads(Path(p).read_text())
def tasks(root):
    root=Path(root); return [root] if (root/'meta/info.json').is_file() else sorted([p for p in root.iterdir() if (p/'meta/info.json').is_file()])
def fmt(task,tmpl,c,f): return task/tmpl.format(chunk_index=int(c),file_index=int(f),episode_chunk=int(c),episode_index=int(f),video_key='')
def eps(task):
    info=read_json(task/'meta/info.json'); tmpl=info.get('data_path'); chunks=int(info.get('chunks_size') or 1000); out=[]
    for mp in sorted((task/'meta/episodes').glob('chunk-*/file-*.parquet')):
        for r in pq.read_table(str(mp)).to_pylist():
            idx=int(r.get('episode_index',len(out))); length=int(r.get('length') or r.get('episode_length') or r.get('num_frames') or 0)
            if length<=1: continue
            c=int(r.get('data/chunk_index',idx//chunks)); f=int(r.get('data/file_index',idx)); fr=int(r.get('dataset_from_index',0)); to=int(r.get('dataset_to_index',fr+length)); out.append((fmt(task,tmpl,c,f),fr,to))
    return out

def analyze_task(task, candidates):
    cache={}; total=0; kept={c:0 for c in candidates}
    for path,fr,to in eps(task):
        arr=cache.get(path)
        if arr is None:
            arr=np.asarray(pq.read_table(str(path),columns=[STATE_KEY]).column(STATE_KEY).to_pylist(),dtype=np.float32)[:,:16]; cache[path]=arr
        x=arr[fr:to]; total+=len(x)
        for c in candidates:
            jd,gd,stride,keep_first=c
            kept[c]+=int(keep_count_hybrid(x,float(jd),float(gd),int(stride),bool(keep_first)))
    return total, kept
cands=[(0.0,0.0,2,True),(0.0,0.0,3,True),(1e-4,1e-4,2,True),(1e-4,1e-4,3,True),(1e-4,1e-3,2,True),(1e-4,1e-3,3,True)]
for root in map(Path,['/inspire/qb-ilm2/project/robot-body/public/bokai/franka_dual_test','/inspire/qb-ilm2/project/robot-body/public/bokai/franka_dual_v2']):
    print('\nROOT',root)
    gtot=0; g={c:0 for c in cands}; per=[]
    for task in tasks(root):
        total, kept=analyze_task(task,cands); gtot+=total
        for c,v in kept.items(): g[c]+=v
        per.append((task.name,total,kept))
    for c in cands:
        print('jd={} gd={} stride={} keep_first={} keep={:.2f}% {}/{}'.format(*c,100*g[c]/gtot,g[c],gtot))
    best=(0.0001,0.0001,2,True)
    print('per-task for jd=1e-4 gd=1e-4 stride2 keep-first')
    for name,total,kept in per:
        print('  {:24s} {:6.2f}%'.format(name,100*kept[best]/total))
