#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R
try:
    from numba import njit
except Exception:
    njit = None

STATE_KEY = "observation.state"
SIM_ROOT = Path("/inspire/qb-ilm2/project/robot-body/public/hairuoliu/franka_dual/sim")
sys.path.insert(0, str(SIM_ROOT))
from fr3_fk import Fr3FK  # noqa: E402

if njit is not None:
    @njit(cache=True)
    def greedy_tcp_counts(pos, quat_xyzw, grip, trans_thresholds, rot_thresholds, grip_thresholds):
        # pos: [N,2,3], quat_xyzw: [N,2,4], grip: [N,2]
        out = np.zeros((len(trans_thresholds), len(rot_thresholds), len(grip_thresholds)), dtype=np.int64)
        n = pos.shape[0]
        if n <= 0:
            return out
        for a in range(len(trans_thresholds)):
            td = trans_thresholds[a]
            for b in range(len(rot_thresholds)):
                rd = rot_thresholds[b]
                for c in range(len(grip_thresholds)):
                    gd = grip_thresholds[c]
                    kept = 1
                    last = 0
                    for i in range(1, n):
                        trans = 0.0
                        for arm in range(2):
                            for xyz_i in range(3):
                                d = abs(pos[i, arm, xyz_i] - pos[last, arm, xyz_i])
                                if d > trans:
                                    trans = d
                        grip_d = 0.0
                        for arm in range(2):
                            d = abs(grip[i, arm] - grip[last, arm])
                            if d > grip_d:
                                grip_d = d
                        rot_d = 0.0
                        for arm in range(2):
                            # abs dot handles q/-q equivalence. angle = 2 acos(|dot|)
                            dot = 0.0
                            for k in range(4):
                                dot += quat_xyzw[i, arm, k] * quat_xyzw[last, arm, k]
                            if dot < 0:
                                dot = -dot
                            if dot > 1:
                                dot = 1.0
                            d = 2.0 * np.arccos(dot)
                            if d > rot_d:
                                rot_d = d
                        if trans > td or rot_d > rd or grip_d > gd:
                            kept += 1
                            last = i
                    out[a, b, c] = kept
        return out
else:
    greedy_tcp_counts = None

def read_json(path: Path):
    return json.loads(path.read_text())

def discover_tasks(root: Path):
    if (root / "meta" / "info.json").is_file():
        return [root]
    return sorted([p for p in root.iterdir() if (p / "meta" / "info.json").is_file()])

def fmt_path(task_dir: Path, tmpl: str, chunk: int, file: int):
    return task_dir / tmpl.format(chunk_index=int(chunk), file_index=int(file), episode_chunk=int(chunk), episode_index=int(file), video_key="")

def load_eps(task_dir: Path):
    info = read_json(task_dir / "meta/info.json")
    tmpl = info.get("data_path") or "data/chunk-{episode_chunk:03d}/file-{episode_index:03d}.parquet"
    chunks = int(info.get("chunks_size") or 1000)
    out=[]
    for mp in sorted((task_dir / "meta/episodes").glob("chunk-*/file-*.parquet")):
        for row in pq.read_table(str(mp)).to_pylist():
            ep = int(row.get("episode_index", len(out)))
            length = int(row.get("length") or row.get("episode_length") or row.get("num_frames") or 0)
            if length <= 1: continue
            dc = int(row.get("data/chunk_index", ep // chunks))
            df = int(row.get("data/file_index", ep))
            fr = int(row.get("dataset_from_index", 0))
            to = int(row.get("dataset_to_index", fr + length))
            out.append((ep, fmt_path(task_dir, tmpl, dc, df), fr, to))
    return sorted(out, key=lambda x: x[0])

def state_cache_get(cache: dict[Path,np.ndarray], path: Path):
    arr = cache.get(path)
    if arr is None:
        arr = np.asarray(pq.read_table(str(path), columns=[STATE_KEY]).column(STATE_KEY).to_pylist(), dtype=np.float32)[:, :16]
        cache[path] = arr
    return arr

def state_to_tcp(states: np.ndarray, fk: Fr3FK):
    states = np.asarray(states, dtype=np.float32)
    poses_l = fk.flange_pose(states[:, 0:7])
    poses_r = fk.flange_pose(states[:, 8:15])
    pos = np.stack([poses_l[:, :3, 3], poses_r[:, :3, 3]], axis=1).astype(np.float32)
    quat = np.stack([
        R.from_matrix(poses_l[:, :3, :3]).as_quat(),
        R.from_matrix(poses_r[:, :3, :3]).as_quat(),
    ], axis=1).astype(np.float32)
    grip = np.stack([states[:, 7], states[:, 15]], axis=1).astype(np.float32)
    return pos, quat, grip

def quantiles(vals):
    vals=np.asarray(vals,dtype=np.float64)
    ps=[0,1,5,10,25,50,75,90,95,99,100]
    if vals.size == 0: return {str(p): None for p in ps}
    qs=np.percentile(vals, ps)
    return {str(p): float(v) for p,v in zip(ps,qs)}

def analyze(root: Path, trans, rot, grip, max_episodes_per_task=None):
    fk=Fr3FK(urdf_path=SIM_ROOT / "fr3v2_1.urdf")
    tasks=discover_tasks(root)
    global_counts={(td,rd,gd): [0,0] for td in trans for rd in rot for gd in grip}
    by_task={}
    all_adj_t=[]; all_adj_r=[]; all_adj_g=[]
    total_frames=0; total_eps=0
    # warm numba
    if greedy_tcp_counts is not None:
        dummy_pos=np.zeros((2,2,3),dtype=np.float32); dummy_quat=np.zeros((2,2,4),dtype=np.float32); dummy_quat[:,:,3]=1; dummy_grip=np.zeros((2,2),dtype=np.float32)
        greedy_tcp_counts(dummy_pos,dummy_quat,dummy_grip,np.asarray(trans,dtype=np.float64),np.asarray(rot,dtype=np.float64),np.asarray(grip,dtype=np.float64))
    for task in tasks:
        eps=load_eps(task)
        if max_episodes_per_task:
            eps=eps[:max_episodes_per_task]
        cache={}; ret={(td,rd,gd): [0,0] for td in trans for rd in rot for gd in grip}
        adj_t=[]; adj_r=[]; adj_g=[]; frames=0
        for ep,path,fr,to in eps:
            states=state_cache_get(cache,path)[fr:to]
            n=len(states)
            if n<=1: continue
            pos, quat, grip_arr = state_to_tcp(states, fk)
            dt=np.max(np.abs(np.diff(pos,axis=0)), axis=(1,2))
            dg=np.max(np.abs(np.diff(grip_arr,axis=0)), axis=1)
            dots=np.abs(np.sum(quat[1:]*quat[:-1], axis=-1)).clip(0,1)
            dr=(2*np.arccos(dots)).max(axis=1)
            adj_t.append(dt); adj_r.append(dr); adj_g.append(dg)
            counts=greedy_tcp_counts(pos, quat, grip_arr, np.asarray(trans,dtype=np.float64), np.asarray(rot,dtype=np.float64), np.asarray(grip,dtype=np.float64))
            for a,td in enumerate(trans):
                for b,rd in enumerate(rot):
                    for c,gd in enumerate(grip):
                        k=int(counts[a,b,c])
                        ret[(td,rd,gd)][0]+=k; ret[(td,rd,gd)][1]+=n
                        global_counts[(td,rd,gd)][0]+=k; global_counts[(td,rd,gd)][1]+=n
            frames += n
        total_frames += frames; total_eps += len(eps)
        at=np.concatenate(adj_t) if adj_t else np.array([]); ar=np.concatenate(adj_r) if adj_r else np.array([]); ag=np.concatenate(adj_g) if adj_g else np.array([])
        all_adj_t.append(at); all_adj_r.append(ar); all_adj_g.append(ag)
        by_task[task.name]={
            'episodes': len(eps), 'frames': frames,
            'adjacent_tcp_trans_quantiles': quantiles(at),
            'adjacent_tcp_rot_quantiles_rad': quantiles(ar),
            'adjacent_gripper_quantiles': quantiles(ag),
            'retention': [{'trans_delta':td,'rot_delta':rd,'gripper_delta':gd,'kept':v[0],'total':v[1],'kept_ratio':v[0]/v[1] if v[1] else 0} for (td,rd,gd),v in sorted(ret.items())],
        }
    gt=np.concatenate(all_adj_t) if all_adj_t else np.array([]); gr=np.concatenate(all_adj_r) if all_adj_r else np.array([]); gg=np.concatenate(all_adj_g) if all_adj_g else np.array([])
    return {
        'root': str(root), 'tasks': len(tasks), 'episodes': total_eps, 'frames': total_frames,
        'threshold_semantics': 'greedy keep first frame, then keep frame i when max_abs(left/right link8 xyz axis delta since last kept)>trans_delta OR max quaternion angle since last kept)>rot_delta OR max gripper abs diff since last kept)>gripper_delta',
        'global_adjacent_tcp_trans_quantiles': quantiles(gt),
        'global_adjacent_tcp_rot_quantiles_rad': quantiles(gr),
        'global_adjacent_gripper_quantiles': quantiles(gg),
        'global_retention': [{'trans_delta':td,'rot_delta':rd,'gripper_delta':gd,'kept':v[0],'total':v[1],'kept_ratio':v[0]/v[1] if v[1] else 0,'delete_ratio':1-(v[0]/v[1] if v[1] else 0)} for (td,rd,gd),v in sorted(global_counts.items())],
        'by_task': by_task,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('roots', nargs='+')
    ap.add_argument('--trans-thresholds', default='0,0.0002,0.0005,0.001,0.002,0.003,0.004,0.005,0.0075,0.01,0.015,0.02')
    ap.add_argument('--rot-thresholds', default='0,0.002,0.005,0.01,0.02,0.04,0.06545,0.1')
    ap.add_argument('--gripper-thresholds', default='0,0.0005,0.001,0.002,0.005')
    ap.add_argument('--output', default='/tmp/franka_tcp_threshold_report.json')
    ap.add_argument('--max-episodes-per-task', type=int, default=None)
    args=ap.parse_args()
    trans=[float(x) for x in args.trans_thresholds.split(',') if x!='']
    rot=[float(x) for x in args.rot_thresholds.split(',') if x!='']
    grip=[float(x) for x in args.gripper_thresholds.split(',') if x!='']
    reps=[analyze(Path(r), trans, rot, grip, args.max_episodes_per_task) for r in args.roots]
    Path(args.output).write_text(json.dumps(reps, indent=2), encoding='utf-8')
    for rep in reps:
        print('\nROOT', rep['root'])
        print('tasks',rep['tasks'],'episodes',rep['episodes'],'frames',rep['frames'])
        print('adj trans q',rep['global_adjacent_tcp_trans_quantiles'])
        print('adj rot q',rep['global_adjacent_tcp_rot_quantiles_rad'])
        print('adj grip q',rep['global_adjacent_gripper_quantiles'])
        cand=[x for x in rep['global_retention'] if 0.60 <= x['kept_ratio'] <= 0.80]
        cand=sorted(cand,key=lambda x: abs(x['kept_ratio']-0.70))[:20]
        print('top threshold candidates kept 60-80%:')
        for x in cand:
            print('  trans={:.5g} rot={:.5g} grip={:.5g} keep={:.2f}% delete={:.2f}% kept={}/{}'.format(x['trans_delta'],x['rot_delta'],x['gripper_delta'],100*x['kept_ratio'],100*x['delete_ratio'],x['kept'],x['total']))
        if cand:
            best=cand[0]
            print('per task for best:')
            for name,t in rep['by_task'].items():
                e=next(y for y in t['retention'] if y['trans_delta']==best['trans_delta'] and y['rot_delta']==best['rot_delta'] and y['gripper_delta']==best['gripper_delta'])
                print('  {:24s} keep={:6.2f}% frames={}/{}'.format(name,100*e['kept_ratio'],e['kept'],e['total']))
    print('\nwrote', args.output)
if __name__=='__main__': main()
