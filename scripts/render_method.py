"""Generic exo2ego baseline overlay renderer (sam2qt / ffmsam2).
Usage: python render_method.py --res /data3/_Nips26/result/<method> --out <out_dir>
Panels: [exo query frame + GT query mask (green)] | [ego: sam2qt pred (red) + GT contour (green)] | [ego: ffmsam2 pred ...]
"""
import json, os, subprocess, sys
from pathlib import Path
import cv2, numpy as np
from pycocotools import mask as mu

import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument('--res', required=True); _ap.add_argument('--out', required=True)
_a = _ap.parse_args(); RES, OUT = Path(_a.res), Path(_a.out)
EGO = 'aria01_214-1'
JOBS = [  # (uid, exo_cam, obj_safe, result_set_suffix, chunk_dirname, data_root, fps)
  ('6fcb3904-2aee-41e7-a45f-4f498c8e1e82','gp02','sheet_music','gtann','chunk_00170_0_20580','/data3/_Nips26/lmeec_gtann_data',10),
  ('2da6c2ce-a88f-4790-9642-8dfa810ae91b','cam01','bicycle_wheel_0','gtann','chunk_00059_0_4860','/data3/_Nips26/lmeec_gtann_data',10),
  ('2da6c2ce-a88f-4790-9642-8dfa810ae91b','cam01','bicycle_wheel_0','subset2_exo2ego','chunk_00008_141_153','/data3/_Nips26/lmeec_data',4),
]
VARIANTS = ['sam2qt','ffmsam2']

def dec(r):
    if r is None: return None
    return mu.decode({'size':r['size'],'counts':r['counts'].encode() if isinstance(r['counts'],str) else r['counts']}).astype(bool)

def fit(m, img):
    if m is None or m.shape == img.shape[:2]: return m
    return cv2.resize(m.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

def overlay(img, m, color, alpha=0.45):
    m = fit(m, img)
    if m is None or not m.any(): return img
    out = img.copy(); out[m] = (img[m]*(1-alpha) + np.array(color)*alpha).astype(np.uint8); return out

def contour(img, m, color, th=2):
    m = fit(m, img)
    if m is None or not m.any(): return img
    cs,_ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.drawContours(img.copy(), cs, -1, color, th)

def label(img, txt, y=28, color=(255,255,255)):
    cv2.putText(img, txt, (8,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(img, txt, (8,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return img

def encode(frames_dir_tmp, out_mp4, fps):
    subprocess.run(['ffmpeg','-y','-loglevel','error','-framerate',str(fps),'-i',f'{frames_dir_tmp}/%06d.png',
                    '-c:v','libx264','-pix_fmt','yuv420p','-crf','18',str(out_mp4)], check=True)

for uid, exo, obj_safe, rset, chunk, droot, fps in JOBS:
    obj = obj_safe.replace('_',' ') if obj_safe=='sheet_music' else obj_safe.replace('bicycle_wheel','bicycle wheel')
    g = json.load(open(f'{droot}/{uid}/final_total_refined_masks_rle.json'))
    gt_q, gt_t = g[exo][obj], g[EGO][obj]
    preds, mets, qtrack = {}, {}, {}
    for v in VARIANTS:
        d = RES/f'rebuttal_exo2ego_{rset}_{v}'/uid/f'{exo}__{EGO}'/obj_safe/chunk
        preds[v] = json.load(open(d/'predictions.json'))
        qtrack[v] = json.load(open(d/'query_track.json')) if (d/'query_track.json').exists() else {}
        mets[v] = {m['frame']:m for m in json.load(open(d/'metrics.json'))['per_frame']}
        mets[v]['_chunk'] = json.load(open(d/'metrics.json'))['chunk']
    fids = sorted(set(int(k) for v in VARIANTS for k in preds[v]))
    tag = f'{uid[:8]}_{obj_safe}_{rset}_{chunk}'
    odir = OUT/uid; odir.mkdir(parents=True, exist_ok=True)
    tmp = {k: odir/f'.tmp_{tag}_{k}' for k in ['cmp']+VARIANTS}
    for t in tmp.values(): t.mkdir(exist_ok=True); [p.unlink() for p in t.glob('*.png')]
    H = 448
    for i, f in enumerate(fids):
        qi = cv2.imread(f'{droot}/{uid}/{exo}/frames/{f}.jpg'); ti = cv2.imread(f'{droot}/{uid}/{EGO}/frames/{f}.jpg')
        if qi is None or ti is None: continue
        qm = dec(gt_q.get(str(f)))
        qt = dec(qtrack['sam2qt'].get(str(f))) if qtrack.get('sam2qt') else None
        qp = overlay(qi, qt, (0,140,255)); qp = contour(qp, qt, (0,200,255), 1); qp = contour(qp, qm, (0,255,0))
        label(qp, f'EXO query ({exo})  frame {f}')
        qtxt = 'green = GT query' + ('' if qm is not None and qm.any() else ' (none)') + (', orange = SAM2-tracked query (sam2qt)' if qtrack.get('sam2qt') else '')
        label(qp, qtxt, y=H-12, color=(0,255,0))
        panels = [qp]
        tm = dec(gt_t.get(str(f)))
        for v in VARIANTS:
            pm = dec(preds[v].get(str(f)))
            tp = overlay(ti, pm, (0,0,230)); tp = contour(tp, tm, (0,255,0)); tp = contour(tp, pm, (0,0,255), 1)
            m = mets[v].get(f)
            sc = preds[v].get(str(f), {}).get('score')
            jf = (f"J={m['J']:.2f} F={m['F']:.2f}" if m else 'no GT') + (f"  s={sc:.2f}" if sc is not None else '')
            label(tp, f'EGO pred: {v}')
            label(tp, jf, y=52, color=(0,200,255))
            label(tp, 'red = pred, green = GT', y=H-12, color=(0,200,255))
            panels.append(tp)
            cv2.imwrite(str(tmp[v]/f'{i:06d}.png'), np.concatenate([qp, tp], 1))
        cv2.imwrite(str(tmp['cmp']/f'{i:06d}.png'), np.concatenate(panels, 1))
    encode(tmp['cmp'], odir/f'{tag}_compare.mp4', fps)
    for v in VARIANTS: encode(tmp[v], odir/f'{tag}_{v}.mp4', fps)
    for t in tmp.values():
        [p.unlink() for p in t.glob('*.png')]; t.rmdir()
    print(tag, 'frames', len(fids), {v: round(mets[v]['_chunk']['mean_JF'],3) for v in VARIANTS}, flush=True)
