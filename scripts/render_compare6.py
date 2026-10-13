"""6-panel comparison on our val600_cherry samples.
Panels (2x4): Query first-frame GT (blue) | GT (blue) | Ours/XTrack (green) | ObjectRelator (red) / info | O-MaMa | LM-EEC | V2-SAM (red).
Two frame sets per sample:
  sparse : baseline gtann predictions (30-frame stride, real sparse GT) -> all samples, 2 fps
  dense  : baseline subset1_500 13-frame chunks (where they exist) + dense refined GT (lmeec_data), 10 fps
Both directions (ego2exo / exo2ego) and both baseline variants (sam2qt / ffmsam2).
Our exo outputs are sometimes a local crop of the full frame; the crop box is recovered by template matching.
"""
import csv, json, subprocess
from pathlib import Path
import cv2, numpy as np
from pycocotools import mask as mu

PP = Path('/home/ubuntu/dahyun/_Nips26/project_page'); OURS = Path('/data3/_Nips26/xtrack_hf/val600_cherry'); OUT = Path('/data3/_Nips26/project_page_vis/compare6_vis')
RES = Path('/data3/_Nips26/result'); GTANN = Path('/data3/_Nips26/lmeec_gtann_data'); DENSE = Path('/data3/_Nips26/lmeec_data')
METHODS = [('objectrelator','ObjectRelator'),('o-mama','O-MaMa'),('lm-eec','LM-EEC'),('v2-sam','V2-SAM')]
VARIANTS = ['sam2qt','ffmsam2']
BLUE, GREEN, RED = (255,80,0), (0,200,0), (0,0,230)   # BGR

win = {}
for r in csv.DictReader(open('/data3/_Nips26/egoexo_usable_chunks_val_600frames_refined_2obj.csv')):
    win[(r['take_uid'], r['shared_object'].replace(' ','_').replace('/','_'))] = (int(r['frame_start']), r['ego_cam_id'], r['exo_cam_id'], r['shared_object'])

def dec(r):
    if r is None: return None
    return mu.decode({'size':r['size'],'counts':r['counts'].encode() if isinstance(r['counts'],str) else r['counts']}).astype(bool)
def fit(m, img):
    if m is None or m.shape == img.shape[:2]: return m
    return cv2.resize(m.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
def paint(img, m, color, alpha=0.5):
    m = fit(m, img); out = img.copy()
    if m is None or not m.any(): return out
    out[m] = (img[m]*(1-alpha) + np.array(color)*alpha).astype(np.uint8)
    cs,_ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.drawContours(out, cs, -1, color, 2)
def label(img, txt, y=26, color=(255,255,255), sc=0.65, x=8):
    cv2.putText(img, txt, (x,y), cv2.FONT_HERSHEY_SIMPLEX, sc, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(img, txt, (x,y), cv2.FONT_HERSHEY_SIMPLEX, sc, color, 2, cv2.LINE_AA); return img
def iou(a, b):
    if a is None or b is None: return None
    u = (a|b).sum(); return (a&b).sum()/u if u else 1.0
def read_video(p, gray=False):
    cap = cv2.VideoCapture(str(p)); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if gray else f)
    return fr

def recover_crop(ov_frame, pred_mask, full_gray):
    """Find (scale, x, y) placing our 800x448 crop frame inside the full frame (mask region excluded)."""
    tg = cv2.cvtColor(ov_frame, cv2.COLOR_BGR2GRAY); best = None
    for s in np.arange(0.2, 1.001, 0.01):
        w, h = int(round(tg.shape[1]*s)), int(round(tg.shape[0]*s))
        if w > full_gray.shape[1] or h > full_gray.shape[0]: continue
        t = cv2.resize(tg, (w,h), interpolation=cv2.INTER_AREA)
        mk = cv2.resize((~pred_mask).astype(np.uint8)*255, (w,h), interpolation=cv2.INTER_NEAREST)
        r = cv2.matchTemplate(full_gray, t, cv2.TM_CCOEFF_NORMED, mask=mk); _, mx, _, loc = cv2.minMaxLoc(r)
        if best is None or mx > best[0]: best = (mx, w, h, loc[0], loc[1])
    return best
def letterbox(img, H, W):
    s = min(H / img.shape[0], W / img.shape[1]); r = cv2.resize(img, (int(img.shape[1]*s), int(img.shape[0]*s)))
    out = np.zeros((H, W, 3), np.uint8); y, x = (H - r.shape[0])//2, (W - r.shape[1])//2; out[y:y+r.shape[0], x:x+r.shape[1]] = r; return out
def place(m, box, shape):
    _, w, h, x, y = box; out = np.zeros(shape, bool)
    out[y:y+h, x:x+w] = cv2.resize(m.astype(np.uint8), (w,h), interpolation=cv2.INTER_NEAREST) > 0; return out

def render(tag, odir, fids, fs, uid, tcam, obj, gt_t, ours_pred, box, preds, variant, direction, fps, note, qpanel):
    tmp = odir/f'.tmp_{tag}'; tmp.mkdir(parents=True, exist_ok=True); [p.unlink() for p in tmp.glob('*.png')]
    J = {k: [] for k in ['ours'] + [m for m,_ in METHODS]}; n = 0
    for f in fids:
        img = cv2.imread(str(GTANN/uid/tcam/'frames'/f'{f}.jpg'))
        if img is None or f - fs >= len(ours_pred): continue
        H, W = img.shape[:2]
        gm = dec(gt_t.get(str(f)))
        om = ours_pred[f - fs]; om = place(om, box, (H,W)) if box else fit(om, img)
        panels = []
        p = paint(img, gm, BLUE); label(p, 'GT' + ('' if gm is not None else '  (no GT this frame)'), color=(255,200,120)); panels.append(p)
        p = paint(img, om, GREEN); j = iou(om, gm); label(p, 'Ours (XTrack)' + (f'  J={j:.2f}' if j is not None else ''), color=(120,255,120)); panels.append(p)
        if j is not None: J['ours'].append(j)
        for m, name in METHODS:
            if preds[m] is None:
                p = (img*0.35).astype(np.uint8); label(p, f'{name}  (no result for {direction})', color=(160,160,160))
            elif str(f) not in preds[m]:
                p = (img*0.6).astype(np.uint8); label(p, f'{name}  (no pred this frame)', color=(160,160,160))
            else:
                bm = fit(dec(preds[m][str(f)]), img); p = paint(img, bm, RED); j = iou(bm, gm)
                label(p, name + (f'  J={j:.2f}' if j is not None else ''), color=(120,120,255))
                if j is not None: J[m].append(j)
            panels.append(p)
        for p in panels: label(p, f'frame {f}  (+{(f-fs)/30:.1f}s)', y=H-10, sc=0.55)
        qp = letterbox(qpanel[0], H, W); label(qp, qpanel[1], color=(255,200,120)); label(qp, 'first-frame GT query mask', y=H-10, sc=0.55, color=(255,200,120))
        info = np.zeros((H, W, 3), np.uint8)
        for k, (txt, col) in enumerate([(f'object: {obj}', (255,255,255)), (f'{direction}   target view = {tcam}', (255,255,255)),
                                        (f'baselines: {variant}', (255,255,255)), (note, (200,200,200)), ('', None),
                                        ('blue  = GT / query mask', (255,200,120)), ('green = Ours (XTrack)', (120,255,120)), ('red   = baselines', (120,120,255))]):
            if txt: label(info, txt, y=40 + 30*k, color=col, sc=0.6, x=14)
        grid = np.concatenate([np.concatenate([qp] + panels[:3], 1), np.concatenate([info] + panels[3:], 1)], 0)
        cv2.imwrite(str(tmp/f'{n:06d}.png'), grid); n += 1
    if n == 0: return None
    out_mp4 = odir/f'{tag}.mp4'
    subprocess.run(['ffmpeg','-y','-loglevel','error','-framerate',str(fps),'-i',f'{tmp}/%06d.png','-c:v','libx264','-pix_fmt','yuv420p','-crf','18',str(out_mp4)], check=True)
    step = max(1, n // 4)
    sheet = np.concatenate([cv2.imread(str(tmp/f'{i:06d}.png')) for i in range(0, n, step)][:4], 0)
    cv2.imwrite(str(odir/f'{tag}_sheet.jpg'), cv2.resize(sheet, None, fx=0.5, fy=0.5), [cv2.IMWRITE_JPEG_QUALITY, 85])
    [p.unlink() for p in tmp.glob('*.png')]; tmp.rmdir()
    return {'n_frames': n, 'mean_J': {k: (round(float(np.mean(v)),3) if v else None) for k, v in J.items()}, 'video': str(out_mp4)}

import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument('--only', default=None, help='substring filter on uid/object, e.g. d6f0adfc'); _ap.add_argument('--sets', default='sparse,dense,full')
_a = _ap.parse_args(); SETS = _a.sets.split(',')
existing = json.load(open(OUT/'summary.json')) if (OUT/'summary.json').exists() and _a.only else []
summary = []
for dirname, direction in [('ego_to_exo_merged','ego2exo'), ('exo_to_ego_merged','exo2ego')]:
    for sdir in sorted(OURS.glob(f'{dirname}/*/*/')):
        if not (sdir/'take_metrics.json').exists(): continue
        uid, obj_safe = sdir.parts[-2], sdir.parts[-1]
        if (uid, obj_safe) not in win: print('SKIP (no window)', uid, obj_safe); continue
        if _a.only and _a.only not in f'{uid}/{obj_safe}': continue
        fs, ego, exo, obj = win[(uid, obj_safe)]
        tcam, tview = (exo, 'exo') if direction == 'ego2exo' else (ego, 'ego')
        g_sparse = json.load(open(GTANN/uid/'final_total_refined_masks_rle.json')).get(tcam, {}).get(obj, {})
        g_dense = json.load(open(DENSE/uid/'final_total_refined_masks_rle.json')).get(tcam, {}).get(obj, {}) if (DENSE/uid/'final_total_refined_masks_rle.json').exists() else {}
        ours_pred = [m > 127 for m in read_video(sdir/f'pred_{tview}.mp4', gray=True)]
        ours_gt = [m > 127 for m in read_video(sdir/f'gt_{tview}.mp4', gray=True)]
        # --- crop check: does our GT video match the full-frame GT? if not, recover the crop box
        chk = [int(k) for k in g_sparse if fs <= int(k) < fs + len(ours_gt)]
        box = None
        if chk:
            full0 = cv2.imread(str(GTANN/uid/tcam/'frames'/f'{chk[0]}.jpg'))
            v = [iou(fit(ours_gt[f-fs], full0), dec(g_sparse[str(f)])) for f in chk[:6]]
            if np.nanmean(v) < 0.5:
                ov = read_video(sdir/f'{tview}_overlay_pred.mp4')
                cands = [recover_crop(ov[f-fs], ours_pred[f-fs], cv2.cvtColor(cv2.imread(str(GTANN/uid/tcam/'frames'/f'{f}.jpg')), cv2.COLOR_BGR2GRAY)) for f in chk[:3]]
                box = max(cands, key=lambda b: b[0])
                v2 = [iou(place(ours_gt[f-fs], box, full0.shape[:2]), dec(g_sparse[str(f)])) for f in chk[:6]]
                print(f'  [crop] {uid[:8]} {obj_safe} {tview}: IoU {np.nanmean(v):.2f} -> {np.nanmean(v2):.2f} with box scale={box[1]/800:.2f} xy=({box[3]},{box[4]}) size={box[1]}x{box[2]}')
        # --- query panel: first-frame GT mask in the query view (our model's query = frame fs)
        qcam = ego if direction == 'ego2exo' else exo
        gq_dense = json.load(open(DENSE/uid/'final_total_refined_masks_rle.json')).get(qcam, {}).get(obj, {}) if (DENSE/uid/'final_total_refined_masks_rle.json').exists() else {}
        gq_sparse = json.load(open(GTANN/uid/'final_total_refined_masks_rle.json')).get(qcam, {}).get(obj, {})
        fq, qm = fs, None
        if str(fs) in gq_dense: qm = dec(gq_dense[str(fs)])
        else:
            cand = sorted(int(k) for k in gq_sparse if int(k) >= fs)
            if cand: fq, qm = cand[0], dec(gq_sparse[str(cand[0])])
        qimg = cv2.imread(str(GTANN/uid/qcam/'frames'/f'{fq}.jpg'))
        qpanel = (paint(qimg, qm, BLUE), f'Query view ({qcam})  frame {fq}')
        for variant in VARIANTS:
            # ---- sparse (gtann) ----
            preds = {}
            for m, _ in METHODS:
                hits = list((RES/m).glob(f'rebuttal_{direction}_gtann_{variant}/{uid}/{exo}__{ego}/{obj_safe}/chunk_*/predictions.json'))
                preds[m] = json.load(open(hits[0])) if hits else None
            fids = sorted({int(k) for p in preds.values() if p for k in p if fs <= int(k) < fs + len(ours_pred)})
            tag = f'{uid[:8]}_{obj_safe}_{direction}_{variant}_sparse'
            r = None if 'sparse' not in SETS else render(tag, OUT/direction, fids, fs, uid, tcam, obj, g_sparse, ours_pred, box, preds, variant, direction, 2, 'sparse: 1 frame / 30 (gtann)', qpanel) if fids else None
            if r: summary.append({'direction':direction,'variant':variant,'set':'sparse','uid':uid,'object':obj,'target_cam':tcam,**r}); print(tag, r['n_frames'], r['mean_J'], flush=True)
            # ---- dense (subset1_500) ----
            preds = {}
            for m, _ in METHODS:
                merged = {}
                for pj in (RES/m).glob(f'rebuttal_{direction}_subset1_500_{variant}/{uid}/{exo}__{ego}/{obj_safe}/chunk_*/predictions.json'):
                    merged.update(json.load(open(pj)))
                preds[m] = merged or None
            if 'dense' in SETS and not all(p is None for p in preds.values()):
                fids = sorted({int(k) for p in preds.values() if p for k in p if fs <= int(k) < fs + len(ours_pred)})
                tag = f'{uid[:8]}_{obj_safe}_{direction}_{variant}_dense'
                r = render(tag, OUT/direction, fids, fs, uid, tcam, obj, g_dense or g_sparse, ours_pred, box, preds, variant, direction, 10, 'dense: subset1_500 chunks', qpanel)
                if r: summary.append({'direction':direction,'variant':variant,'set':'dense','uid':uid,'object':obj,'target_cam':tcam,**r}); print(tag, r['n_frames'], r['mean_J'], flush=True)
            # ---- full (every frame of the 600-frame window, single query at frame fs) ----
            if 'full' not in SETS: continue
            preds = {}
            for m, _ in METHODS:
                merged = {}
                for pj in (RES/m).glob(f'rebuttal_{direction}_full_*_{variant}/{uid}/{exo}__{ego}/{obj_safe}/chunk_*/predictions.json'):
                    merged.update(json.load(open(pj)))
                preds[m] = merged or None
            if all(p is None for p in preds.values()): continue
            fids = [f for f in range(fs, fs + len(ours_pred))]
            tag = f'{uid[:8]}_{obj_safe}_{direction}_{variant}_full'
            r = render(tag, OUT/direction, fids, fs, uid, tcam, obj, g_dense or g_sparse, ours_pred, box, preds, variant, direction, 10, 'full: every frame, single first-frame query', qpanel)
            if r: summary.append({'direction':direction,'variant':variant,'set':'full','uid':uid,'object':obj,'target_cam':tcam,**r}); print(tag, r['n_frames'], r['mean_J'], flush=True)
new = {r['video'] for r in summary}
summary = [r for r in existing if r['video'] not in new] + summary
json.dump(summary, open(OUT/'summary.json','w'), indent=1)
