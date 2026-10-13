"""Build project-page video/image assets.

A. Motivation (cutting_board_0, exo2ego, take d6f0adfc, frames 8521-9120, query = exo gp03, target = ego aria01_214-1):
   static/motivation/query.jpg                      exo first frame + GT query mask (green)
   static/motivation/gt.mp4                         ego target, GT mask (green)
   static/motivation/ours.mp4                       ego target, XTrack prediction (green)
   static/motivation/{method}_{variant}.mp4         ego target, baseline prediction (red)
B. Ours qualitative (val600_cherry_whole): per sample
   static/qual/{dir}/{uid8}_{obj}_query.jpg         query-view first frame + GT query mask (green)
   static/qual/{dir}/{uid8}_{obj}.mp4               hconcat(ego_overlay_pred, exo_overlay_pred)
All videos at 30 fps (= 2x the 15-fps HF videos), h264 yuv420p.
"""
import csv, json, subprocess, shutil
from pathlib import Path
import cv2, numpy as np
from pycocotools import mask as mu

PP = Path('/home/ubuntu/dahyun/_Nips26/project_page'); ST = PP/'static'
HF = Path('/data3/_Nips26/xtrack_hf'); RES = Path('/data3/_Nips26/result')
GTANN = Path('/data3/_Nips26/lmeec_gtann_data'); DENSE = Path('/data3/_Nips26/lmeec_data')
FPS = 30; CRF = '24'
GREEN, RED = (0, 200, 0), (0, 0, 230)   # BGR

win = {}
for r in csv.DictReader(open('/data3/_Nips26/egoexo_usable_chunks_val_600frames_refined_2obj.csv')):
    win[(r['take_uid'], r['shared_object'].replace(' ', '_').replace('/', '_'))] = (int(r['frame_start']), r['ego_cam_id'], r['exo_cam_id'], r['shared_object'])

def dec(r):
    if r is None: return None
    return mu.decode({'size': r['size'], 'counts': r['counts'].encode() if isinstance(r['counts'], str) else r['counts']}).astype(bool)
def fit(m, img):
    if m is None or m.shape == img.shape[:2]: return m
    return cv2.resize(m.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
def paint(img, m, color, alpha=0.5, th=2):
    m = fit(m, img); out = img.copy()
    if m is None or not m.any(): return out
    out[m] = (img[m] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.drawContours(out, cs, -1, color, th)
def read_video(p, gray=False):
    cap = cv2.VideoCapture(str(p)); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if gray else f)
    return fr
def encode(frames, out, fps=FPS):
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path('/data3/_Nips26/project_page_vis/.tmp_frames') / out.stem; tmp.mkdir(parents=True, exist_ok=True); [p.unlink() for p in tmp.glob('*.png')]
    for i, f in enumerate(frames): cv2.imwrite(str(tmp / f'{i:06d}.png'), f)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', f'{tmp}/%06d.png', '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p', '-crf', CRF, '-movflags', '+faststart', str(out)], check=True)
    shutil.rmtree(tmp)
def gt_for(uid, cam, obj):
    g = {}
    if (DENSE/uid/'final_total_refined_masks_rle.json').exists():
        g.update(json.load(open(DENSE/uid/'final_total_refined_masks_rle.json')).get(cam, {}).get(obj, {}))
    gs = json.load(open(GTANN/uid/'final_total_refined_masks_rle.json')).get(cam, {}).get(obj, {})
    for k, v in gs.items(): g.setdefault(k, v)
    return g
def query_image(uid, qcam, obj, fs, out):
    g = gt_for(uid, qcam, obj)
    cand = sorted(int(k) for k in g if int(k) >= fs)
    fq = fs if str(fs) in g else (cand[0] if cand else fs)
    img = cv2.imread(str(GTANN/uid/qcam/'frames'/f'{fq}.jpg'))
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), paint(img, dec(g.get(str(fq))), GREEN, alpha=0.45, th=3), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return fq

# ───────────────────────── A. motivation ─────────────────────────
def build_motivation():
    uid, obj_safe = 'd6f0adfc-5c66-4c3e-95b0-902d12d7bb86', 'cutting_board_0'
    fs, ego, exo, obj = win[(uid, obj_safe)]; n = 598
    od = ST/'motivation'; od.mkdir(parents=True, exist_ok=True)
    fq = query_image(uid, exo, obj, fs, od/'query.jpg'); print('motivation query frame', fq)
    frames = [cv2.imread(str(GTANN/uid/ego/'frames'/f'{f}.jpg')) for f in range(fs, fs + n)]
    gt = gt_for(uid, ego, obj)
    encode([paint(im, dec(gt.get(str(fs + i))), GREEN) for i, im in enumerate(frames)], od/'gt.mp4')
    ours = [m > 127 for m in read_video(HF/'val600_cherry/exo_to_ego_merged'/uid/obj_safe/'pred_ego.mp4', gray=True)]
    encode([paint(im, ours[i], GREEN) for i, im in enumerate(frames)], od/'ours.mp4')
    for m, name in [('objectrelator', 'objectrelator'), ('o-mama', 'omama'), ('lm-eec', 'lmeec'), ('v2-sam', 'v2sam')]:
        for v in ['sam2qt', 'ffmsam2']:
            pj = next((RES/m).glob(f'rebuttal_exo2ego_full_cutting_board_{v}/{uid}/{exo}__{ego}/{obj_safe}/chunk_*/predictions.json'))
            pred = json.load(open(pj))
            encode([paint(im, dec(pred.get(str(fs + i))), RED) for i, im in enumerate(frames)], od/f'{name}_{v}.mp4')
            print('motivation', name, v)
    # plain RGB of the target view (for an optional "input" panel)
    encode(frames, od/'ego_rgb.mp4')

# ───────────────────────── B. ours qualitative ─────────────────────────
def build_qual():
    index = []
    for dirname, direction in [('ego_to_exo_merged', 'ego2exo'), ('exo_to_ego_merged', 'exo2ego')]:
        for sdir in sorted((HF/'val600_cherry_whole'/dirname).glob('*/*/')):
            uid, obj_safe = sdir.parts[-2], sdir.parts[-1]
            if (uid, obj_safe) not in win: print('SKIP', uid, obj_safe); continue
            fs, ego, exo, obj = win[(uid, obj_safe)]
            qcam = ego if direction == 'ego2exo' else exo
            tag = f'{uid[:8]}_{obj_safe}'; od = ST/'qual'/direction
            fq = query_image(uid, qcam, obj, fs, od/f'{tag}_query.jpg')
            e = read_video(sdir/'ego_overlay_pred.mp4'); x = read_video(sdir/'exo_overlay_pred.mp4')
            H = e[0].shape[0]
            frames = [np.concatenate([a, cv2.resize(b, (int(b.shape[1] * H / b.shape[0]), H))], 1) for a, b in zip(e, x)]
            encode(frames, od/f'{tag}.mp4')
            tm = json.load(open(sdir/'take_metrics.json'))
            index.append({'direction': direction, 'uid': uid, 'object': obj, 'tag': tag, 'query_cam': qcam, 'query_frame': fq,
                          'frame_start': fs, 'n_frames': len(frames), 'ego_JF': round(tm['ego']['JF'], 3), 'exo_JF': round(tm['exo']['JF'], 3),
                          'video': f'static/qual/{direction}/{tag}.mp4', 'query': f'static/qual/{direction}/{tag}_query.jpg'})
            print('qual', direction, tag, len(frames), flush=True)
    json.dump(index, open(ST/'qual'/'index.json', 'w'), indent=1)

if __name__ == '__main__':
    import sys
    what = sys.argv[1:] or ['motivation', 'qual']
    if 'motivation' in what: build_motivation()
    if 'qual' in what: build_qual()
