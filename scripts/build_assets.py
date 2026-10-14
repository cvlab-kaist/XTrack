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
FPS = 30; CRF = '26'
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
def query_image(uid, qcam, obj, fs, out, zoom_if_smaller_than=0.45):
    """Query-view first frame with the GT query mask. If the object is small, the main image is a zoom crop
    (same aspect as the frame) around the object and the full frame is shown as an inset with the crop box."""
    g = gt_for(uid, qcam, obj)
    cand = sorted(int(k) for k in g if int(k) >= fs)
    fq = fs if str(fs) in g else (cand[0] if cand else fs)
    img = cv2.imread(str(GTANN/uid/qcam/'frames'/f'{fq}.jpg'))
    m = fit(dec(g.get(str(fq))), img)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), compose_query(img, m, zoom_if_smaller_than), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return fq

def compose_query(img, m, zoom_if_smaller_than=0.45):
    H, W = img.shape[:2]
    main = paint(img, m, GREEN, alpha=0.45, th=3)
    if m is not None and m.any():
        ys, xs = np.where(m); bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        if bw < zoom_if_smaller_than * W and bh < zoom_if_smaller_than * H:
            side = max(bw, bh) * 2.6; cw, ch = side * W / max(W, H), side * H / max(W, H)
            if W >= H: cw, ch = max(side * W / H, bw * 2.2), max(side, bh * 2.2)
            else: cw, ch = max(side, bw * 2.2), max(side * H / W, bh * 2.2)
            cw, ch = max(cw, 0.42 * W), max(ch, 0.42 * H)  # limit the zoom factor to ~2.4x so the crop stays sharp
            cw, ch = int(min(cw, W)), int(min(ch, H))
            cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
            x0 = int(np.clip(cx - cw // 2, 0, W - cw)); y0 = int(np.clip(cy - ch // 2, 0, H - ch))
            zoom = cv2.resize(main[y0:y0 + ch, x0:x0 + cw], (W, H), interpolation=cv2.INTER_CUBIC)
            full = paint(img, m, GREEN, alpha=0.45, th=2); cv2.rectangle(full, (x0, y0), (x0 + cw, y0 + ch), (0, 220, 255), 3)
            iw = int(W * 0.34); ih = int(H * 0.34); ins = cv2.resize(full, (iw, ih), interpolation=cv2.INTER_AREA)
            cv2.rectangle(ins, (0, 0), (iw - 1, ih - 1), (255, 255, 255), 2)
            zoom[H - ih - 8:H - 8, 8:8 + iw] = ins
            cv2.putText(zoom, 'zoom', (W - 62, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(zoom, 'zoom', (W - 62, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            main = zoom
    return main

QUAL_SHORT = {'set': 'val600_cherry_short_obj_p1_top30', 'exo2ego': [
    'rank_02_JF0.9908_b60db7b1_Instruction_manual_0_c36-41', 'rank_06_JF0.9859_517207e4_stainless_steel_tray_0_c0-5',
    'rank_07_JF0.9856_b50aba6e_chopping_board_0_c3-8', 'rank_10_JF0.9803_4356d71a_plastic_bowl_0_c0-5',
    'rank_14_JF0.9753_b511dfed_covid_19_test_package_0_c8-13', 'rank_15_JF0.9748_2d042a82_egg_bowl_0_c9-14',
    'rank_19_JF0.9653_af96da5b_cooked_egg_and_tomato_0_c40-45', 'rank_21_JF0.9561_fac6316e_White_chopping_board_0_c38-43',
    'rank_22_JF0.9554_2041eb29_White_flat_plate_0_c25-30', 'rank_23_JF0.9528_4356d71a_pancake_pan_0_c37-42',
    'rank_25_JF0.9480_09e9fcd9_fry_pan_0_c31-36', 'rank_28_JF0.9431_ede7b9c1_sterile_swab_package_1_c24-29',
    'rank_29_JF0.9380_6101e4e5_sugar_container_0_c9-14'], 'ego2exo': [
    'rank_04_JF0.9879_e5bfb696_Egg_Crate_0_c14-19', 'rank_05_JF0.9820_2ca53d8d_chopping_board_0_c31-36',
    'rank_08_JF0.9747_2041eb29_White_flat_plate_0_c0-5', 'rank_11_JF0.9731_ede7b9c1_sterile_swab_package_1_c4-9',
    'rank_13_JF0.9711_cd3aff9e_Pan_0_c26-31', 'rank_15_JF0.9634_2d042a82_egg_bowl_0_c40-45',
    'rank_16_JF0.9616_6fbe4821_bicycle_wheel_0_c30-35', 'rank_19_JF0.9574_bbfcc023_milk_gallon_0_c0-5',
    'rank_20_JF0.9538_2f53a0ce_CPR_dummy_c19-24', 'rank_23_JF0.9443_622995a4_Grind_Pepper_container_0_c24-29',
    'rank_26_JF0.9402_6101e4e5_hand_towel_0_c22-27', 'rank_29_JF0.9348_4356d71a_pancake_pan_0_c19-24',
    'rank_30_JF0.9347_6ded8a71_Instruction_sheet_0_c31-36']}

def build_qual_short():
    """Short-window samples straight from the HF videos: query = first frame of rgb_{q} + gt_{q} mask; video = ego|exo overlay_pred."""
    index = json.load(open(ST/'qual'/'index.json'))
    index = [r for r in index if not r.get('short')]
    for direction in ['exo2ego', 'ego2exo']:
        dirname = 'exo_to_ego' if direction == 'exo2ego' else 'ego_to_exo'; q = 'exo' if direction == 'exo2ego' else 'ego'
        for name in QUAL_SHORT[direction]:
            sdir = HF/QUAL_SHORT['set']/dirname/name; w = json.load(open(sdir/'window.json'))
            tag = 'short_' + name.split('_', 3)[3]  # drop rank/JF prefix
            od = ST/'qual'/direction; od.mkdir(parents=True, exist_ok=True)
            img = read_video(sdir/f'rgb_{q}.mp4')[0]; m = read_video(sdir/f'gt_{q}.mp4', gray=True)[0] > 127
            cv2.imwrite(str(od/f'{tag}_query.jpg'), compose_query(img, m), [cv2.IMWRITE_JPEG_QUALITY, 90])
            e = read_video(sdir/'ego_overlay_pred.mp4'); x = read_video(sdir/'exo_overlay_pred.mp4'); H = e[0].shape[0]
            encode([np.concatenate([a, cv2.resize(b, (int(b.shape[1] * H / b.shape[0]), H))], 1) for a, b in zip(e, x)], od/f'{tag}.mp4')
            index.append({'direction': direction, 'uid': w['take_uid'], 'object': w['object'], 'tag': tag, 'query_cam': q, 'query_frame': None,
                          'frame_start': None, 'n_frames': len(e), 'short': True, 'rank': w.get('rank'), 'JF': round(w.get('JF', 0), 3),
                          'video': f'static/qual/{direction}/{tag}.mp4', 'query': f'static/qual/{direction}/{tag}_query.jpg'})
            print('qual-short', direction, tag, len(e), flush=True)
    json.dump(index, open(ST/'qual'/'index.json', 'w'), indent=1)

def rebuild_queries():
    """Regenerate only the query images (motivation + qualitative) without touching the videos."""
    uid, obj_safe = 'd6f0adfc-5c66-4c3e-95b0-902d12d7bb86', 'cutting_board_0'
    fs, ego, exo, obj = win[(uid, obj_safe)]
    query_image(uid, exo, obj, fs, ST/'motivation'/'query.jpg')
    index = json.load(open(ST/'qual'/'index.json'))
    for s in index:
        if s.get('short'): continue
        fs, ego, exo, obj = win[(s['uid'], s['tag'].split('_', 1)[1])]
        query_image(s['uid'], s['query_cam'], obj, fs, PP/s['query']); print('query', s['tag'])

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
# diverse hand-picked set (source folder, take uid, object); cutting_board is reserved for the motivation figure
QUAL_SAMPLES = {
    'ego2exo': [('val600_cherry_whole', 'b74169ba-77b0-447d-aeff-cb7934ff6711', 'sheet_music'),
                ('val600_cherry_whole', 'd8db054f-b5ee-41c3-ba66-cb84ed2d286e', 'piano'),
                ('val600_cherry_whole', 'b63d213f-25ae-41bb-8719-01efd84dbe17', 'hoop'),
                ('val600_cherry', '4d799166-0362-4daa-9f26-72e27a4feb31', 'egg_crate_0'),
                ('val600_cherry', '2d042a82-6409-4986-8bfb-af031f5e061e', 'scrambled_egg_0'),
                ('val600_cherry', '79a728ac-d543-4a7a-bf6d-132a676ca685', 'violin_bow'),
                ('val600_cherry', '5c12c6e3-d34e-4e9a-bd7f-74aed58427a7', 'disposable_swab_with_package_0'),
                ],
    'exo2ego': [('val600_cherry_whole', '2ca53d8d-0120-4dad-a856-c8f20aaa2cc4', 'chopping_board_0'),
                ('val600_cherry_whole', '0fe5b647-cdd0-43e9-8710-b33a2e0f83ef', 'piano'),
                ('val600_cherry_whole', 'ae2207b2-9b33-4f6a-aeb5-e2718a253c3b', 'sheet_music'),
                ('val600_cherry', '32dfc6fe-c801-4ce5-bb39-8841fca7a075', 'soccer_ball'),
                ('val600_cherry', '9e08e1e7-179a-4dac-bd0e-6e98e47407d2', 'hoop'),
                ('val600_cherry', 'dc1600e9-dd79-4700-882a-9f53cdd29d20', 'CPR_dummy'),
                ('val600_cherry', '2d042a82-6409-4986-8bfb-af031f5e061e', 'scrambled_egg_0')],
}

def build_qual():
    index = []
    import shutil as _sh
    if (ST/'qual').exists(): _sh.rmtree(ST/'qual')
    for dirname, direction in [('ego_to_exo_merged', 'ego2exo'), ('exo_to_ego_merged', 'exo2ego')]:
        for src, uid, obj_safe in QUAL_SAMPLES[direction]:
            sdir = HF/src/dirname/uid/obj_safe
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
    if 'queries' in what: rebuild_queries()
    if 'short' in what: build_qual_short()
