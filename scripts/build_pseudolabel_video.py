"""Illustrative video of the pseudo-label curation pipeline (run with /data2/envs/v2sam/bin/python).

Take d6f0adfc (cutting board), ego view, frames 8520-9120. Sparse 1-FPS GT anchors (every 30 frames) are propagated
with SAM2 forward (from the left anchor) and backward (from the right anchor) inside every 30-frame chunk.
Chunks with low forward/backward mIoU are "unreliable"; the curated dense labels (lmeec_data) show what the pipeline kept.
Panels: forward | backward | exo pseudo-GT reference | curated pseudo-label, plus a chunk timeline.
Output: static/method/pseudolabel.mp4 (30 fps).
"""
import os, sys, json, shutil, tempfile, subprocess
from pathlib import Path
import cv2, numpy as np, torch
from pycocotools import mask as mu

SAM2_ROOT = '/home/ubuntu/dahyun/_Nips26/sam2_upstream'; sys.path.insert(0, SAM2_ROOT)
from sam2.build_sam import build_sam2_video_predictor  # noqa

PP = Path('/home/ubuntu/dahyun/_Nips26/project_page'); OUT = PP/'static/method'; OUT.mkdir(parents=True, exist_ok=True)
GTANN = Path('/data3/_Nips26/lmeec_gtann_data'); DENSE = Path('/data3/_Nips26/lmeec_data')
UID, EGO, EXO, OBJ = 'd6f0adfc-5c66-4c3e-95b0-902d12d7bb86', 'aria01_214-1', 'gp03', 'cutting board_0'
F0, F1, STEP = 8520, 9120, 30
PINK, CYAN, GREEN, GREY = (170, 90, 240), (230, 190, 60), (0, 200, 0), (140, 140, 140)  # BGR
S = 0.62; H = int(448 * S)

def dec(r): return None if r is None else mu.decode({'size': r['size'], 'counts': r['counts'].encode()}).astype(bool)
def iou(a, b):
    if a is None or b is None: return 0.0
    u = (a | b).sum(); return float((a & b).sum() / u) if u else 1.0
def paint(img, m, color, alpha=0.5, th=2):
    out = img.copy()
    if m is None or not m.any(): return out
    out[m] = (img[m] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.drawContours(out, cs, -1, color, th)
def label(img, txt, y=22, color=(255, 255, 255), sc=0.55, x=8):
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, color, 1, cv2.LINE_AA); return img

def propagate(pred, frames_dir, fids, seed):
    tmp = tempfile.mkdtemp(prefix='pl_')
    try:
        for i, f in enumerate(fids): os.symlink(str(frames_dir / f'{f}.jpg'), f'{tmp}/{i:07d}.jpg')
        out = {}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            st = pred.init_state(video_path=tmp, offload_video_to_cpu=False, async_loading_frames=True)
            pred.add_new_mask(inference_state=st, frame_idx=0, obj_id=0, mask=seed.astype(np.uint8))
            for fi, _, logits in pred.propagate_in_video(st, start_frame_idx=0):
                out[fids[fi]] = (logits[0] > 0).cpu().numpy().squeeze().astype(bool)
            pred.reset_state(st)
        return out
    finally: shutil.rmtree(tmp, ignore_errors=True)

def main():
    g_anchor = json.load(open(GTANN/UID/'final_total_refined_masks_rle.json'))[EGO][OBJ]
    g_dense = json.load(open(DENSE/UID/'final_total_refined_masks_rle.json'))
    ego_final, exo_ref = g_dense[EGO][OBJ], g_dense[EXO][OBJ]
    anchors = [f for f in range(F0, F1 + 1, STEP) if str(f) in g_anchor]
    print('anchors', anchors, flush=True)
    pred = build_sam2_video_predictor(config_file='configs/sam2.1/sam2.1_hiera_l.yaml', ckpt_path=f'{SAM2_ROOT}/checkpoints/sam2.1_hiera_large.pt', device='cuda')
    fwd, bwd, chunks = {}, {}, []
    for a, b in zip(anchors, anchors[1:]):
        fids = list(range(a, b + 1))
        fwd.update(propagate(pred, GTANN/UID/EGO/'frames', fids, dec(g_anchor[str(a)])))
        bwd.update(propagate(pred, GTANN/UID/EGO/'frames', fids[::-1], dec(g_anchor[str(b)])))
        for ca in range(a, b, STEP):
            cb = min(ca + STEP, b); sub = [f for f in range(ca + 1, cb) ]
            ne = lambda m_: m_ is not None and m_.any()
            cand = [f for f in sub if ne(fwd.get(f)) or ne(bwd.get(f))]
            ious = [iou(fwd[f], bwd[f]) for f in cand]
            m = float(np.mean(ious)) if ious else 1.0
            fin = {f: dec(ego_final.get(str(f))) for f in sub}
            n_fin = sum(1 for x in fin.values() if ne(x))
            if not cand: dec_ = 'no object'
            elif m >= 0.7: dec_ = 'reliable'
            elif n_fin < len(sub) * 0.5: dec_ = 'discarded'
            else:
                af = np.mean([iou(fin[f], fwd[f]) for f in sub if ne(fin[f])]); ab = np.mean([iou(fin[f], bwd[f]) for f in sub if ne(fin[f])])
                dec_ = 'refined: forward' if af >= ab else 'refined: backward'
            chunks.append({'a': ca, 'b': cb, 'miou': m, 'decision': dec_}); print(ca, cb, f'mIoU={m:.2f}', dec_, flush=True)
    json.dump(chunks, open(OUT/'pseudolabel_chunks.json', 'w'), indent=1)

    # ---------------- render ----------------
    tmp = Path('/data3/_Nips26/project_page_vis/.tmp_frames/pseudolabel'); tmp.mkdir(parents=True, exist_ok=True); [p.unlink() for p in tmp.glob('*.png')]
    W4 = int(448 * S); WX = int(796 * S); TL_H = 46; GAP = 6
    total_w = 3 * W4 + WX + 3 * GAP
    col = {'reliable': (90, 190, 90), 'no object': (225, 225, 225), 'refined: forward': PINK, 'refined: backward': CYAN, 'discarded': GREY}
    n = 0
    for f in range(F0, F1 + 1):
        ego = cv2.imread(str(GTANN/UID/EGO/'frames'/f'{f}.jpg')); exo = cv2.imread(str(GTANN/UID/EXO/'frames'/f'{f}.jpg'))
        ch = next((c for c in chunks if c['a'] <= f <= c['b']), chunks[-1])
        is_anchor = str(f) in g_anchor
        pf = cv2.resize(paint(ego, fwd.get(f), PINK), (W4, H)); label(pf, 'Forward propagation (SAM2)', color=(220, 170, 255))
        pb = cv2.resize(paint(ego, bwd.get(f), CYAN), (W4, H)); label(pb, 'Backward propagation (SAM2)', color=(255, 230, 150))
        px = cv2.resize(paint(exo, dec(exo_ref.get(str(f))), GREEN), (WX, H)); label(px, 'Exo view reference (static camera)', color=(150, 255, 150))
        fm = dec(ego_final.get(str(f))); pr = cv2.resize(paint(ego, fm, GREEN), (W4, H)); label(pr, 'Curated pseudo-label', color=(150, 255, 150))
        if fm is None or not fm.any():
            both_empty = not ((fwd.get(f) is not None and fwd[f].any()) or (bwd.get(f) is not None and bwd[f].any()))
            label(pr, 'no mask (object absent)' if both_empty else 'no mask (candidate discarded)', y=H - 12, color=(200, 200, 200))
        if is_anchor:
            for p in (pf, pb, pr): cv2.rectangle(p, (0, 0), (W4 - 1, H - 1), (0, 220, 255), 4); label(p, 'GT anchor (1 FPS)', y=H - 12, color=(0, 220, 255))
        else:
            label(pf, f"fwd/bwd mIoU = {ch['miou']:.2f}  ({ch['decision']})", y=H - 12, color=(255, 255, 255))
        row = np.full((H, total_w, 3), 255, np.uint8); x = 0
        for p in (pf, pb, px, pr): row[:, x:x + p.shape[1]] = p; x += p.shape[1] + GAP
        # timeline
        tl = np.full((TL_H, total_w, 3), 255, np.uint8)
        for c in chunks:
            x0 = int((c['a'] - F0) / (F1 - F0) * total_w); x1 = int((c['b'] - F0) / (F1 - F0) * total_w)
            cv2.rectangle(tl, (x0, 8), (x1 - 1, 30), col[c['decision']], -1); cv2.rectangle(tl, (x0, 8), (x1 - 1, 30), (255, 255, 255), 1)
        xc = int((f - F0) / (F1 - F0) * total_w); cv2.line(tl, (xc, 4), (xc, 34), (0, 0, 0), 2)
        cv2.putText(tl, 'chunks: green = fwd/bwd agree   pink/yellow = low agreement -> refined (fwd/bwd kept after cross-view + semantic verification)   grey = discarded   white = no object',
                    (6, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.imwrite(str(tmp/f'{n:06d}.png'), np.concatenate([row, tl], 0)); n += 1
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '30', '-i', f'{tmp}/%06d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '24',
                    '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-movflags', '+faststart', str(OUT/'pseudolabel.mp4')], check=True)
    shutil.rmtree(tmp); print('wrote', OUT/'pseudolabel.mp4', n, 'frames')

if __name__ == '__main__':
    main()
