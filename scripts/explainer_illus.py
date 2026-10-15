"""Pre-render illustration images for the explainer's grounding scene (needs pycocotools -> run with /data2/envs/v2sam/bin/python)."""
import json, cv2, numpy as np
from pathlib import Path
from pycocotools import mask as mu
G = Path('/data3/_Nips26/lmeec_gtann_data'); OUT = Path('/data3/_Nips26/project_page_vis/explainer/illus'); OUT.mkdir(parents=True, exist_ok=True)
GREEN, ORANGE, YELLOW = (0, 200, 0), (0, 140, 255), (0, 220, 255)
def dec(r): return mu.decode({'size': r['size'], 'counts': r['counts'].encode()}).astype(bool)
def paint(img, m, color, alpha=0.5, th=3):
    out = img.copy(); out[m] = (img[m] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); return cv2.drawContours(out, cs, -1, color, th)
# --- hard negative: similar-class distractor. query = front wheel (correct) vs. back wheel (mismatched), same frame
u = 'c692c40e'; u = [d for d in G.iterdir() if d.name.startswith(u)][0].name; g = json.load(open(G/u/'final_total_refined_masks_rle.json')); f = 1050
EXO, EGO, POS, NEG = 'cam01', 'aria04_214-1', 'front wheel_0', 'back wheel_0'
exo = cv2.imread(str(G/u/EXO/'frames'/f'{f}.jpg')); ego = cv2.imread(str(G/u/EGO/'frames'/f'{f}.jpg'))
cv2.imwrite(str(OUT/'hn_query_pos.jpg'), paint(exo, dec(g[EXO][POS][str(f)]), GREEN))
cv2.imwrite(str(OUT/'hn_query_neg.jpg'), paint(exo, dec(g[EXO][NEG][str(f)]), ORANGE))
cv2.imwrite(str(OUT/'hn_target_pos.jpg'), paint(ego, dec(g[EGO][POS][str(f)]), GREEN))
cv2.imwrite(str(OUT/'hn_target_neg.jpg'), paint(ego, dec(g[EGO][NEG][str(f)]), ORANGE))
# --- local context crop: small object (hoop) in the ego view; full frame with box + enlarged crop
u2 = 'b63d213f-25ae-41bb-8719-01efd84dbe17'; g2 = json.load(open(G/u2/'final_total_refined_masks_rle.json'))
best = None
for k, r in g2['aria01_214-1']['hoop'].items():
    m = dec(r); a = m.sum()
    if 150 < a and (best is None or a < best[1]) and 300 <= int(k) <= 900: best = (int(k), a)
f2 = best[0]; m = dec(g2['aria01_214-1']['hoop'][str(f2)]); img = cv2.imread(str(G/u2/'aria01_214-1/frames'/f'{f2}.jpg'))
ys, xs = np.where(m); cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2; s = int(max(xs.max() - xs.min(), ys.max() - ys.min()) * 2.2); s = max(s, 96)
x0, y0 = int(np.clip(cx - s // 2, 0, 448 - s)), int(np.clip(cy - s // 2, 0, 448 - s))
full = paint(img, m, GREEN, th=2); cv2.rectangle(full, (x0, y0), (x0 + s, y0 + s), YELLOW, 3)
cv2.imwrite(str(OUT/'lc_full.jpg'), full); cv2.imwrite(str(OUT/'lc_plain.jpg'), img)
crop = cv2.resize(paint(img, m, GREEN, th=2)[y0:y0 + s, x0:x0 + s], (448, 448), interpolation=cv2.INTER_CUBIC); cv2.rectangle(crop, (0, 0), (447, 447), YELLOW, 6)
cv2.imwrite(str(OUT/'lc_crop.jpg'), crop)
json.dump({'hn_frame': f, 'lc_frame': f2, 'lc_box': [x0, y0, s]}, open(OUT/'meta.json', 'w')); print('ok', f, f2, (x0, y0, s))
