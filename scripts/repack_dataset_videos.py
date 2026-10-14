"""Re-pack the curation example videos without padding and with uniform panel labels.

VGGT videos (1776x250 after caption crop): panels = exo query (444 wide) | forward | backward | VGGT selection,
each ego panel is a 250x250 image left-aligned in a 444-wide black panel. The selection panel's mask colour
(red = forward candidate, blue = backward candidate) tells which candidate VGGT picked.
DINO video (900x296): three 300-wide panels, baked labels on top and a frame counter at the bottom.
"""
import subprocess, shutil
from pathlib import Path
import cv2, numpy as np

D = Path('/home/ubuntu/dahyun/_Nips26/project_page/static/dataset'); TMP = Path('/data3/_Nips26/project_page_vis/.tmp_frames/repack')
GAP = 6; BAND = 38

def read(p):
    cap = cv2.VideoCapture(str(p)); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    return fr
def band(img, txt, color=(255, 255, 255)):
    out = img.copy(); out[:BAND] = 0
    cv2.putText(out, txt, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA); return out
def pick(panel):
    b, g, r = panel[..., 0].astype(int), panel[..., 1].astype(int), panel[..., 2].astype(int)
    red = ((r > 140) & (g < 90) & (b < 90)).sum(); blue = ((b > 140) & (r < 90) & (g < 120)).sum()
    return 'forward' if red > blue else 'backward' if blue > 0 else None
def encode(frames, out, fps):
    TMP.mkdir(parents=True, exist_ok=True); [p.unlink() for p in TMP.glob('*.png')]
    for i, f in enumerate(frames):
        if f.shape[0] % 2 or f.shape[1] % 2: f = cv2.copyMakeBorder(f, 0, f.shape[0] % 2, 0, f.shape[1] % 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.imwrite(str(TMP / f'{i:06d}.png'), f)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', f'{TMP}/%06d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '22', '-movflags', '+faststart', str(out)], check=True)
    shutil.rmtree(TMP); print('wrote', out, len(frames), frames[0].shape)

def repack_vggt(src, dst, n_max=None):
    fr = read(src)[:n_max]; out = []
    for im in fr:
        exo = im[:, 0:444]; ego = [im[:, 444 * k:444 * k + 250] for k in (1, 2, 3)]
        sel = pick(ego[2]); sel_txt = f'VGGT selection: {sel}' if sel else 'VGGT selection'
        panels = [band(exo, 'Exo view (query)'), band(ego[0], 'Forward candidate', (120, 120, 255)), band(ego[1], 'Backward candidate', (255, 180, 120)), band(ego[2], sel_txt, (120, 255, 120))]
        row = np.full((250, sum(p.shape[1] for p in panels) + GAP * 3, 3), 255, np.uint8); x = 0
        for p in panels: row[:, x:x + p.shape[1]] = p; x += p.shape[1] + GAP
        out.append(row)
    encode(out, dst, 15)

def to_green(panel):
    """Recolour the red (forward) overlay of a panel to green: used to show the GT anchor the forward pass starts from."""
    out = panel.copy(); b, g, r = out[..., 0].astype(int), out[..., 1].astype(int), out[..., 2].astype(int)
    m = (r > 120) & (r > g + 50) & (r > b + 50)
    out[m, 1], out[m, 2] = out[m, 2], out[m, 1]  # swap R<->G
    return out

def repack_dino(src, dst, n_max=None):
    fr = read(src)[:n_max]; out = []
    anchor = band(to_green(fr[0][:-20, 0:300]), 'Ego anchor (first-frame GT)', (120, 255, 120))
    for im in fr:
        im = im[:-20]  # drop baked frame counter
        p = [im[:, 300 * k:300 * (k + 1)] for k in range(3)]
        panels = [anchor, band(p[0], 'Forward candidate', (120, 120, 255)), band(p[1], 'Backward candidate', (255, 180, 120)), band(p[2], 'DINOv3 check vs. anchor', (120, 255, 120))]
        row = np.full((im.shape[0], 1200 + GAP * 3, 3), 255, np.uint8); x = 0
        for q in panels: row[:, x:x + 300] = q; x += 300 + GAP
        out.append(row)
    encode(out, dst, 15)

if __name__ == '__main__':
    import sys
    if 'dino' not in sys.argv[1:]:
        repack_vggt(D / 'vggt_basketball.mp4', D / 'vggt_basketball_packed.mp4')
        repack_vggt(D / 'vggt_bike.mp4', D / 'vggt_bike_packed.mp4')
        repack_vggt(D / 'vggt_soccer.mp4', D / 'vggt_soccer_packed.mp4', 110)
    repack_dino(D / 'dino_soccer.mp4', D / 'dino_soccer_packed.mp4', 110)
