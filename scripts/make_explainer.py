"""Render a ~2 min narrated explainer video for XTrack from the project-page assets.

Usage: /data2/envs/multiworld/bin/python scripts/make_explainer.py [--no-tts]   (--no-tts = silent video with fixed scene lengths)
Output: /data3/_Nips26/project_page_vis/explainer/xtrack_explainer.mp4 (1920x1080, 30 fps, AAC narration)
Narration: edge-tts (voice en-US-AriaNeural) via /data2/envs/_build/ettsvenv; scene length = narration length + padding.
"""
import json, subprocess, sys, shutil
from pathlib import Path
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

PP = Path('/home/ubuntu/dahyun/_Nips26/project_page'); ST = PP/'static'; FULL = Path('/data3/_Nips26/project_page_vis/fullres_backup')
OUT = Path('/data3/_Nips26/project_page_vis/explainer'); OUT.mkdir(parents=True, exist_ok=True)
TTS = '/data2/envs/_build/ettsvenv/bin/edge-tts'; VOICE = 'en-US-AriaNeural'
W, H, FPS = 1920, 1080, 30
F = {k: f'/usr/share/fonts/truetype/liberation2/LiberationSans-{v}.ttf' for k, v in [('r', 'Regular'), ('b', 'Bold'), ('i', 'Italic')]}
YELLOW_PIL = (230, 180, 0)
INK, MUTED, GREEN, RED, BLUE, BG = (34, 34, 34), (110, 110, 110), (31, 184, 74), (224, 49, 49), (59, 91, 219), (255, 255, 255)

# ───────────────────────── helpers ─────────────────────────
def font(size, w='r'): return ImageFont.truetype(F[w], size)
def canvas(): return Image.new('RGB', (W, H), BG)
def text(img, xy, s, size=40, w='r', fill=INK, anchor='la', maxw=None, line=1.25):
    d = ImageDraw.Draw(img); f = font(size, w)
    if maxw:
        words, lines, cur = s.split(), [], ''
        for wd in words:
            t = (cur + ' ' + wd).strip()
            if d.textlength(t, font=f) <= maxw: cur = t
            else: lines.append(cur); cur = wd
        lines.append(cur); x, y = xy
        for ln in lines: d.text((x, y), ln, font=f, fill=fill, anchor=anchor); y += int(size * line)
        return y
    d.text(xy, s, font=f, fill=fill, anchor=anchor); return xy[1] + int(size * line)
def paste(img, arr_bgr, box, fit='contain', radius=14):
    """Paste a BGR frame into box=(x,y,w,h) keeping aspect; returns the pasted rect."""
    x, y, bw, bh = box; h0, w0 = arr_bgr.shape[:2]; s = min(bw / w0, bh / h0) if fit == 'contain' else max(bw / w0, bh / h0)
    nw, nh = (int(np.ceil(w0 * s)), int(np.ceil(h0 * s))) if fit == 'cover' else (int(w0 * s), int(h0 * s))
    fr = cv2.resize(arr_bgr, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    if fit == 'cover':
        ox, oy = (nw - bw) // 2, (nh - bh) // 2; fr = fr[oy:oy + bh, ox:ox + bw]; nh, nw = fr.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    if radius:
        m = Image.new('L', (nw, nh), 0); ImageDraw.Draw(m).rounded_rectangle((0, 0, nw - 1, nh - 1), radius, fill=255)
    else: m = None
    px, py = x + (bw - nw) // 2, y + (bh - nh) // 2; img.paste(pil, (px, py), m); return (px, py, nw, nh)
def img_bgr(p): return cv2.imread(str(p))
def read_video(p):
    cap = cv2.VideoCapture(str(p)); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    return fr
def vf(frames, t, fps_src=30, loop=True):
    i = int(t * fps_src); return frames[i % len(frames)] if loop else frames[min(i, len(frames) - 1)]
def ease(t, t0, dur=0.5): return float(np.clip((t - t0) / dur, 0, 1))
def fade(img, a):
    if a >= 1: return img
    return Image.blend(Image.new('RGB', img.size, BG), img, a)
def header(img, title, sub=None):
    text(img, (96, 64), title, 54, 'b'); ImageDraw.Draw(img).line((96, 134, 96 + 70, 134), fill=GREEN, width=6)
    if sub: text(img, (96, 150), sub, 30, 'r', MUTED, maxw=W - 192)
def caption(img, xy, s, size=26, fill=MUTED): text(img, xy, s, size, 'r', fill, anchor='ma')
def bullets(img, x, y, items, t, t0, size=34, gap=1.5, maxw=1100, stagger=0.9):
    for k, it in enumerate(items):
        a = ease(t, t0 + k * stagger, 0.5)
        if a <= 0: continue
        col = tuple(int(BG[c] + (INK[c] - BG[c]) * a) for c in range(3)); gcol = tuple(int(BG[c] + (GREEN[c] - BG[c]) * a) for c in range(3))
        ImageDraw.Draw(img).ellipse((x, y + size * 0.35, x + 14, y + size * 0.35 + 14), fill=gcol)
        y = text(img, (x + 32, y), it, size, 'r', col, maxw=maxw) + int(size * (gap - 1.25))
    return y

# ───────────────────────── narration ─────────────────────────
SCENES = [
 ('title', "XTrack: Repurposing Video Diffusion Transformers for Cross-View Temporal Object Correspondence. A NeurIPS 2026 paper from KAIST AI and Korea University."),
 ('task', "Given an ego video, an exo video, and a single first-frame query mask in one view, the goal is to recover temporally coherent masks of the same object in both views. We call this task cross-view temporal object correspondence."),
 ('motivation', "Existing frame-level matchers must be combined with a tracker. Track-then-match propagates the query and matches every frame independently, which produces temporally inconsistent matches and false positives when the object is absent. Match-then-track commits to one initial match and tracks each view on its own, so an unreliable match propagates and the object is lost once it reappears. XTrack instead predicts both views jointly over time."),
 ('method', "XTrack repurposes a pretrained video diffusion transformer. The ego and exo videos are placed side by side and fed as one input, and flow matching transports the RGB latents directly to mask latents. Cross-view matching and within-view tracking happen in the same attention, so evidence from one view resolves ambiguity in the other."),
 ('grounding', "To make grounding reliable, we add focal attention alignment, which supervises cross-view attention to land on the query object; hard negative conditioning, which forces the prediction to depend on the query identity; and a local context crop for small objects. Training proceeds in two stages: image-level pretraining with a cycle-consistency loss, followed by video-level fine-tuning."),
 ('curation', "Ego-Exo4D provides masks at only one frame per second. Our curation pipeline propagates the annotations forward and backward with SAM2, selects the consistent candidate by geometric verification against the exo view, and discards remaining errors by semantic verification against the ego anchor, yielding dense pseudo-labels at thirty frames per second."),
 ('results', "From a single query mask, XTrack keeps the object in both views through viewpoint changes, occlusion, and re-appearance, without any per-frame query."),
 ('closing', "Please visit our project page for more results. Thank you."),
]
FIXED = {'title': 5, 'task': 7, 'motivation': 20, 'method': 15, 'grounding': 17, 'curation': 17, 'results': 28, 'closing': 5}  # silent version
PAD = {'title': 1.0, 'task': 1.5, 'motivation': 2.0, 'method': 1.5, 'grounding': 1.5, 'curation': 1.5, 'results': 4.0, 'closing': 1.5}

def synth(no_tts):
    dur = {}
    for name, txt in SCENES:
        mp3 = OUT/f'nar_{name}.mp3'
        if not no_tts and not mp3.exists():
            subprocess.run([TTS, '--voice', VOICE, '--rate=-4%', '--text', txt, '--write-media', str(mp3)], check=True, capture_output=True)
        if mp3.exists():
            dur[name] = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(mp3)], capture_output=True, text=True).stdout.strip())
        else: dur[name] = max(4.0, len(txt.split()) / 2.6)
    return dur

# ───────────────────────── assets ─────────────────────────
qual = json.load(open(ST/'qual/index.json'))
def qual_item(direction, tag_prefix):
    return next(q for q in qual if q['direction'] == direction and q['tag'].startswith(tag_prefix))
A = {}
def load_assets():
    A['fig1'] = img_bgr(ST/'images/fig1_paradigms.png'); A['fig2'] = img_bgr(ST/'images/fig2_architecture.png')
    A['fig3'] = img_bgr(ST/'images/fig3_cycle.png'); A['fig4'] = img_bgr(ST/'images/fig4_pseudolabel.png')
    A['mq'] = img_bgr(ST/'motivation/query.jpg')
    for n in ['gt', 'ours', 'v2sam_sam2qt', 'objectrelator_ffmsam2']:
        p = FULL/'motivation'/f'{n}.mp4'; A['m_' + n] = read_video(p if p.exists() else ST/'motivation'/f'{n}.mp4')
    for n in ['vggt_basketball_packed', 'dino_soccer_packed']: A[n] = read_video(ST/'dataset'/f'{n}.mp4')
    IL = Path('/data3/_Nips26/project_page_vis/explainer/illus'); meta = json.load(open(IL/'meta.json'))
    A['illus'] = {n: img_bgr(IL/f'{n}.jpg') for n in ['hn_query_pos', 'hn_query_neg', 'hn_target_pos', 'hn_target_neg', 'lc_full', 'lc_crop']}
    A['illus']['focal'] = img_bgr(IL/'focal.png'); A['illus']['lc_box'] = meta['lc_box']
    A['task'] = qual_item('exo2ego', 'short_fac6316e'); A['task_v'] = read_video(PP/A['task']['video']); A['task_q'] = img_bgr(PP/A['task']['query'])
    A['res'] = [qual_item('exo2ego', '2ca53d8d'), qual_item('ego2exo', 'short_ede7b9c1'), qual_item('exo2ego', '9e08e1e7'), qual_item('ego2exo', 'short_6fbe4821')]
    for r in A['res']: r['_v'] = read_video(PP/r['video']); r['_q'] = img_bgr(PP/r['query'])

# ───────────────────────── scenes ─────────────────────────
def s_title(t, T):
    img = canvas(); a = ease(t, 0.2, 0.8)
    text(img, (W // 2, 330), 'NeurIPS 2026', 40, 'b', BLUE, anchor='ma')
    y = text(img, (W // 2, 410), 'Repurposing Video Diffusion Transformers for', 62, 'b', anchor='ma')
    text(img, (W // 2, y + 6), 'Cross-View Temporal Object Correspondence', 62, 'b', anchor='ma')
    text(img, (W // 2, 620), 'Siyoon Jin*¹   Dahyun Chung*¹   Honggyu An¹   Sangbeom Lim²', 34, 'r', MUTED, anchor='ma')
    text(img, (W // 2, 670), 'Wonjin Nam¹   Jaewoo Jung¹   WonJun Moon¹   Seungryong Kim¹', 34, 'r', MUTED, anchor='ma')
    text(img, (W // 2, 740), '¹KAIST AI     ²Korea University', 34, 'b', INK, anchor='ma')
    return fade(img, a)

def s_task(t, T):
    img = canvas(); header(img, 'Cross-view temporal object correspondence')
    r = paste(img, A['task_q'], (96, 300, 560, 320)); caption(img, (r[0] + r[2] // 2, r[1] + r[3] + 12), 'Query mask (exo view, first frame)')
    r2 = paste(img, vf(A['task_v'], t), (700, 300, 1124, 404)); caption(img, (r2[0] + r2[2] // 2, r2[1] + r2[3] + 12), 'Prediction in both views over time   (ego | exo)')
    bullets(img, 96, 780, ['Input: ego video + exo video + one first-frame query mask in either view', 'Output: temporally coherent masks of that object in both views',
                           'No per-frame query, no separate matching stage'], t, 1.0, size=32, maxw=1700)
    return fade(img, ease(t, 0, 0.4))

def s_motiv(t, T):
    img = canvas(); header(img, 'Why joint view-time prediction?')
    paste(img, A['fig1'], (260, 165, 1400, 280), radius=0)
    cells = [('Query (exo)', A['mq'], None), ('Ground truth', None, 'm_gt'), ('XTrack (ours)', None, 'm_ours'),
             ('(a) Track-then-Match', None, 'm_v2sam_sam2qt'), ('(b) Match-then-Track', None, 'm_objectrelator_ffmsam2')]
    x = 96; y = 490; cw = 284; gap = 22
    for k, (lab, im, key) in enumerate(cells):
        a = ease(t, 2.0 + k * 0.5, 0.4)
        if a <= 0: continue
        fr = im if im is not None else vf(A[key], t)
        box = (x, y, 480 if k == 0 else cw, cw)
        r = paste(img, fr, box)
        col = GREEN if k in (1, 2) else RED if k >= 3 else INK
        text(img, (r[0] + r[2] // 2, r[1] + r[3] + 12), lab, 26, 'b', col, anchor='ma')
        x += box[2] + gap
    text(img, (96, 890), 'Track-then-Match: temporally inconsistent matching, false positives when the target is absent', 28, 'r', RED)
    text(img, (96, 932), 'Match-then-Track: propagates an unreliable initial match, fails when the object reappears', 28, 'r', RED)
    text(img, (96, 974), 'XTrack: one model, both views, all frames', 28, 'b', GREEN)
    return fade(img, ease(t, 0, 0.4))

def s_method(t, T):
    img = canvas(); header(img, 'XTrack: RGB-to-mask flow in a video DiT')
    paste(img, A['fig2'], (96, 170, 1728, 560), radius=0)
    bullets(img, 96, 770, ['Ego and exo videos side by side in one input; both views predicted in one pass',
                           'Flow matching from RGB latents to mask latents, keeping the pretrained video priors',
                           'Within-view tracking and cross-view matching share the same attention'], t, 3.0, size=32, maxw=1700, stagger=3.5)
    return fade(img, ease(t, 0, 0.4))

def chip(img, xy, txt, col=(255, 255, 255), bg=(0, 0, 0), size=16):
    d = ImageDraw.Draw(img, 'RGBA'); f = font(size, 'b'); tw = d.textlength(txt, font=f); x, y = xy
    d.rounded_rectangle((x, y, x + tw + 14, y + size + 10), 6, fill=bg + (190,)); d.text((x + 7, y + 4), txt, font=f, fill=col)
def arrow(d, x0, y0, x1, y1, col=(130, 130, 130), w=3, head=11):
    d.line((x0, y0, x1, y1), fill=col, width=w); vx, vy = x1 - x0, y1 - y0; L = (vx * vx + vy * vy) ** 0.5; ux, uy = vx / L, vy / L
    d.polygon([(x1, y1), (x1 - head * ux + head * 0.55 * uy, y1 - head * uy - head * 0.55 * ux), (x1 - head * ux - head * 0.55 * uy, y1 - head * uy + head * 0.55 * ux)], fill=col)

def s_ground(t, T):
    img = canvas(); header(img, 'Enhancing cross-view object grounding'); d = ImageDraw.Draw(img)
    IL = A['illus']; PAD = 24; PW = 560; GAP = 24; PH = 545; y0 = 160; xs = [96 + k * (PW + GAP) for k in range(3)]
    titles = ['Focal attention alignment', 'Hard negative conditioning', 'Local context crop']
    descs = ['Cross-view attention is supervised to attend to the query object.',
             'Same-class instances as hard negatives: their predictions are pushed apart by a margin.',
             'Detail tokens from a crop around small objects, with RoPE positions of the full frame.']
    for k, x in enumerate(xs):
        d.rounded_rectangle((x, y0, x + PW, y0 + PH), 18, fill=(248, 249, 250), outline=(229, 229, 229), width=2)
        text(img, (x + PAD, y0 + 18), titles[k], 33, 'b'); cy = y0 + 76
        if k == 0:
            paste(img, IL['focal'], (x + PAD, cy, PW - 2 * PAD, 320), radius=0)
        elif k == 1:
            iw, ih = 222, 125; xq, xp = x + PAD, x + PW - PAD - iw; ys = [cy, cy + ih + 70]
            for ri, (qn, tn, lab, col) in enumerate([('hn_query_pos', 'hn_target_pos', 'query: front wheel', (40, 200, 40)), ('hn_query_neg', 'hn_target_neg', 'hard negative: back wheel', (255, 150, 40))]):
                yy = ys[ri]; r1 = paste(img, IL[qn], (xq, yy, iw, ih), fit='cover', radius=8); r2 = paste(img, IL[tn], (xp, yy, iw, ih), fit='cover', radius=8)
                d = ImageDraw.Draw(img); arrow(d, r1[0] + iw + 10, yy + ih // 2, r2[0] - 10, yy + ih // 2)
                chip(img, (r1[0] + 6, yy + 6), lab, col, size=18); chip(img, (r2[0] + 6, yy + 6), 'prediction', size=18)
            # the loss acts between the two predictions: vertical double arrow in the gap under the prediction column
            d = ImageDraw.Draw(img); gx = xp + 30; ga, gb = ys[0] + ih + 8, ys[1] - 8
            arrow(d, gx, (ga + gb) // 2, gx, ga, RED, 3, 9); arrow(d, gx, (ga + gb) // 2, gx, gb, RED, 3, 9)
            text(img, (gx + 16, ga + 4), 'margin loss', 21, 'b', RED); text(img, (gx + 16, ga + 32), 'pushes the two apart', 19, 'r', RED)
        else:
            full = IL['lc_full']; crop = IL['lc_crop']; bx, by, bs = IL['lc_box']
            fs_ = 300; r = paste(img, full, (x + PAD, cy, fs_, fs_), radius=8); sc = r[2] / 448
            cs_ = 190; ex0, ey0 = x + PW - PAD - cs_, cy + 8
            img.paste(Image.fromarray(cv2.cvtColor(cv2.resize(crop, (cs_, cs_)), cv2.COLOR_BGR2RGB)), (ex0, ey0)); d = ImageDraw.Draw(img)
            tx0, ty0, ts = r[0] + bx * sc, r[1] + by * sc, bs * sc
            d.line((tx0 + ts, ty0, ex0, ey0), fill=YELLOW_PIL, width=2); d.line((tx0 + ts, ty0 + ts, ex0, ey0 + cs_), fill=YELLOW_PIL, width=2)
            text(img, (ex0 + cs_ // 2, ey0 + cs_ + 8), 'local crop tokens', 20, 'b', (160, 120, 0), anchor='ma')
        text(img, (x + PAD, y0 + PH - 112), descs[k], 24, 'r', (68, 68, 68), maxw=PW - 2 * PAD, line=1.3)
    # ---- training: 2D pretraining with cycle consistency (schematic with real frames) + video fine-tuning
    ty = y0 + PH + 26; d = ImageDraw.Draw(img); d.rounded_rectangle((96, ty, 1824, ty + 300), 18, fill=(240, 251, 242), outline=(200, 230, 205), width=2)
    text(img, (96 + PAD, ty + 16), 'Training', 33, 'b', GREEN)
    d.ellipse((120, ty + 76, 134, ty + 90), fill=GREEN); text(img, (146, ty + 66), '2D pretraining with cycle consistency', 26, 'b', INK)
    text(img, (146, ty + 104), 'Predict the mask in the other view, feed it back as the query, and recover the original query mask.', 22, 'r', (68, 68, 68), maxw=560, line=1.3)
    d.ellipse((120, ty + 212, 134, ty + 226), fill=GREEN); text(img, (146, ty + 202), 'Video-level fine-tuning', 26, 'b', INK)
    text(img, (146, ty + 240), 'On curated dense pseudo-labels.', 22, 'r', (68, 68, 68), maxw=560)
    iw, ih = 230, 130; gy = ty + 50; x1, x2, x3 = 760, 1165, 1570
    r1 = paste(img, IL['hn_query_pos'], (x1, gy, iw, ih), fit='cover', radius=8); r2 = paste(img, IL['hn_target_pos'], (x2, gy, iw, ih), fit='cover', radius=8); r3 = paste(img, IL['hn_query_pos'], (x3, gy, iw, ih), fit='cover', radius=8)
    d = ImageDraw.Draw(img)
    chip(img, (r1[0] + 6, gy + 6), 'query mask (exo)', (40, 200, 40), size=18); chip(img, (r2[0] + 6, gy + 6), 'predicted mask (ego)', size=18); chip(img, (r3[0] + 6, gy + 6), 'reconstructed (exo)', size=18)
    arrow(d, r1[0] + iw + 10, gy + ih // 2, r2[0] - 10, gy + ih // 2); arrow(d, r2[0] + iw + 10, gy + ih // 2, r3[0] - 10, gy + ih // 2)
    text(img, ((r1[0] + iw + r2[0]) // 2, gy + ih // 2 - 28), 'predict', 19, 'b', MUTED, anchor='ma')
    text(img, ((r2[0] + iw + r3[0]) // 2, gy + ih // 2 - 28), 'feed back as query', 17, 'b', MUTED, anchor='ma')
    # return arrow underneath: reconstruction compared with the original query
    by_ = gy + ih + 34; d.line((r3[0] + iw // 2, gy + ih + 8, r3[0] + iw // 2, by_), fill=RED, width=3); d.line((r3[0] + iw // 2, by_, r1[0] + iw // 2, by_), fill=RED, width=3)
    arrow(d, r1[0] + iw // 2, by_, r1[0] + iw // 2, gy + ih + 10, RED, 3, 10)
    text(img, ((r1[0] + r3[0] + iw) // 2, by_ + 8), 'cycle-consistency loss: the reconstructed mask must match the original query', 21, 'b', RED, anchor='ma')
    return fade(img, ease(t, 0, 0.4))

def s_curation(t, T):
    img = canvas(); header(img, 'Pseudo-label curation: 1 FPS annotations to dense labels')
    paste(img, A['fig4'], (96, 165, 640, 520), radius=0)
    r = paste(img, vf(A['vggt_basketball_packed'], t), (780, 180, 1044, 240)); caption(img, (r[0] + r[2] // 2, r[1] + r[3] + 8), '(b) Cross-view geometric verification with VGGT: select the candidate consistent with the exo view')
    r = paste(img, vf(A['dino_soccer_packed'], t), (780, 470, 1044, 250)); caption(img, (r[0] + r[2] // 2, r[1] + r[3] + 8), '(c) Within-view semantic verification with DINOv3: discard chunks inconsistent with the ego anchor')
    bullets(img, 96, 760, ['(a) Forward and backward SAM2 propagation between anchors; disagreeing chunks are verified',
                           '(b) Geometric verification selects between the two candidates using the static exo view',
                           '(c) Semantic verification discards chunks that are still wrong'], t, 1.0, size=30, maxw=1700, stagger=4.0)
    return fade(img, ease(t, 0, 0.4))

def s_results(t, T):
    img = canvas(); header(img, 'Qualitative results')
    per = T / len(A['res']); k = min(int(t / per), len(A['res']) - 1); r = A['res'][k]; tl = t - k * per
    qv = 'Exo' if r['direction'] == 'exo2ego' else 'Ego'
    b = paste(img, r['_q'], (96, 250, 520, 420)); caption(img, (b[0] + b[2] // 2, b[1] + b[3] + 12), f'Query mask ({qv} view, first frame)')
    b2 = paste(img, vf(r['_v'], tl), (660, 250, 1164, 420)); caption(img, (b2[0] + b2[2] // 2, b2[1] + b2[3] + 12), 'XTrack prediction   (ego | exo)')
    text(img, (96, 760), f'{"Exo to Ego" if r["direction"] == "exo2ego" else "Ego to Exo"}   ·   sample {k + 1} / {len(A["res"])}', 28, 'b', MUTED)
    text(img, (96, 820), 'Masks stay on the queried object in both views through viewpoint change, occlusion and re-appearance.', 30, 'r', maxw=1700)
    return fade(img, min(ease(tl, 0, 0.35), 1.0))

def s_closing(t, T):
    img = canvas(); a = ease(t, 0.2, 0.8)
    text(img, (W // 2, 400), 'XTrack', 90, 'b', anchor='ma')
    text(img, (W // 2, 530), 'Project page, paper and more results', 36, 'r', MUTED, anchor='ma')
    text(img, (W // 2, 600), 'cvlab-kaist.github.io/XTrack', 48, 'b', BLUE, anchor='ma')
    text(img, (W // 2, 720), 'Thank you', 40, 'r', INK, anchor='ma')
    return fade(img, a)

RENDER = {'title': s_title, 'task': s_task, 'motivation': s_motiv, 'method': s_method, 'grounding': s_ground, 'curation': s_curation, 'results': s_results, 'closing': s_closing}

def main():
    no_tts = '--no-tts' in sys.argv
    load_assets()
    if no_tts: T = dict(FIXED)
    else:
        dur = synth(False); T = {n: dur[n] + PAD[n] for n, _ in SCENES}
    total = sum(T.values()); print('scene seconds:', {k: round(v, 1) for k, v in T.items()}, 'total', round(total, 1), flush=True)
    tmp = Path('/data3/_Nips26/project_page_vis/.tmp_frames/explainer'); tmp.mkdir(parents=True, exist_ok=True); [p.unlink() for p in tmp.glob('*.png')]
    n = 0
    for name, _ in SCENES:
        nf = int(round(T[name] * FPS))
        for i in range(nf):
            img = RENDER[name](i / FPS, T[name])
            # fade out in the last 0.4 s of every scene
            rem = (nf - i) / FPS
            if rem < 0.4: img = fade(img, rem / 0.4)
            img.save(tmp/f'{n:06d}.png', compress_level=1); n += 1
        print('rendered', name, nf, flush=True)
    if no_tts:
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS), '-i', f'{tmp}/%06d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20',
                        '-movflags', '+faststart', str(OUT/'xtrack_explainer.mp4')], check=True)
        shutil.rmtree(tmp); print('wrote (silent)', OUT/'xtrack_explainer.mp4', 'frames', n); return
    # audio: narration clips padded to scene lengths
    parts = []
    for name, _ in SCENES:
        mp3 = OUT/f'nar_{name}.mp3'; wav = OUT/f'nar_{name}.wav'
        if mp3.exists(): subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(mp3), '-af', f'apad=whole_dur={T[name]:.3f}', '-t', f'{T[name]:.3f}', '-ar', '44100', '-ac', '2', str(wav)], check=True)
        else: subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo', '-t', f'{T[name]:.3f}', str(wav)], check=True)
        parts.append(wav)
    (OUT/'concat.txt').write_text(''.join(f"file '{p}'\n" for p in parts))
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', str(OUT/'concat.txt'), '-c:a', 'aac', '-b:a', '160k', str(OUT/'narration.m4a')], check=True)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS), '-i', f'{tmp}/%06d.png', '-i', str(OUT/'narration.m4a'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20',
                    '-c:a', 'copy', '-shortest', '-movflags', '+faststart', str(OUT/'xtrack_explainer.mp4')], check=True)
    shutil.rmtree(tmp); print('wrote', OUT/'xtrack_explainer.mp4', 'frames', n)

if __name__ == '__main__':
    main()
