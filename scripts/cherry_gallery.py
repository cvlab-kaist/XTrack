"""Cherry-picking gallery: one low-res strip video per sample = [query first frame (GT) | ego prediction | exo prediction].
Writes project_page/_cherry/<set>/<dir>/<sample>.mp4 and project_page/_cherry/index.html (served by the preview server,
git-ignored). Usage: python scripts/cherry_gallery.py [set_folder_name]   (default: val600_cherry_short_obj_p1_top30)
"""
import json, subprocess, sys, shutil
from pathlib import Path
import cv2, numpy as np

HF = Path('/data3/_Nips26/xtrack_hf'); PP = Path('/home/ubuntu/dahyun/_Nips26/project_page'); OUT = PP/'_cherry'
SET = sys.argv[1] if len(sys.argv) > 1 else 'val600_cherry_short_obj_p1_top30'
H = 150; FPS = 15; TMP = Path('/data3/_Nips26/project_page_vis/.tmp_frames/cherry')

def read(p):
    cap = cv2.VideoCapture(str(p)); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    return fr
def rs(im): return cv2.resize(im, (int(im.shape[1] * H / im.shape[0]), H), interpolation=cv2.INTER_AREA)
def label(img, txt):
    cv2.putText(img, txt, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA); return img

items = {}
for dirname in ['ego_to_exo', 'exo_to_ego']:
    items[dirname] = []
    for sdir in sorted((HF/SET/dirname).glob('rank_*')):
        q = 'ego' if dirname == 'ego_to_exo' else 'exo'
        w = json.load(open(sdir/'window.json'))
        qf = label(rs(read(sdir/f'{q}_overlay_gt.mp4')[0]), f'query ({q})')
        ego = [label(rs(f), 'ego pred') for f in read(sdir/'ego_overlay_pred.mp4')]
        exo = [label(rs(f), 'exo pred') for f in read(sdir/'exo_overlay_pred.mp4')]
        n = min(len(ego), len(exo))
        TMP.mkdir(parents=True, exist_ok=True); [p.unlink() for p in TMP.glob('*.png')]
        for i in range(n):
            fr = np.concatenate([qf, np.full((H, 4, 3), 255, np.uint8), ego[i], np.full((H, 4, 3), 255, np.uint8), exo[i]], 1)
            if fr.shape[1] % 2: fr = fr[:, :-1]
            cv2.imwrite(str(TMP/f'{i:04d}.png'), fr)
        od = OUT/SET/dirname; od.mkdir(parents=True, exist_ok=True); mp4 = od/f'{sdir.name}.mp4'
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS), '-i', f'{TMP}/%04d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '27', '-movflags', '+faststart', str(mp4)], check=True)
        items[dirname].append({'name': sdir.name, 'rank': w.get('rank'), 'JF': round(w.get('JF', 0), 3), 'object': w.get('object'), 'uid': w.get('take_uid', '')[:8],
                               'chunks': f"{w.get('chunk_start')}-{w.get('chunk_end')}", 'video': f'{SET}/{dirname}/{sdir.name}.mp4'})
        print(dirname, sdir.name, n, flush=True)
shutil.rmtree(TMP, ignore_errors=True)

def section(dirname):
    cards = ''.join(f'''<div class="card" data-name="{it['name']}">
  <video src="{it['video']}" muted autoplay loop playsinline preload="metadata"></video>
  <div class="meta"><label><input type="checkbox" class="pick" data-dir="{dirname}" data-name="{it['name']}"> <b>#{it['rank']}</b> {it['object']} &middot; JF {it['JF']} &middot; {it['uid']} &middot; c{it['chunks']}</label></div>
</div>''' for it in items[dirname])
    return f'<h2>{dirname} ({len(items[dirname])})</h2><div class="grid">{cards}</div>'

html = f'''<!DOCTYPE html><html><head><meta charset="utf-8"><title>cherry: {SET}</title>
<style>
body{{font-family:sans-serif;margin:12px;background:#111;color:#eee}} h2{{margin:18px 0 8px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(560px,1fr));gap:8px}}
.card{{background:#222;border-radius:6px;padding:4px}} .card video{{width:100%;display:block;border-radius:4px}}
.meta{{font-size:12px;margin-top:3px}} .card.picked{{outline:2px solid #4c6}}
#bar{{position:sticky;top:0;background:#111;padding:6px 0;border-bottom:1px solid #333;z-index:5}}
textarea{{width:100%;height:70px;background:#000;color:#9f9;font-size:11px}} button{{margin-right:6px}}
</style></head><body>
<div id="bar"><b>{SET}</b> &nbsp; <button onclick="toggleAll(true)">play all</button><button onclick="toggleAll(false)">pause all</button>
<button onclick="restart()">restart</button> &nbsp; picked: <span id="n">0</span>
<textarea id="out" readonly placeholder="checked sample folder names appear here"></textarea></div>
{section('exo_to_ego')}{section('ego_to_exo')}
<script>
function toggleAll(p){{document.querySelectorAll('video').forEach(v=>p?v.play():v.pause())}}
function restart(){{document.querySelectorAll('video').forEach(v=>{{v.currentTime=0;v.play()}})}}
function upd(){{const ps=[...document.querySelectorAll('.pick:checked')];document.getElementById('n').textContent=ps.length;
 document.getElementById('out').value=ps.map(p=>p.dataset.dir+'/'+p.dataset.name).join('\\n');
 document.querySelectorAll('.card').forEach(c=>c.classList.toggle('picked',!!c.querySelector('.pick:checked')));
 try{{localStorage.setItem('cherry_{SET}',JSON.stringify(ps.map(p=>p.dataset.dir+'/'+p.dataset.name)))}}catch(e){{}}}}
document.querySelectorAll('.pick').forEach(p=>p.addEventListener('change',upd));
try{{const saved=JSON.parse(localStorage.getItem('cherry_{SET}')||'[]');document.querySelectorAll('.pick').forEach(p=>{{if(saved.includes(p.dataset.dir+'/'+p.dataset.name))p.checked=true}});upd()}}catch(e){{}}
</script></body></html>'''
(OUT/'index.html').write_text(html); json.dump(items, open(OUT/f'{SET}.json', 'w'), indent=1)
print('gallery:', OUT/'index.html')
