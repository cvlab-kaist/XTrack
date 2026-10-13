"""Generate index.html for the XTrack project page from the built assets (run build_assets.py first)."""
import json
from pathlib import Path

PP = Path('/home/ubuntu/dahyun/_Nips26/project_page')
qual = json.load(open(PP/'static/qual/index.json')) if (PP/'static/qual/index.json').exists() else []

TITLE = 'Repurposing Video Diffusion Transformers for Cross-View Temporal Object Correspondence'
AUTHORS = [('Siyoon Jin', '*'), ('Dahyun Chung', '*'), ('Honggyu An', ''), ('Sangbeom Lim', ''),
           ('Wonjin Nam', ''), ('Jaewoo Jung', ''), ('WonJun Moon', ''), ('Seungryong Kim', '')]
ABSTRACT = ('Cross-view object correspondence identifies the same object across views. However, existing methods treat this as a '
            'frame-level matching problem, necessitating a query mask at every frame. Furthermore, without modeling temporal context, '
            'they suffer from severe ambiguity under appearance changes or occlusions. Therefore, we introduce <b>cross-view temporal '
            'object correspondence</b>, which requires preserving object identity across both views and time from a query mask. To address '
            'this, we propose <b>XTrack</b>, a streamlined framework which repurposes a pretrained video DiT with semantic and spatiotemporal '
            'priors for RGB-to-mask latent transport, jointly modeling cross-view matching and temporal tracking. However, naive video DiTs '
            'do not guarantee accurate cross-view grounding and often attend to semantically similar yet incorrect objects across views. '
            'Thus, we introduce focal cross-view attention alignment, hard negative conditioning, and local context crop strategies to ensure '
            'reliable grounding. Finally, beyond these architectural enhancements, we also tackle the lack of dense annotation by introducing '
            'a pseudo-label curation pipeline that converts sparse Ego-Exo4D annotations into dense supervision to enable effective '
            'video-level training. Our highly competitive performances validate the effectiveness of XTrack as a streamlined framework, '
            'ensuring robust temporal consistency and identity preservation.')

METHODS = [('objectrelator', 'ObjectRelator'), ('omama', 'O-MaMa'), ('lmeec', 'LM-EEC'), ('v2sam', 'V2-SAM')]

def vid(src, cls='', group='motiv'):
    return f'<video class="sync {cls}" data-group="{group}" src="{src}" muted playsinline loop preload="metadata"></video>'

def motiv_row(label, cells, note='', note_cls='', row_cls=''):
    cells_html = ''.join(f'<div class="cell"><div class="cell-label">{c[0]}</div>{c[1]}</div>' for c in cells)
    note_html = f'<div class="row-note {note_cls}">{note}</div>' if note else ''
    return f'''<div class="motiv-row {row_cls}">
  <div class="row-label"><span>{label}</span></div>
  <div class="row-cells">{cells_html}</div>
  {note_html}
</div>'''

motiv_html = ''.join([
    motiv_row('Query<br><small>(Exo view, first frame)</small>',
              [('Query mask', '<img src="static/motivation/query.jpg" class="query-img">')], row_cls='row-query'),
    motiv_row('GT', [('Ground truth (Ego view)', vid('static/motivation/gt.mp4'))], row_cls='row-gt'),
    motiv_row('(a) Track-then-Match',
              [(name, vid(f'static/motivation/{m}_sam2qt.mp4')) for m, name in METHODS],
              '&#10007; Temporally inconsistent matching &nbsp;&nbsp;&nbsp; &#10007; False positives in the absence of the target', 'bad'),
    motiv_row('(b) Match-then-Track',
              [(name, vid(f'static/motivation/{m}_ffmsam2.mp4')) for m, name in METHODS],
              '&#10007; Propagates errors from an unreliable initial match &nbsp;&nbsp;&nbsp; &#10007; Fails when the object reappears', 'bad'),
    motiv_row('(c) XTrack (Ours)', [('Ours', vid('static/motivation/ours.mp4'))],
              '&#10003; Temporally consistent prediction, robust to disappearance and reappearance', 'good', 'row-ours'),
])

def qual_cards(direction):
    cards = []
    for s in [q for q in qual if q['direction'] == direction]:
        qv = 'Ego' if direction == 'ego2exo' else 'Exo'
        cards.append(f'''<div class="qual-card">
  <div class="qual-query"><img src="{s['query']}"><div class="cell-label">Query mask ({qv} view, frame 0)</div></div>
  <div class="qual-video">{vid(s['video'], 'qual-vid', 'qual-' + direction)}<div class="cell-label">XTrack prediction &mdash; Ego view (left) | Exo view (right)</div></div>
  <div class="qual-meta">{s['object']}</div>
</div>''')
    return ''.join(cards)

html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>XTrack: Cross-View Temporal Object Correspondence</title>
<meta name="description" content="{TITLE}">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/bulma/0.9.4/css/bulma.min.css">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
<link href="https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Noto+Sans:wght@400;500;700&display=swap" rel="stylesheet">
<style>
  :root {{ --green:#1fb84a; --red:#e03131; --ink:#222; --muted:#6b6b6b; --line:#e5e5e5; }}
  body {{ font-family:'Google Sans','Noto Sans',sans-serif; color:var(--ink); }}
  .hero .title {{ font-size:2.1rem; line-height:1.25; }}
  .authors {{ font-size:1.15rem; }}
  .authors a {{ color:#3273dc; }}
  .affil {{ font-size:1.05rem; color:var(--muted); margin-top:.3rem; }}
  .eq {{ font-size:.85rem; color:var(--muted); }}
  .link-block .button {{ margin:.25rem; }}
  .section-title {{ font-weight:700; }}
  .abstract p {{ text-align:justify; font-size:1.05rem; }}
  .figure-img {{ width:100%; height:auto; display:block; }}
  .caption {{ font-size:.95rem; color:var(--muted); margin-top:.5rem; }}
  /* motivation grid */
  .motiv {{ border:1px solid var(--line); border-radius:12px; padding:1rem; background:#fafafa; }}
  .motiv-row {{ display:flex; align-items:stretch; gap:1rem; padding:.6rem 0; border-top:1px dashed var(--line); flex-wrap:wrap; }}
  .motiv-row:first-child {{ border-top:none; }}
  .row-label {{ flex:0 0 150px; display:flex; align-items:center; font-weight:600; font-size:.95rem; }}
  .row-label small {{ font-weight:400; color:var(--muted); }}
  .row-cells {{ flex:1 1 auto; display:flex; gap:.6rem; flex-wrap:wrap; }}
  .cell {{ flex:0 1 200px; max-width:220px; }}
  .cell video, .cell img {{ width:100%; aspect-ratio:1/1; object-fit:cover; border-radius:8px; background:#000; display:block; }}
  .row-query .cell img {{ aspect-ratio:796/448; }}
  .row-query .cell {{ flex-basis:356px; max-width:380px; }}
  .cell-label {{ font-size:.8rem; color:var(--muted); text-align:center; margin-top:.25rem; }}
  .row-note {{ flex:0 0 100%; padding-left:166px; font-size:.95rem; font-weight:600; }}
  .row-note.bad {{ color:var(--red); }}
  .row-note.good {{ color:var(--green); }}
  .row-ours {{ background:#f0fbf2; border-radius:8px; }}
  .legend {{ font-size:.9rem; color:var(--muted); margin-top:.6rem; }}
  .legend .sw {{ display:inline-block; width:12px; height:12px; border-radius:3px; vertical-align:middle; margin:0 4px 0 10px; }}
  /* qualitative */
  .qual-grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(520px, 1fr)); gap:1.2rem; }}
  .qual-card {{ border:1px solid var(--line); border-radius:12px; padding:.8rem; background:#fff; display:grid; grid-template-columns:150px 1fr; gap:.8rem; align-items:start; }}
  .qual-query img {{ width:100%; border-radius:8px; display:block; background:#000; }}
  .qual-video video {{ width:100%; border-radius:8px; display:block; background:#000; }}
  .qual-meta {{ grid-column:1 / -1; font-size:.9rem; color:var(--muted); }}
  .tabs.is-toggle a {{ font-weight:600; }}
  .speed-note {{ font-size:.85rem; color:var(--muted); }}
  .ctrl {{ margin:.5rem 0; }}
  .ctrl button {{ margin-right:.4rem; }}
  pre.bib {{ background:#f5f5f5; border-radius:8px; padding:1rem; font-size:.9rem; white-space:pre-wrap; }}
  footer {{ color:var(--muted); font-size:.9rem; }}
  @media (max-width: 800px) {{ .row-label {{ flex-basis:100%; }} .row-note {{ padding-left:0; }} .qual-card {{ grid-template-columns:1fr; }} .qual-grid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>

<section class="hero">
  <div class="hero-body">
    <div class="container is-max-desktop has-text-centered">
      <h1 class="title publication-title">{TITLE}</h1>
      <div class="authors">
        {' &nbsp; '.join(f'<span class="author-block"><a href="#">{n}</a><sup>{s}</sup></span>' for n, s in AUTHORS)}
      </div>
      <div class="affil">KAIST AI</div>
      <div class="eq"><sup>*</sup>Equal contribution</div>
      <div class="link-block" style="margin-top:1rem">
        <a class="button is-dark is-rounded" href="static/paper.pdf"><span class="icon"><i class="fas fa-file-pdf"></i></span><span>Paper</span></a>
        <a class="button is-dark is-rounded" href="#"><span class="icon"><i class="ai ai-arxiv"></i>&nbsp;</span><span>arXiv</span></a>
        <a class="button is-dark is-rounded" href="#"><span class="icon"><i class="fab fa-github"></i></span><span>Code</span></a>
        <a class="button is-dark is-rounded" href="https://huggingface.co/datasets/clwm515/XTrack"><span class="icon"><i class="fas fa-database"></i></span><span>Results</span></a>
      </div>
    </div>
  </div>
</section>

<section class="section pt-0">
  <div class="container is-max-desktop">
    <h2 class="title is-4 section-title has-text-centered">Abstract</h2>
    <div class="abstract"><p>{ABSTRACT}</p></div>
  </div>
</section>

<section class="section pt-0">
  <div class="container is-max-widescreen">
    <h2 class="title is-4 section-title has-text-centered">Motivation: why joint view-time mask prediction?</h2>
    <p class="has-text-centered" style="max-width:900px;margin:0 auto 1rem">Given a first-frame query mask in one view, the task is to recover temporally coherent masks of the same object in <b>both</b> views.
    Existing frame-level matchers must be extended with a tracker, which leads to two decoupled pipelines; both fail under temporary disappearance and viewpoint change.
    XTrack instead predicts masks jointly across views and time within a single video DiT.</p>
    <img class="figure-img" src="static/images/fig1_paradigms.png" alt="paradigms">
    <div class="caption has-text-centered">(a) Track-then-Match propagates the query mask in the query view and matches it per frame. (b) Match-then-Track estimates one initial match and tracks each view independently. (c) XTrack jointly models cross-view correspondence and temporal dynamics.</div>

    <div class="ctrl has-text-centered" style="margin-top:1.2rem">
      <button class="button is-small" onclick="syncRestart('motiv')"><i class="fas fa-rotate-left"></i>&nbsp;Restart</button>
      <button class="button is-small" onclick="syncToggle('motiv')" id="btn-motiv"><i class="fas fa-pause"></i>&nbsp;Pause</button>
      <span class="speed-note">All videos are synchronized and played at 2&times; speed. Exo2Ego setting: query in the exo view, prediction in the ego view (Ego-Exo4D, "cutting board").</span>
    </div>
    <div class="motiv">
      {motiv_html}
    </div>
    <div class="legend has-text-centered">
      <span class="sw" style="background:var(--green)"></span> GT / query mask and XTrack prediction &nbsp;
      <span class="sw" style="background:var(--red)"></span> baseline prediction
    </div>
  </div>
</section>

<section class="section pt-0">
  <div class="container is-max-desktop">
    <h2 class="title is-4 section-title has-text-centered">Method</h2>
    <img class="figure-img" src="static/images/fig2_architecture.png" alt="architecture">
    <div class="caption has-text-centered">Overall architecture of XTrack. The ego and exo videos are fed jointly as a diptych to a pretrained video DiT, which transports RGB latents to mask latents under flow matching. Focal cross-view attention alignment, hard negative conditioning, and local context crops provide query-specific grounding.</div>
  </div>
</section>

<section class="section pt-0">
  <div class="container is-max-widescreen">
    <h2 class="title is-4 section-title has-text-centered">Qualitative results of XTrack</h2>
    <p class="has-text-centered" style="max-width:900px;margin:0 auto 1rem">Given a single query mask (left), XTrack predicts the object masks in <b>both</b> the ego and exo videos (right). Videos are played at 2&times; speed.</p>
    <div class="tabs is-toggle is-centered" id="qual-tabs">
      <ul>
        <li class="is-active" data-tab="ego2exo"><a>Ego &rarr; Exo &nbsp;<small>(query in ego view)</small></a></li>
        <li data-tab="exo2ego"><a>Exo &rarr; Ego &nbsp;<small>(query in exo view)</small></a></li>
      </ul>
    </div>
    <div class="qual-grid" id="qual-ego2exo">{qual_cards('ego2exo')}</div>
    <div class="qual-grid" id="qual-exo2ego" style="display:none">{qual_cards('exo2ego')}</div>
  </div>
</section>

<section class="section pt-0">
  <div class="container is-max-desktop">
    <h2 class="title is-4 section-title">BibTeX</h2>
<pre class="bib">@article{{xtrack2026,
  title   = {{Repurposing Video Diffusion Transformers for Cross-View Temporal Object Correspondence}},
  author  = {{Jin, Siyoon and Chung, Dahyun and An, Honggyu and Lim, Sangbeom and Nam, Wonjin and Jung, Jaewoo and Moon, WonJun and Kim, Seungryong}},
  journal = {{arXiv preprint}},
  year    = {{2026}}
}}</pre>
  </div>
</section>

<footer class="footer" style="padding:1.5rem">
  <div class="container is-max-desktop has-text-centered">
    Website template adapted from <a href="https://nerfies.github.io">Nerfies</a>. Videos from Ego-Exo4D.
  </div>
</footer>

<script>
// ---- synchronized playback per group ----
function groupVideos(g) {{ return Array.from(document.querySelectorAll('video.sync[data-group="' + g + '"]')); }}
function syncRestart(g) {{ groupVideos(g).forEach(v => {{ v.currentTime = 0; v.play().catch(()=>{{}}); }}); }}
function syncToggle(g) {{
  const vs = groupVideos(g); const paused = vs.length && vs[0].paused;
  vs.forEach(v => paused ? v.play().catch(()=>{{}}) : v.pause());
  const b = document.getElementById('btn-' + g); if (b) b.innerHTML = paused ? '<i class="fas fa-pause"></i>&nbsp;Pause' : '<i class="fas fa-play"></i>&nbsp;Play';
}}
// keep each group aligned: the first video of a group is the master; others follow when drift > 0.15 s
function startGroup(g) {{
  const vs = groupVideos(g); if (!vs.length) return;
  const master = vs[0];
  master.addEventListener('timeupdate', () => {{
    vs.slice(1).forEach(v => {{ if (Math.abs(v.currentTime - master.currentTime) > 0.15) v.currentTime = master.currentTime; }});
  }});
  master.addEventListener('seeked', () => vs.slice(1).forEach(v => v.currentTime = master.currentTime));
  Promise.all(vs.map(v => new Promise(r => {{ if (v.readyState >= 1) r(); else v.addEventListener('loadedmetadata', r, {{once:true}}); }})))
    .then(() => syncRestart(g));
}}
// play only when visible (saves bandwidth), start groups lazily
const groups = new Set(Array.from(document.querySelectorAll('video.sync')).map(v => v.dataset.group));
groups.forEach(startGroup);
const io = new IntersectionObserver(entries => entries.forEach(e => {{
  const v = e.target; if (e.isIntersecting) v.play().catch(()=>{{}}); else v.pause();
}}), {{threshold: 0.1}});
document.querySelectorAll('video.qual-vid').forEach(v => io.observe(v));
// ---- tabs ----
document.querySelectorAll('#qual-tabs li').forEach(li => li.addEventListener('click', () => {{
  document.querySelectorAll('#qual-tabs li').forEach(x => x.classList.remove('is-active')); li.classList.add('is-active');
  ['ego2exo','exo2ego'].forEach(t => document.getElementById('qual-' + t).style.display = (t === li.dataset.tab ? 'grid' : 'none'));
  syncRestart('qual-' + li.dataset.tab);
}}));
</script>
</body>
</html>
'''
(PP/'index.html').write_text(html)
print('wrote index.html with', len(qual), 'qualitative samples')
