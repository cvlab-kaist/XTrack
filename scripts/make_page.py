"""Generate index.html for the XTrack project page from the built assets (run build_assets.py first)."""
import json
from pathlib import Path

PP = Path(__file__).resolve().parent.parent
qual = json.load(open(PP/'static/qual/index.json')) if (PP/'static/qual/index.json').exists() else []

TITLE = 'Repurposing Video Diffusion Transformers for Cross-View Temporal Object Correspondence'
VENUE = 'NeurIPS 2026'
AUTHOR_ROWS = [[('Siyoon Jin', '*'), ('Dahyun Chung', '*'), ('Honggyu An', ''), ('Sangbeom Lim', '')],
               [('Wonjin Nam', ''), ('Jaewoo Jung', ''), ('WonJun Moon', ''), ('Seungryong Kim', '')]]
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

def cell(label, inner, cls=''):
    return f'<div class="cell {cls}">{inner}<div class="cell-label">{label}</div></div>'

# ---------- motivation grid (6 columns; left 2 columns hold the row description) ----------
def mblock(title, lines, cls):
    return f'<div class="mblock {cls}"><div class="mtitle">{title}</div>' + ''.join(f'<div class="mline">{l}</div>' for l in lines) + '</div>'

motiv_html = f"""
<div class="mgrid">
  {cell('Query mask &middot; Exo view, first frame', '<img src="static/motivation/query.jpg">', 'wide')}
  {cell('Ground truth', vid('static/motivation/gt.mp4'))}
  {cell('<b>XTrack (Ours)</b>', vid('static/motivation/ours.mp4'), 'ours')}
  {mblock('Exo2Ego &middot; Ego-Exo4D dataset', ['Query: <b>exo</b> &rarr; prediction: <b>ego</b>', '<span class="sw" style="background:#1fb84a"></span> GT / query / ours &nbsp; <span class="sw" style="background:#e03131"></span> baselines', '2&times; speed'], 'info')}

  {mblock('(a) Track-then-Match', ['<span class="x">&#10007;</span> temporally inconsistent matching', '<span class="x">&#10007;</span> false positives when the target is absent'], 'bad')}
  {''.join(cell(name, vid(f'static/motivation/{m}_sam2qt.mp4')) for m, name in METHODS)}

  {mblock('(b) Match-then-Track', ['<span class="x">&#10007;</span> propagates an unreliable initial match', '<span class="x">&#10007;</span> fails when the object reappears'], 'bad')}
  {''.join(cell(name, vid(f'static/motivation/{m}_ffmsam2.mp4')) for m, name in METHODS)}
</div>
"""

def qual_cards(direction):
    qv = 'Ego' if direction == 'ego2exo' else 'Exo'
    cards = []
    for s in [q for q in qual if q['direction'] == direction]:
        cards.append(f'''<div class="qual-card">
  <div class="qual-query"><img src="{s['query']}"><div class="cell-label">Query mask<br>({qv} view, first frame)</div></div>
  <div class="qual-video">{vid(s['video'], 'qual-vid', 'qual-' + direction)}<div class="cell-label">XTrack prediction &mdash; Ego view (left) &nbsp;|&nbsp; Exo view (right)</div></div>
</div>''')
    return ''.join(cards)

def card(icon, title, bullets, extra=''):
    return f"""<div class="mcard"><div class="mcard-head"><span class="mcard-icon"><i class="fas {icon}"></i></span><span class="mcard-title">{title}</span></div>
<ul class="mcard-list">{''.join(f'<li>{b}</li>' for b in bullets)}</ul>{extra}</div>"""

METHOD_SECTIONS = f"""
<img class="figure-img" src="static/images/fig2_architecture.png" alt="architecture">
<div class="caption" style="margin-bottom:1.2rem">Overall architecture. The ego and exo videos are placed side by side in a single frame and fed to a pretrained video DiT, which transports RGB latents to mask latents for both views at once.</div>

<div class="mcards">
  {card('fa-arrow-right-arrow-left', 'RGB-to-mask flow', ['Flow matching from the RGB latent to the mask latent, keeping the video DiT priors.'])}
  {card('fa-table-columns', 'Unified ego-exo tokens', ['Ego and exo frames are placed side by side in one input, so cross-view matching and within-view tracking happen in the same attention.'])}
  {card('fa-bullseye', 'Focal attention alignment', ['Cross-view attention is supervised to land on the query object.'])}
  {card('fa-code-compare', 'Hard negative conditioning', ['Run the same videos with the correct query mask and with another object\'s mask; a margin loss keeps the two predictions apart.'])}
  {card('fa-magnifying-glass-plus', 'Local context crop', ['Object crop tokens with original RoPE positions for small objects.'])}
  {card('fa-layer-group', 'Two-stage training', ['Image-level cycle-consistency pretraining, then video fine-tuning on curated pseudo-labels.'])}
</div>

<h3 class="title is-5" style="margin-top:2rem">Pseudo-label curation: 1 FPS annotations &rarr; 30 FPS verified labels</h3>
<img class="figure-img" src="static/images/fig4_pseudolabel.png" alt="pseudo-label pipeline" style="max-width:900px;margin:0 auto .8rem">
<div class="columns" style="margin-top:.2rem">
  <div class="column"><div class="step"><span class="step-n">a</span><div><b>Bidirectional propagation</b><br>Forward and backward SAM2 between anchors; chunks where the two propagations disagree are selected for verification.</div></div></div>
  <div class="column"><div class="step"><span class="step-n">b</span><div><b>Cross-view geometric verification</b><br>VGGT similarity to the exo view selects one of the two candidates.</div></div></div>
  <div class="column"><div class="step"><span class="step-n">c</span><div><b>Within-view semantic verification</b><br>DINOv3 similarity to the first-frame ego anchor discards chunks that are still wrong.</div></div></div>
</div>

<h4 class="title is-6" style="margin-top:1.6rem;margin-bottom:.4rem">Cross-view geometric verification</h4>
<p class="dsnote">Exo query (left), the forward (red) and backward (blue) ego candidates, and the candidate selected by VGGT similarity to the exo view (right).</p>
<div class="dsrow"><video class="sync" data-group="ds1" src="static/dataset/vggt_basketball_packed.mp4" muted playsinline loop preload="metadata"></video></div>
<div class="dsrow"><video class="sync" data-group="ds2" src="static/dataset/vggt_bike_packed.mp4" muted playsinline loop preload="metadata"></video></div>

<h4 class="title is-6" style="margin-top:1.6rem;margin-bottom:.4rem">Within-view semantic verification</h4>
<p class="dsnote">Geometric verification only <i>selects</i> between the forward and backward candidates, so it cannot recover a chunk where both candidates are wrong.
Semantic verification complements it: the selected candidate is compared with the first-frame ego anchor, and chunks with low similarity are discarded.</p>
<div class="dsrow"><video class="sync" data-group="ds3" src="static/dataset/vggt_soccer_packed.mp4" muted playsinline loop preload="metadata"></video><div class="cell-label"><b>Cross-view geometric verification</b> &mdash; both candidates miss the ball, so selecting one of them is not enough.</div></div>
<div class="dsrow"><video class="sync" data-group="ds3" src="static/dataset/dino_soccer_packed.mp4" muted playsinline loop preload="metadata"></video><div class="cell-label"><b>Within-view semantic verification</b> &mdash; compared with the first-frame ego anchor (left, green), the candidates have low DINOv3 similarity and the chunk is discarded.</div></div>
"""

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
  .venue {{ font-weight:700; font-size:1.6rem; color:var(--ink); margin:.8rem 0 .2rem; }}
  .authors {{ font-size:1.15rem; line-height:1.9; }}
  .authors .arow {{ display:block; }}
  .authors a {{ color:#3273dc; margin:0 .55rem; }}
  .affil {{ font-size:1.05rem; color:var(--muted); margin-top:.3rem; }}
  .eq {{ font-size:.85rem; color:var(--muted); }}
  .link-block .button {{ margin:.25rem; }}
  .section-title {{ font-weight:700; }}
  .abstract p {{ text-align:justify; font-size:1.05rem; }}
  .figure-img {{ width:100%; height:auto; display:block; }}
  .caption {{ font-size:.92rem; color:var(--muted); margin-top:.5rem; text-align:center; }}
  .lead {{ max-width:900px; margin:0 auto 1rem; text-align:center; }}
  /* ---- motivation grid ---- */
  .motiv {{ border:1px solid var(--line); border-radius:12px; padding:.6rem; background:#fafafa; }}
  .fig1 {{ max-width:960px; margin:0 auto; }}
  .mgrid {{ display:grid; grid-template-columns:repeat(6, 1fr); gap:.45rem; align-items:start; }}
  .mblock {{ grid-column:span 2; border-radius:8px; padding:.55rem .7rem; font-size:.82rem; line-height:1.35; }}
  .mblock.info {{ background:#eef1f5; font-size:1.2rem; line-height:1.45; }}
  .mblock.info .mtitle {{ font-size:1.3rem; }}
  .mblock.info .sw {{ width:16px; height:16px; }}
  .mblock.bad {{ background:none; padding-left:.2rem; }}
  .mblock.bad .mtitle {{ color:var(--ink); }}
  .mblock.bad .mline {{ color:var(--ink); }}
  .x {{ color:var(--red); font-weight:700; }}
  .mtitle {{ font-weight:700; font-size:.95rem; margin-bottom:.25rem; }}
  .mline {{ margin-bottom:.15rem; }}
  .mblock .sw {{ display:inline-block; width:10px; height:10px; border-radius:2px; vertical-align:middle; }}
  .cell video, .cell img {{ width:100%; aspect-ratio:1/1; object-fit:cover; border-radius:8px; background:#000; display:block; }}
  .cell.wide {{ grid-column:span 2; }}
  .cell.wide img {{ aspect-ratio:2/1; object-fit:cover; }}
  .cell.ours video {{ outline:3px solid var(--green); outline-offset:-3px; }}
  .cell-label {{ font-size:.8rem; color:var(--muted); text-align:center; margin-top:.25rem; }}
  .legend {{ font-size:.9rem; color:var(--muted); margin-top:.6rem; }}
  .legend .sw {{ display:inline-block; width:12px; height:12px; border-radius:3px; vertical-align:middle; margin:0 4px 0 10px; }}
  /* ---- method ---- */
  .mcards {{ display:grid; grid-template-columns:repeat(3, 1fr); gap:.9rem; }}
  .mcard {{ background:#fff; border:1px solid var(--line); border-radius:12px; padding:.9rem 1rem; box-shadow:0 1px 3px rgba(0,0,0,.04); }}
  .mcard-head {{ display:flex; align-items:center; gap:.6rem; margin-bottom:.5rem; }}
  .mcard-icon {{ width:32px; height:32px; border-radius:8px; background:#edf2ff; color:#3b5bdb; display:inline-flex; align-items:center; justify-content:center; }}
  .mcard-title {{ font-weight:700; font-size:.98rem; white-space:nowrap; }}
  .mcard-list {{ list-style:none; padding-left:0; font-size:.88rem; color:#444; }}
  .mcard-list li {{ margin-bottom:.3rem; }}
  .step {{ display:flex; gap:.7rem; font-size:.92rem; }}
  .dsrow video {{ width:100%; max-width:1212px; display:block; margin:0 auto; border-radius:8px; background:#000; }}
  .dsrow {{ margin-bottom:.6rem; }}
  .dsnote {{ font-size:.92rem; color:#444; margin-bottom:.5rem; }}
  .step-n {{ flex:0 0 26px; height:26px; border-radius:50%; background:#d3f9d8; color:#2b8a3e; font-weight:700; display:inline-flex; align-items:center; justify-content:center; }}
  /* ---- qualitative ---- */
  .qual-grid {{ display:grid; grid-template-columns:repeat(2, 1fr); gap:1rem; }}
  .qual-card {{ border:1px solid var(--line); border-radius:12px; padding:.7rem; background:#fff; display:grid; grid-template-columns:140px 1fr; gap:.7rem; align-items:start; }}
  .qual-query img {{ width:100%; border-radius:8px; display:block; background:#000; }}
  .qual-video video {{ width:100%; border-radius:8px; display:block; background:#000; }}
  .tabs.is-toggle a {{ font-weight:600; }}
  .speed-note {{ font-size:.85rem; color:var(--muted); }}
  .ctrl {{ margin:.5rem 0; }}
  .ctrl button {{ margin-right:.4rem; }}
  pre.bib {{ background:#f5f5f5; border-radius:8px; padding:1rem; font-size:.9rem; white-space:pre-wrap; }}
  footer {{ color:var(--muted); font-size:.9rem; }}
  @media (max-width: 900px) {{ .mcards {{ grid-template-columns:1fr; }} .mgrid {{ grid-template-columns:repeat(3, 1fr); }} .mblock {{ grid-column:span 3; }} .qual-grid {{ grid-template-columns:1fr; }} .qual-card {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>

<section class="hero">
  <div class="hero-body">
    <div class="container is-max-desktop has-text-centered">
      <h1 class="title publication-title">{TITLE}</h1>
      <div class="authors">
        {''.join('<span class="arow">' + ''.join(f'<a href="#">{n}</a><sup>{s}</sup>' for n, s in row) + '</span>' for row in AUTHOR_ROWS)}
      </div>
      <div class="affil">KAIST AI</div>
      <div class="eq"><sup>*</sup>Equal contribution</div>
      <div class="venue">{VENUE}</div>
      <div class="link-block" style="margin-top:1rem">
        <a class="button is-dark is-rounded" href="static/paper.pdf"><span class="icon"><i class="fas fa-file-pdf"></i></span><span>Paper</span></a>
        <a class="button is-dark is-rounded" href="#"><span class="icon"><i class="fas fa-scroll"></i></span><span>arXiv</span></a>
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
    <p class="lead">Given a first-frame query mask in one view, the task is to recover temporally coherent masks of the same object in <b>both</b> views.
    Extending frame-level matchers with a tracker yields two decoupled pipelines: <b>(a) Track-then-Match</b> propagates the query mask in the query view and matches it per frame,
    and <b>(b) Match-then-Track</b> estimates one initial match and tracks each view independently. Both break under temporary disappearance and viewpoint change.
    <b>(c) XTrack</b> jointly models cross-view correspondence and temporal dynamics within a single video DiT.</p>
    <img class="figure-img fig1" src="static/images/fig1_paradigms.png" alt="paradigms">

    <div class="ctrl has-text-centered" style="margin-top:1rem">
      <button class="button is-small" onclick="syncRestart('motiv')"><i class="fas fa-rotate-left"></i>&nbsp;Restart</button>
      <button class="button is-small" onclick="syncToggle('motiv')" id="btn-motiv"><i class="fas fa-pause"></i>&nbsp;Pause</button>
      <span class="speed-note">Synchronized videos, 2&times; speed.</span>
    </div>
    <div class="motiv">{motiv_html}</div>
  </div>
</section>

<section class="section pt-0">
  <div class="container is-max-desktop">
    <h2 class="title is-4 section-title has-text-centered">Method</h2>
    {METHOD_SECTIONS}
  </div>
</section>

<section class="section pt-0">
  <div class="container is-max-widescreen">
    <h2 class="title is-4 section-title has-text-centered">Qualitative results</h2>
    <p class="lead">From a single query mask (left), XTrack predicts the object masks in <b>both</b> the ego and exo videos (right) in one pass.
    The predicted masks stay on the queried object in both views through viewpoint changes, occlusions and re-appearance, without any per-frame query. Videos are played at 2&times; speed.</p>
    <div class="tabs is-toggle is-centered" id="qual-tabs">
      <ul>
        <li class="is-active" data-tab="exo2ego"><a>Exo &rarr; Ego &nbsp;<small>(query in exo view)</small></a></li>
        <li data-tab="ego2exo"><a>Ego &rarr; Exo &nbsp;<small>(query in ego view)</small></a></li>
      </ul>
    </div>
    <div class="qual-grid" id="qual-exo2ego">{qual_cards('exo2ego')}</div>
    <div class="qual-grid" id="qual-ego2exo" style="display:none">{qual_cards('ego2exo')}</div>
  </div>
</section>

<section class="section pt-0">
  <div class="container is-max-desktop">
    <h2 class="title is-4 section-title">BibTeX</h2>
<pre class="bib">@inproceedings{{xtrack2026,
  title     = {{Repurposing Video Diffusion Transformers for Cross-View Temporal Object Correspondence}},
  author    = {{Jin, Siyoon and Chung, Dahyun and An, Honggyu and Lim, Sangbeom and Nam, Wonjin and Jung, Jaewoo and Moon, WonJun and Kim, Seungryong}},
  booktitle = {{Advances in Neural Information Processing Systems (NeurIPS)}},
  year      = {{2026}}
}}</pre>
  </div>
</section>

<footer class="footer" style="padding:1.5rem">
  <div class="container is-max-desktop has-text-centered">
    Website template adapted from <a href="https://nerfies.github.io">Nerfies</a>. Videos from Ego-Exo4D.
  </div>
</footer>

<script>
function groupVideos(g) {{ return Array.from(document.querySelectorAll('video.sync[data-group="' + g + '"]')); }}
function syncRestart(g) {{ groupVideos(g).forEach(v => {{ v.currentTime = 0; v.play().catch(()=>{{}}); }}); }}
function syncToggle(g) {{
  const vs = groupVideos(g); const paused = vs.length && vs[0].paused;
  vs.forEach(v => paused ? v.play().catch(()=>{{}}) : v.pause());
  const b = document.getElementById('btn-' + g); if (b) b.innerHTML = paused ? '<i class="fas fa-pause"></i>&nbsp;Pause' : '<i class="fas fa-play"></i>&nbsp;Play';
}}
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
const groups = new Set(Array.from(document.querySelectorAll('video.sync')).map(v => v.dataset.group));
groups.forEach(startGroup);
const io = new IntersectionObserver(entries => entries.forEach(e => {{
  const v = e.target; if (e.isIntersecting) v.play().catch(()=>{{}}); else v.pause();
}}), {{threshold: 0.1}});
document.querySelectorAll('video.qual-vid').forEach(v => io.observe(v));
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
