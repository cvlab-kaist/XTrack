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

# ---------- motivation grid: row 1 = query | GT | ours ; row 2/3 = four baselines per paradigm ----------
motiv_html = f'''
<div class="mrow">
  <div class="mrow-head"><span class="tag-ref">Reference</span></div>
  <div class="mrow-cells">
    {cell('Query mask &middot; Exo view, first frame', '<img src="static/motivation/query.jpg">', 'wide')}
    {cell('Ground truth &middot; Ego view', vid('static/motivation/gt.mp4'))}
    {cell('<b>XTrack (Ours)</b> &middot; Ego view', vid('static/motivation/ours.mp4'), 'ours')}
  </div>
</div>
<div class="mrow">
  <div class="mrow-head"><span class="tag-bad">(a) Track-then-Match</span>
    <span class="note bad">&#10007; temporally inconsistent matching &nbsp; &#10007; false positives in the absence of the target</span></div>
  <div class="mrow-cells">{''.join(cell(name, vid(f'static/motivation/{m}_sam2qt.mp4')) for m, name in METHODS)}</div>
</div>
<div class="mrow">
  <div class="mrow-head"><span class="tag-bad">(b) Match-then-Track</span>
    <span class="note bad">&#10007; propagates errors from an unreliable initial match &nbsp; &#10007; fails when the object reappears</span></div>
  <div class="mrow-cells">{''.join(cell(name, vid(f'static/motivation/{m}_ffmsam2.mp4')) for m, name in METHODS)}</div>
</div>
'''

def qual_cards(direction):
    qv = 'Ego' if direction == 'ego2exo' else 'Exo'
    cards = []
    for s in [q for q in qual if q['direction'] == direction]:
        cards.append(f'''<div class="qual-card">
  <div class="qual-query"><img src="{s['query']}"><div class="cell-label">Query mask ({qv} view, first frame)</div></div>
  <div class="qual-video">{vid(s['video'], 'qual-vid', 'qual-' + direction)}<div class="cell-label">XTrack prediction &mdash; Ego view (left) &nbsp;|&nbsp; Exo view (right)</div></div>
</div>''')
    return ''.join(cards)

METHOD_SECTIONS = f'''
<div class="method-block">
  <img class="figure-img" src="static/images/fig2_architecture.png" alt="architecture"><div class="caption" style="margin-bottom:1rem">Overall architecture of XTrack.</div>
  <h3 class="title is-5">1. Cross-view RGB-to-mask flow via video DiT repurposing</h3>
  <div class="columns is-vcentered">
    <div class="column">
      <p>Pretrained video DiTs implicitly learn to preserve object identity under motion, occlusion and viewpoint change. We repurpose these priors
      for mask prediction by <b>redefining the flow-matching transport endpoints from noise&rarr;RGB to RGB&rarr;mask</b>: the source state is the RGB latent
      of the paired ego-exo videos (which already encodes scene layout, appearance and temporal context), and the target is the mask latent.
      The model is trained to predict <code>v* = z<sub>mask</sub> &minus; z<sub>rgb</sub></code>, conditioned on the RGB videos and the one-sided query mask,
      so the generative process is redirected toward query-conditioned mask prediction without discarding the DiT's spatiotemporal and semantic priors.</p>
    </div>
  </div>
</div>

<div class="method-block">
  <h3 class="title is-5">2. Unified ego-exo token space (diptych)</h3>
  <p>Ego and exo frames are spatially concatenated into a single <b>diptych</b> frame; the query mask conditions the query side while the other side receives an
  empty mask. Full 3D attention over ego tokens, exo tokens and conditioning tokens yields a block structure: <i>within-view</i> attention (ego&rarr;ego, exo&rarr;exo)
  captures spatial and temporal interactions inside each video, while <i>cross-view</i> attention (ego&rarr;exo, exo&rarr;ego) establishes object correspondence.
  Because every layer can access cross-view evidence, ambiguity in one view is resolved with the other view during prediction instead of relying on a fixed initial match,
  so cross-view matching and within-view temporal propagation are modeled jointly in a single process.</p>
</div>

<div class="method-block">
  <h3 class="title is-5">3. Enhancing cross-view object grounding</h3>
  <div class="columns">
    <div class="column">
      <div class="box-lite"><b>Focal attention alignment.</b> A unified token space does not guarantee instance-level alignment: cross-view attention may diffuse over
      background or distractors and the target-side prediction drifts to a wrong instance. We directly supervise the cross-view attention blocks so that
      target-side foreground tokens attend to query-side foreground tokens, with a <b>focal weight</b> that is strong when foreground attention is weak and
      vanishes when the model is already well grounded. The same loss covers Ego2Exo and Exo2Ego by swapping query and target sides.</div>
    </div>
    <div class="column">
      <div class="box-lite"><b>Hard negative conditioning.</b> Video diffusion models tend to attend to semantically similar regions, producing query-agnostic
      predictions when similar objects coexist. Given the correct query mask and a <b>mismatched query mask from another object</b>, a margin loss forces the two
      target-side predictions to differ, so the model relies on the query identity rather than generic object saliency.</div>
    </div>
    <div class="column">
      <div class="box-lite"><b>Local context crop for small objects.</b> Small objects occupy only a few tokens in the full-scene latent. An object-centered
      crop is encoded into additional local tokens and concatenated with the global ego-exo tokens; its <b>RoPE positions are taken from the crop's original
      coordinates</b> in the full frame, so high-resolution object evidence stays aligned with the global token layout.</div>
    </div>
  </div>
</div>

<div class="method-block">
  <h3 class="title is-5">4. Stage-wise training: 2D pretraining with cycle consistency, then video fine-tuning</h3>
  <div class="columns is-vcentered">
    <div class="column is-8">
      <p>Learning correspondence and temporal consistency at once is unstable: weak early correspondence is carried through time and causes identity drift.
      We therefore first train on <b>paired ego-exo frames</b>. The model predicts the mask in the other view and then maps it back to the original view;
      a <b>cycle-consistency loss</b> between the reconstructed and the original query mask validates the correspondence and discourages shortcut learning.
      We then <b>fine-tune on video sequences</b> with dense pseudo-labels (below), so the model learns to maintain identity under motion, occlusion and
      reappearance while keeping cross-view grounding. At inference, both views are predicted jointly over time in one pass.</p>
    </div>
    <div class="column is-4"><img class="figure-img" src="static/images/fig3_cycle.png" alt="cycle consistency"><div class="caption">2D pretraining with cycle consistency.</div></div>
  </div>
</div>

<div class="method-block">
  <h3 class="title is-5">5. Pseudo-label curation pipeline</h3>
  <div class="columns is-vcentered">
    <div class="column is-6"><img class="figure-img" src="static/images/fig4_pseudolabel.png" alt="pseudo-label pipeline"></div>
    <div class="column is-6">
      <p>Ego-Exo4D provides 30 FPS video but masks only at 1 FPS, and SAM2 propagation alone is unreliable in the ego view. Our pipeline turns sparse annotations into
      dense, reliable supervision:</p>
      <ul class="method-list">
        <li><b>(a) Bidirectional propagation and unreliable-region detection.</b> Propagate forward and backward from two annotated anchors; chunks with low
        forward/backward mIoU are flagged for refinement.</li>
        <li><b>(b) Cross-view geometric verification.</b> Static exo cameras give stable pseudo ground truth. Geometry-aware VGGT features compare each ego candidate
        with the exo reference, and the candidate whose similarity passes a threshold is kept.</li>
        <li><b>(c) Within-view semantic verification.</b> Remaining candidates are checked against the ego anchor with DINOv3 similarity; inconsistent masks are discarded.</li>
      </ul>
    </div>
  </div>
</div>
'''

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
  .venue {{ display:inline-block; font-weight:700; letter-spacing:.04em; color:#3b5bdb; border:2px solid #3b5bdb; border-radius:999px; padding:.15rem .9rem; margin-bottom:.8rem; }}
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
  .motiv {{ border:1px solid var(--line); border-radius:12px; padding:.6rem 1rem .4rem; background:#fafafa; }}
  .mrow {{ padding:.5rem 0 .4rem; border-top:1px dashed var(--line); }}
  .mrow:first-child {{ border-top:none; }}
  .mrow-head {{ display:flex; align-items:center; gap:1rem; flex-wrap:wrap; margin-bottom:.4rem; }}
  .tag-ref, .tag-bad, .tag-good {{ font-weight:700; font-size:.9rem; padding:.1rem .6rem; border-radius:6px; }}
  .tag-ref {{ background:#e9ecef; }}
  .tag-bad {{ background:#ffe3e3; color:#c92a2a; }}
  .tag-good {{ background:#d3f9d8; color:#2b8a3e; }}
  .note {{ font-size:.9rem; font-weight:600; }}
  .note.bad {{ color:var(--red); }}
  .mrow-cells {{ display:grid; grid-template-columns:repeat(4, 1fr); gap:.6rem; }}
  .cell video, .cell img {{ width:100%; aspect-ratio:1/1; object-fit:cover; border-radius:8px; background:#000; display:block; }}
  .cell.wide {{ grid-column:span 2; }}
  .cell.wide img {{ aspect-ratio:796/448; object-fit:cover; }}
  .cell.ours video {{ outline:3px solid var(--green); outline-offset:-3px; }}
  .cell-label {{ font-size:.8rem; color:var(--muted); text-align:center; margin-top:.25rem; }}
  .legend {{ font-size:.9rem; color:var(--muted); margin-top:.6rem; }}
  .legend .sw {{ display:inline-block; width:12px; height:12px; border-radius:3px; vertical-align:middle; margin:0 4px 0 10px; }}
  /* ---- method ---- */
  .method-block {{ margin-bottom:1.8rem; }}
  .method-block p, .method-list {{ text-align:justify; font-size:1rem; }}
  .method-list li {{ margin-bottom:.4rem; }}
  .box-lite {{ background:#f8f9fa; border:1px solid var(--line); border-radius:10px; padding:.9rem; font-size:.95rem; height:100%; }}
  code {{ background:#f1f3f5; color:#222; }}
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
  @media (max-width: 900px) {{ .mrow-cells {{ grid-template-columns:repeat(2, 1fr); }} .qual-grid {{ grid-template-columns:1fr; }} .qual-card {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>

<section class="hero">
  <div class="hero-body">
    <div class="container is-max-desktop has-text-centered">
      <div class="venue">{VENUE}</div>
      <h1 class="title publication-title">{TITLE}</h1>
      <div class="authors">
        {''.join('<span class="arow">' + ''.join(f'<a href="#">{n}</a><sup>{s}</sup>' for n, s in row) + '</span>' for row in AUTHOR_ROWS)}
      </div>
      <div class="affil">KAIST AI</div>
      <div class="eq"><sup>*</sup>Equal contribution</div>
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
    Extending frame-level matchers with a tracker yields two decoupled pipelines, and both break under temporary disappearance and viewpoint change.
    XTrack predicts masks jointly across views and time within a single video DiT.</p>
    <img class="figure-img" src="static/images/fig1_paradigms.png" alt="paradigms">
    <div class="caption">(a) Track-then-Match propagates the query mask in the query view and matches it per frame. (b) Match-then-Track estimates one initial match and tracks each view independently. (c) XTrack jointly models cross-view correspondence and temporal dynamics.</div>

    <div class="ctrl has-text-centered" style="margin-top:1rem">
      <button class="button is-small" onclick="syncRestart('motiv')"><i class="fas fa-rotate-left"></i>&nbsp;Restart</button>
      <button class="button is-small" onclick="syncToggle('motiv')" id="btn-motiv"><i class="fas fa-pause"></i>&nbsp;Pause</button>
      <span class="speed-note">Synchronized, 2&times; speed. Exo2Ego: query in the exo view, prediction in the ego view (Ego-Exo4D). The object leaves the ego view and reappears near the end.</span>
    </div>
    <div class="motiv">{motiv_html}</div>
    <div class="legend has-text-centered">
      <span class="sw" style="background:var(--green)"></span> ground truth / query mask / XTrack &nbsp;
      <span class="sw" style="background:var(--red)"></span> baseline prediction
    </div>
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
    <p class="lead">From a single query mask (left), XTrack predicts the object masks in <b>both</b> the ego and exo videos (right) in one pass. Videos are played at 2&times; speed.</p>
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
