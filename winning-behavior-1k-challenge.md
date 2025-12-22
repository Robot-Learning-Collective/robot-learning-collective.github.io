---
layout: default
title: Winning the BEHAVIOR-1K Challenge
description: Applying modified Pi0.5 for multytask paroblem in massive simulation
header_links:
  - text: View on GitHub
    url: https://github.com/IliaLarchenko/behavior-1k-solution
  - text: ArXiv
    url: https://arxiv.org/abs/2512.06951
---

We took **1st place** in the 2025 **BEHAVIOR-1K Challenge**, a large-scale benchmark of 50 long-horizon household tasks in photo-realistic simulation. Each episode spans minutes of bimanual manipulation and navigation, with a single policy expected to generalize across diverse activities such as turning on a radio, cooking a hotdog, or tidying a room.

<style>
.top-buttons { display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 1.5rem 0; }
.top-buttons a { transition: transform 0.1s, opacity 0.1s; }
.top-buttons a:hover { transform: translateY(-2px); opacity: 0.9; }
.top-buttons img { height: 32px; }
.eval-block { background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 8px; padding: 1rem; margin: 1.5rem 0; }
.eval-block h3 { margin: 0 0 0.25rem 0; }
.eval-block .subtitle { margin: 0 0 1rem 0; color: #4b5563; font-size: 0.95rem; }
.eval-container { display: flex; gap: 1rem; align-items: flex-start; flex-wrap: wrap; margin: 0; }
.eval-tabs { display: flex; flex-direction: column; gap: 0.5rem; flex: 1 1 160px; min-width: 140px; max-width: 240px; }
.eval-tabs button { width: 100%; text-align: left; padding: 0.45rem 0.7rem; border: 1px solid #ccc; background: #f5f5f5; cursor: pointer; border-radius: 4px; font-size: clamp(0.7rem, 2vw, 0.85rem); }
.eval-tabs button.active { background: #0366d6; color: white; border-color: #0366d6; }
.eval-panels { flex: 1 1 400px; min-width: 300px; }
.eval-panel { display: none; }
.eval-panel.active { display: block; }
.eval-panel video { width: 100%; height: auto; max-width: 100%; display: block; border-radius: 4px; }
.eval-instruction { margin-top: 0.5rem; padding: 0.4rem 0.55rem; background: #fffaf0; border-left: 3px solid #d97706; border-radius: 4px; font-size: 0.9rem; color: #5a3b00; }
.eval-note { margin-top: 0.5rem; padding: 0.35rem 0.5rem; background: #f0f6ff; border-left: 3px solid #0366d6; border-radius: 4px; font-size: 0.85rem; color: #0b2d52; }
.lightbox-trigger { display: inline-block; }
.lightbox { position: fixed; inset: 0; background: rgba(0,0,0,0.78); display: none; align-items: center; justify-content: center; padding: 1.5rem; z-index: 9999; cursor: zoom-out; }
.lightbox.active { display: flex; }
.lightbox-content { position: relative; max-width: 98vw; max-height: 98vh; cursor: default; text-align: center; }
.lightbox-content img { max-width: 96vw; max-height: 96vh; display: block; border-radius: 6px; margin: 0 auto; }
.lightbox-caption { margin-top: 0.5rem; color: #f5f5f5; text-align: center; font-size: 0.95rem; }
.side-by-side { display: flex; gap: 1.5rem; align-items: flex-start; flex-wrap: wrap; }
.side-by-side .text-col { flex: 1 1 300px; min-width: 250px; }
.side-by-side .img-col { flex: 1 1 400px; min-width: 300px; }
.side-by-side .text-col ul { margin: 0; padding-left: 1.2rem; }
.side-by-side .text-col li { margin-bottom: 0.75rem; }
</style>


<nav style="background: #f6f8fa; padding: 0.75rem 1rem; border-radius: 6px; margin: 1.5rem 0; font-size: 14px;">
  <strong>Contents:</strong>
  <a href="#challenge" style="margin-left: 1rem;">Challenge</a> ·
  <a href="#task-examples">Task Examples</a> ·
  <a href="#approach">Approach</a> ·
  <a href="#results">Results</a> ·
  <a href="#recovery-from-cross-task-learning">Recovery</a>
</nav>

## Challenge

50 tasks etc.

<style>
.video-tabs { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1rem; }
.video-tabs button { padding: 0.5rem 1rem; border: 1px solid #ccc; background: #f5f5f5; cursor: pointer; border-radius: 4px; font-size: 0.85rem; }
.video-tabs button.active { background: #0366d6; color: white; border-color: #0366d6; }
.video-panel { display: none; }
.video-panel.active { display: block; }
.video-panel video { width: 100%; height: auto; max-width: 640px; display: block; margin: 0 auto; border-radius: 6px; }
</style>

<script>
// Reusable tabbed-video initializer (shared across intro, success, fail blocks)
function initTabbedVideos({
  rootSelector,
  root,
  tabsSelector,
  panelsSelector,
  jsonUrl,
  panelClass = 'eval-panel',
  renderButtonLabel,
  renderPanelHTML,
}) {
  const rootEl = root || document.querySelector(rootSelector);
  if (!rootEl) return;
  const tabs = rootEl.querySelector(tabsSelector);
  const panels = rootEl.querySelector(panelsSelector);
  if (!tabs || !panels) return;

  const panelSelector = `.${panelClass.split(' ').join('.')}`;
  let activeIndex = 0;
  let isRootVisible = false;
  let allItems = [];
  let pendingFilterFn = null;

  const pauseAll = () => {
    panels.querySelectorAll(panelSelector).forEach((panel) => {
      const vid = panel.querySelector('video');
      if (vid) vid.pause();
    });
  };

  const playActive = () => {
    const activePanel = panels.querySelector(`${panelSelector}.active`);
    const vid = activePanel ? activePanel.querySelector('video') : null;
    if (vid) vid.play().catch(() => {});
  };

  const observer =
    typeof IntersectionObserver !== 'undefined'
      ? new IntersectionObserver(
          (entries) => {
            entries.forEach((entry) => {
              if (entry.target !== rootEl) return;
              isRootVisible = entry.isIntersecting;
              if (isRootVisible) {
                playActive();
              } else {
                pauseAll();
              }
            });
          },
          { threshold: 0.2 }
        )
      : null;

  if (observer) observer.observe(rootEl);

  const activate = (index) => {
    activeIndex = index;
    tabs.querySelectorAll('button').forEach((btn, i) => {
      const isActive = i === index;
      btn.classList.toggle('active', isActive);
    });
    panels.querySelectorAll(panelSelector).forEach((panel, i) => {
      const isActive = i === index;
      panel.classList.toggle('active', isActive);
      panel.style.display = isActive ? 'block' : 'none';
      const vid = panel.querySelector('video');
      if (vid && !isActive) vid.pause();
    });
    if (isRootVisible) playActive();
  };

  const renderItems = (items) => {
    tabs.innerHTML = '';
    panels.innerHTML = '';
    items.forEach((item, idx) => {
      const btn = document.createElement('button');
      btn.textContent = renderButtonLabel(item, idx);
      btn.addEventListener('click', () => activate(idx));
      if (idx === 0) btn.classList.add('active');
      tabs.appendChild(btn);

      const panel = document.createElement('div');
      panel.className = panelClass;
      panel.id = `${rootEl.id || 'panel'}-${idx}`;
      panel.innerHTML = renderPanelHTML(item, idx);
      panel.style.display = idx === 0 ? 'block' : 'none';
      if (idx === 0) panel.classList.add('active');
      panels.appendChild(panel);
    });
    if (items.length) activate(0);
    else tabs.innerHTML = '<small>No matching episodes found.</small>';
  };

  fetch(jsonUrl)
    .then((resp) => {
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      return resp.json();
    })
    .then((items) => {
      allItems = items;
      if (pendingFilterFn) {
        renderItems(allItems.filter(pendingFilterFn));
      } else {
        renderItems(allItems);
      }
    })
    .catch((err) => {
      tabs.innerHTML = '<small>Failed to load clips.</small>';
      console.error('Failed to load tabbed videos', err);
    });

  // Expose filter method on the root element
  rootEl.filterItems = (filterFn) => {
    if (allItems.length === 0) {
      pendingFilterFn = filterFn;
    } else {
      const filtered = filterFn ? allItems.filter(filterFn) : allItems;
      renderItems(filtered);
    }
  };
}

// Predefined renderers so blocks can be configured with data attributes
const TAB_RENDERERS = {
  intro: {
    panelClass: 'eval-panel video-panel',
    renderButtonLabel: (item) => item.label,
    renderPanelHTML: (item) => `
      <video autoplay muted playsinline controls>
        <source src="${item.src}" type="video/mp4">
      </video>
      ${item.instruction ? `<div class="eval-note"><strong>Instruction:</strong> ${item.instruction}</div>` : ''}
    `,
  },
  success: {
    panelClass: 'eval-panel',
    renderButtonLabel: (clip) => clip.label,
    renderPanelHTML: (clip) => `
      <video autoplay muted playsinline controls>
        <source src="${clip.src}" type="video/mp4">
      </video>
      ${clip.instruction ? `<div class="eval-instruction"><strong>Instruction:</strong> ${clip.instruction}</div>` : ''}
      <div class="eval-note"><strong>Note:</strong> ${clip.note}</div>
    `,
  },
  fail: {
    panelClass: 'eval-panel fail-panel',
    renderButtonLabel: (ex) => `${ex.task} #${ex.episode}`,
    renderPanelHTML: (ex) => `
      <video autoplay muted playsinline controls>
        <source src="${ex.video_path}" type="video/mp4">
      </video>
      ${ex.reason ? `<div class="eval-instruction"><strong>Reason:</strong> ${ex.reason}</div>` : ''}
      <div class="eval-note"><strong>Note:</strong> ${ex.note}</div>
    `,
  },
};

function initAllTabbedVideoBlocks() {
  document.querySelectorAll('[data-tabbed-json]').forEach((block) => {
    const type = block.dataset.tabbedType || 'intro';
    const renderer = TAB_RENDERERS[type];
    if (!renderer) return;
    initTabbedVideos({
      root: block,
      tabsSelector: block.dataset.tabsSelector || '.eval-tabs',
      panelsSelector: block.dataset.panelsSelector || '.eval-panels',
      jsonUrl: block.dataset.tabbedJson,
      panelClass: block.dataset.panelClass || renderer.panelClass,
      renderButtonLabel: renderer.renderButtonLabel,
      renderPanelHTML: renderer.renderPanelHTML,
    });
  });
}

document.addEventListener('DOMContentLoaded', initAllTabbedVideoBlocks);
</script>

<div class="eval-block" id="intro-evals" data-tabbed-json="assets/winning-behavior-1k-challenge/task_examples.json" data-tabbed-type="intro">
  <h3 id="task-examples" style="margin-top: 0;">Task Examples</h3>
  <p class="subtitle"><em>Videos shown at 8× speed (RGB video from head camera in original 720x720 resolution).</em></p>
  <div class="eval-container">
    <div class="video-tabs eval-tabs"></div>
    <div class="video-panels eval-panels"></div>
  </div>
</div>

## Approach

Our system builds on the **Pi0.5 vision-language-action architecture**, with several key changes:

- **Correlated noise for flow matching**: We train with noise drawn from an empirical action covariance matrix, making training more sample-efficient and enabling correlation-aware inpainting at inference.

- **System 2 stage tracker**: A lightweight module that predicts discrete task stages and feeds progress signals back to the policy, helping avoid early termination and order errors.

- **Correction rules**: Simple heuristics that detect and recover from common failure modes like accidental gripper closures.

- **Action compression**: Spline interpolation for 1.3× speedup without hurting success rate.

<table><tr>
<td><img src="assets/winning-behavior-1k-challenge/popcorn1.png" alt="Grasping door handle"><br><small>Grasping the microwave door</small></td>
<td><img src="assets/winning-behavior-1k-challenge/popcorn2.png" alt="Pressing start button"><br><small>Pressing the start button</small></td>
</tr></table>

## Results

On the held-out evaluation, our approach achieves **q-score ~0.26** with minimal public–private gap.

| Rank | Team | Affiliation | Task Success (private) | Q-Score (private) |
|:----:|:-----|:------------|:----------------------:|:-----------------:|
| **1** | **Robot Learning Collective (ours)** | Independent | 0.124 | **0.260** |
| 2 | Comet | NVIDIA Research | 0.114 | 0.251 |
| 3 | SimpleAI Robot | Beijing Simple AI Technology | 0.108 | 0.159 |
| 4 | The North Star | Huawei CRI EAI Team | 0.076 | 0.120 |
| 5 | Embodied Intelligence | Independent | 0.052 | 0.095 |

Top 5 teams on the held-out test set ([leaderboard](https://behavior.stanford.edu/challenge/leaderboard.html))

### Per individual task results

<div class="side-by-side">
  <div class="text-col">
    <ul>
      <li>Some tasks are almost solved, except under particularly tricky initial conditions.</li>
      <li>For tasks with 0 success, we do not observe that they are generally impossible; instead, they usually contain one tricky step that involves very high-precision manipulation (with low success rate even for human teleoperators) or a carefully followed sequence that is slightly beyond the current model's limits.</li>
      <li>Task duration does not appear to be a fundamental obstacle: longer tasks simply have many more steps, which makes full success harder, but partial success remains very achievable.</li>
    </ul>
  </div>
  <div class="img-col">
    <div class="lightbox-trigger" onclick="openLightbox('assets/winning-behavior-1k-challenge/per_task_and_eps_score.png')">
      <img src="assets/winning-behavior-1k-challenge/per_task_and_eps_score.png" alt="Per-task scores" style="max-height: 600px; width: auto; border-radius: 4px; cursor: zoom-in;">
    </div>
    <small style="display: block; color: #666; margin-top: 0.5rem;">Per-task and per-episode scores sorted by task duration. Green = success; light green = partial success; red = failure. <em>Click to enlarge.</em></small>
  </div>
</div>

<div id="lightbox" class="lightbox" onclick="closeLightbox(event)">
  <div class="lightbox-content">
    <img id="lightbox-img" alt="" onclick="event.stopPropagation()">
    <div id="lightbox-caption"></div>
  </div>
</div>

<div class="eval-block" id="success-evals" data-tabbed-json="assets/winning-behavior-1k-challenge/eval_clips.json" data-tabbed-type="success">
  <h3>Example of 100% Succesful Episodes</h3>
  <p class="subtitle">Select an episode to show 10X-speed.</p>
  <div class="eval-container">
    <div class="eval-tabs"></div>
    <div class="eval-panels"></div>
  </div>
</div>

<script>
function openLightbox(src, caption) {
  const box = document.getElementById('lightbox');
  const img = document.getElementById('lightbox-img');
  const cap = document.getElementById('lightbox-caption');
  if (!box || !img || !cap) return;
  img.src = src;
  img.alt = caption || '';
  cap.textContent = caption || '';
  box.classList.add('active');
  document.addEventListener('keydown', handleLightboxKey);
}

function closeLightbox(event) {
  const box = document.getElementById('lightbox');
  if (!box) return;
  box.classList.remove('active');
  document.removeEventListener('keydown', handleLightboxKey);
}

function handleLightboxKey(e) {
  if (e.key === 'Escape') closeLightbox(e);
}
</script>

<script>
async function renderFailureChart() {
  const chart = document.getElementById('failure-chart');
  if (!chart) return;
  try {
    const resp = await fetch('assets/winning-behavior-1k-challenge/failure_reasons.json');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    const max = Math.max(...data.map(d => d.value), 1);
    
    let activeFilter = 'dexterity';
    
    const updateChart = () => {
      chart.innerHTML = data.map(d => {
        const widthPct = (d.value / max) * 100;
        const color = d.color || '#888';
        const isActive = activeFilter === d.label.toLowerCase();
        return `<div class="row" style="cursor: pointer; opacity: ${!activeFilter || isActive ? '1' : '0.35'};" onclick="toggleFailureFilter('${d.label.toLowerCase()}')">
          <span class="lbl" style="font-weight: ${isActive ? 'bold' : 'normal'};">${d.label}</span>
          <div class="bar" style="width:${widthPct}%; background:${color};"></div>
        </div>`;
      }).join('');
    };

    window.toggleFailureFilter = (reason) => {
      const failBlock = document.getElementById('fail-evals');
      if (!failBlock || !failBlock.filterItems) return;
      
      if (activeFilter === reason) {
        activeFilter = null;
        failBlock.filterItems(null);
      } else {
        activeFilter = reason;
        failBlock.filterItems(item => item.reason && item.reason.toLowerCase() === reason);
      }
      updateChart();
    };

    // Apply initial filter once the video component is ready
    const applyInitialFilter = () => {
      const failBlock = document.getElementById('fail-evals');
      if (failBlock && failBlock.filterItems) {
        failBlock.filterItems(item => item.reason && item.reason.toLowerCase() === 'dexterity');
      } else {
        setTimeout(applyInitialFilter, 50);
      }
    };

    applyInitialFilter();
    updateChart();
  } catch (err) {
    chart.innerHTML = '<small>Failed to load failure reasons.</small>';
    console.error('Failed to load failure reasons', err);
  }
}

document.addEventListener('DOMContentLoaded', renderFailureChart);
</script>

### Failure Analysis

We labeled failure reasons on a subset of tasks (15/50). **Click a bar to filter example videos below:**

<style>
.hbar-chart { margin: 1.5rem auto; max-width: 720px; }
.hbar-chart .row { display: flex; align-items: center; margin-bottom: 6px; transition: opacity 0.2s; }
.hbar-chart .row:hover { background: #f0f4f8; border-radius: 4px; }
.hbar-chart .lbl { width: 160px; text-align: right; padding-right: 12px; font-size: 13px; color: #444; flex-shrink: 0; }
.hbar-chart .bar { height: 16px; border-radius: 3px; min-width: 6px; background: #ccc; }
</style>

<div class="hbar-chart" id="failure-chart"></div>


<div class="eval-block" id="fail-evals" data-tabbed-json="assets/winning-behavior-1k-challenge/failure_examples.json" data-tabbed-type="fail">
  <h3>Examples of Failure Episodes</h3>
  <p class="subtitle">Select an episode to show 5X-speed clip</p>
  <div class="eval-container">
    <div class="fail-tabs eval-tabs"></div>
    <div class="fail-panels eval-panels"></div>
  </div>
</div>

Please note that fail reason labelling are subjective, and there to provide the big picture. Refer to all evaluation videos and scores [here](https://drive.google.com/drive/folders/12Wb21mQi6UP8_OMKPGNHOII_-MV3oxk-?usp=sharing).

## Recovery from cross-task learning

Training on all 50 tasks leads to **emergent recovery behaviors**. Single-task models never recover from mistakes; the multi-task model learns to pick up fallen items and retry.

<style>
.video-pair { display: flex; gap: 1rem; margin: 1rem 0; }
.video-pair > div { flex: 1; }
.video-pair video { width: 100%; height: auto; display: block; }
.video-pair .label { font-size: 0.9rem; margin-top: 0.5rem; color: #555; }
@media (max-width: 820px) {
  .video-pair { flex-direction: column; }
  .video-pair > div { width: 100%; }
}
.recovery-tabs { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
.recovery-tabs button { padding: 0.5rem 1rem; border: 1px solid #ccc; background: #f5f5f5; cursor: pointer; border-radius: 4px; font-size: 0.9rem; }
.recovery-tabs button.active { background: #0366d6; color: white; border-color: #0366d6; }
.recovery-panel { display: none; }
.recovery-panel.active { display: block; }
</style>

<div class="recovery-tabs">
  <button class="active" onclick="showRecovery(0)">Example 1</button>
  <button onclick="showRecovery(1)">Example 2</button>
</div>

<div class="recovery-panel active" id="recovery-0">
  <div class="video-pair">
    <div>
      <video autoplay muted playsinline controls><source src="https://pub-a5638afed52c4226aac6a1e71ecc323c.r2.dev/behavior_report/recovery_picture_early_ex1.mp4" type="video/mp4"></video>
      <div class="label">Before: fails to recover</div>
    </div>
    <div>
      <video muted playsinline controls><source src="https://pub-a5638afed52c4226aac6a1e71ecc323c.r2.dev/behavior_report/recovery_picture_late_ex1.mp4" type="video/mp4"></video>
      <div class="label">After: robust recovery</div>
    </div>
  </div>
</div>

<div class="recovery-panel" id="recovery-1">
  <div class="video-pair">
    <div>
      <video autoplay muted playsinline controls><source src="https://pub-a5638afed52c4226aac6a1e71ecc323c.r2.dev/behavior_report/recovery_picture_early_ex2.mp4" type="video/mp4"></video>
      <div class="label">Before: error cascades to failure</div>
    </div>
    <div>
      <video muted playsinline controls><source src="https://pub-a5638afed52c4226aac6a1e71ecc323c.r2.dev/behavior_report/recovery_picture_late_ex2.mp4" type="video/mp4"></video>
      <div class="label">After: recovers and completes task</div>
    </div>
  </div>
</div>

<script>
function showRecovery(index) {
  document.querySelectorAll('.recovery-tabs button').forEach((btn, i) => btn.classList.toggle('active', i === index));
  document.querySelectorAll('.recovery-panel').forEach((panel, i) => panel.classList.toggle('active', i === index));
}
</script>


Join our Discord:
<div class="top-buttons">
  <a href="https://discord.gg/Jr8tcnVNGw" target="_blank"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
</div>

---

<style>
.team-grid { display: flex; justify-content: center; gap: 3rem; flex-wrap: wrap; margin: 2rem 0; }
.team-member { text-align: center; }
.team-member img { width: 120px; height: 120px; border-radius: 50%; object-fit: cover; border: 3px solid #ddd; }
.team-member p { margin-top: 0.75rem; font-weight: 500; color: #333; }
</style>

<div class="team-grid">
  <div class="team-member">
    <img src="assets/winning-behavior-1k-challenge/1707732561897.jpeg" alt="Ilia Larchenko">
    <p><a href="https://www.linkedin.com/in/larchenko/">Ilia Larchenko</a></p>
  </div>
  <div class="team-member">
    <img src="assets/winning-behavior-1k-challenge/1736204199843.jpeg" alt="Gleb Zarin">
    <p><a href="https://www.linkedin.com/in/zaringleb/">Gleb Zarin</a></p>
  </div>
  <div class="team-member">
    <img src="assets/winning-behavior-1k-challenge/1670851469680.jpeg" alt="Akash Karnatak">
    <p><a href="https://www.linkedin.com/in/akash-karnatak-9027371a0/">Akash Karnatak</a></p>
  </div>
</div>
