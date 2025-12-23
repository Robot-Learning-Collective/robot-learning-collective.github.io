---
layout: default
title: Winning the BEHAVIOR-1K Challenge (NeurIPS 2025)
description: Can one robotics model trained via Imitation Learning solve 50 household tasks?
header_links:
  - text: View on GitHub
    url: https://github.com/IliaLarchenko/behavior-1k-solution
  - text: ArXiv
    url: https://arxiv.org/abs/2512.06951
  - text: Challenge
    url: https://behavior.stanford.edu/index.html
---

We present a vision-action policy that achieved first place in the 2025 BEHAVIOR Challenge, a large-scale benchmark of 50 long-horizon household tasks in photo-realistic simulation requiring bimanual manipulation, navigation, and object interaction.
Building on the Pi0.5 architecture, we introduce several innovations, including **correlated noise for flow matching** to improve training efficiency and produce smoother action sequences, as well as **learnable mixed layer attention** and **System 2 stage tracking** for ambiguity resolution. Training uses **multi-sample flow matching** to reduce variance, while inference applies action compression and heuristics to overcome dataset limitations. 
Our method achieves a 26% q-score on both public and private leaderboards, and this report provides a detailed **analysis of observed failure modes**.

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

## Challenge

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
    if (vid) {
      if (vid.dataset.src && !vid.src) {
        vid.src = vid.dataset.src;
        vid.load();
      }
      vid.play().catch(() => {});
    }
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
      if (vid) {
        if (!isActive) {
          vid.pause();
        } else if (vid.dataset.src && !vid.src) {
          vid.src = vid.dataset.src;
          vid.load();
        }
      }
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
      <video autoplay muted playsinline controls data-src="${item.src}">
      </video>
      ${item.instruction ? `<div class="eval-note"><strong>Instruction:</strong> ${item.instruction}</div>` : ''}
    `,
  },
  success: {
    panelClass: 'eval-panel',
    renderButtonLabel: (clip) => clip.label,
    renderPanelHTML: (clip) => `
      <video autoplay muted playsinline controls data-src="${clip.src}">
      </video>
      ${clip.instruction ? `<div class="eval-instruction"><strong>Instruction:</strong> ${clip.instruction}</div>` : ''}
      <div class="eval-note"><strong>Note:</strong> ${clip.note}</div>
    `,
  },
  fail: {
    panelClass: 'eval-panel fail-panel',
    renderButtonLabel: (ex) => `${ex.task} #${ex.episode}`,
    renderPanelHTML: (ex) => `
      <video autoplay muted playsinline controls data-src="${ex.video_path}">
      </video>
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
  <h3 id="task-examples" style="margin-top: 0;">Training Examples</h3>
  <p class="subtitle"><em>Each task comes with 200 demonstrations recorded via teleoperation. Video: robot head camera 8× speed.</em></p>
  <div class="eval-container">
    <div class="video-tabs eval-tabs"></div>
    <div class="video-panels eval-panels"></div>
  </div>
</div>
<style>
.video-tabs { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1rem; }
.video-tabs button { padding: 0.5rem 1rem; border: 1px solid #ccc; background: #f5f5f5; cursor: pointer; border-radius: 4px; font-size: 0.85rem; }
.video-tabs button.active { background: #0366d6; color: white; border-color: #0366d6; }
.video-panel { display: none; }
.video-panel.active { display: block; }
.video-panel video { width: 100%; height: auto; max-width: 640px; display: block; margin: 0 auto; border-radius: 6px; }
</style>

The Challenge includes 50 tasks set in multiple house-like environments. The main difficulties are:
* **Long-horizon execution:** Tasks run for an average of 6.6 minutes.
* **Bimanual manipulation:** Coordinated control of two 7-DOF arms with parallel jaw grippers.
* **Mobile navigation:** Operation in realistic indoor and outdoor environments.
* **Non-Markovian states:** Many states are visually ambiguous. Without memory of past actions or explicit stage tracking, the policy cannot distinguish between these states and may take incorrect actions.
* **No recovery demonstrations:** The training set contains only successful demonstrations, spanning a very narrow manifold of possible trajectories.
* **Data:** Each task provides 200 demonstrations. Public evaluation uses 10 additional instances; the private leaderboard evaluates performance on a separate held-out set of another 10 instances.
* **High compute requirements:** Training on all 1200 hours of data takes weeks, and full evaluation across all tasks takes several days.

## Model

<div id="model-stepper" style="margin: 2rem auto; max-width: 1000px; font-family: sans-serif;">
  <div style="display: flex; justify-content: space-between; align-items: stretch; margin-bottom: 1.5rem; background: #f8fafc; border-radius: 8px; border: 1px solid #e5e7eb; overflow: hidden;">
    <button id="prev-step" style="width: 16.66%; min-width: 60px; cursor: pointer; border: none; border-right: 1px solid #e5e7eb; background: white; font-weight: bold; font-size: 2.5rem; line-height: 1; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">&larr;</button>
    <div style="text-align: center; flex-grow: 1; padding: 1.2rem 1rem; display: flex; flex-direction: column; justify-content: center;">
      <h4 id="step-title" style="margin: 0 0 0.5rem 0; color: #1e293b; font-size: 1.1rem;">VLA Foundation</h4>
      <p id="step-description" style="margin: 0; font-size: 0.95rem; color: #64748b; line-height: 1.4;">Our policy is based on the Pi0.5 vision-language-action (VLA) architecture, combining a SigLIP vision encoder with a Gemma LLM backbone.</p>
    </div>
    <button id="next-step" style="width: 16.66%; min-width: 60px; cursor: pointer; border: none; border-left: 1px solid #e5e7eb; background: white; font-weight: bold; font-size: 2.5rem; line-height: 1; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">&rarr;</button>
  </div>

  <div id="svg-container" style="position: relative; width: 100%; border: 1px solid #eee; border-radius: 8px; background: white; overflow: hidden; min-height: 400px; display: flex; align-items: center; justify-content: center;">
    <div id="svg-loading" style="padding: 2rem; color: #64748b;">Loading interactive model...</div>
  </div>
  
  <div style="margin-top: 1rem; display: flex; justify-content: center; gap: 0.5rem;" id="step-dots"></div>
</div>

<style>
  #model-stepper .dot { width: 10px; height: 10px; border-radius: 50%; background: #cbd5e1; cursor: pointer; transition: background 0.2s; }
  #model-stepper .dot.active { background: #0366d6; }
</style>

<script>
document.addEventListener("DOMContentLoaded", function() {
  const container = document.getElementById("svg-container");
  const titleEl = document.getElementById("step-title");
  const descEl = document.getElementById("step-description");
  const dotsContainer = document.getElementById("step-dots");
  const prevBtn = document.getElementById("prev-step");
  const nextBtn = document.getElementById("next-step");

  let currentStep = 0;
  let svg = null;

  const steps = [
    { title: "VLA Foundation", description: "Our policy is based on the Pi0.5 vision-language-action architecture, which combines a VLM (3B PaliGemma) and a flow-matching transformer-based action expert.", show: ["base"] },
    { title: "Task Embeddings", description: "Instead of text prompts, we use learned task embeddings to maintain multi-task capabilities with simpler inputs, as language comprehension was not required for this challenge.", show: ["base", "task_emb"] },
    { title: "System 2 Reasoning", description: "We add a simple System 2 for task progress tracking; the VLA model predicts the current stage and System 2 uses a heuristic to filter large 'jumps' in the progress estimation before feeding the stage back into the model for the next timestep.", show: ["base", "task_emb", "system2"] },
    { title: "Hybrid Attention", description: "A learnable mixed-layer attention mechanism allows the action expert to attend to optimal combinations of VLM layers.", show: ["base", "system2", "task_emb", "hybrid_att"] },
    { title: "Multi-sample flow matching", description: "During training we sample flow matching multiple times per single VLM inference to optimize training speed.", show: ["base", "system2", "task_emb", "hybrid_att", "expert_steps"] },
    { title: "Correlated noise for flow matching", description: "To account for high correlation across robot action space and time, we train with correlated noise drawn from empirical action covariance to model natural robot motion structure.", show: ["base", "system2", "task_emb", "hybrid_att", "expert_steps", "noise"] },
    { title: "Soft Inpainting", description: "Soft inpainting at inference time ensures smooth, continuous transitions between action sequences.", show: ["base", "system2", "task_emb", "hybrid_att", "expert_steps", "noise", "inpainting"] },
    { title: "Execution Speedup", description: "Cubic spline compression speeds up robot execution by 1.3x without hurting success rate.", show: ["base", "system2", "task_emb", "hybrid_att", "expert_steps", "noise", "inpainting", "speedup"] }
  ];

  const calloutMap = {
    "system2": {
      shapes: ["m15.695", "m20.650"],
      colors: ["#ffd966"],
      textPrefixes: ["m51.943", "m84.177", "m81.410", "m331.146", "m43.884"]
    },
    "task_emb": {
      shapes: ["m358.316"],
      textPrefixes: ["m394.419", "m383.771"]
    },
    "hybrid_att": {
      shapes: ["m559.717"],
      textPrefixes: ["m597.012", "m625.7"]
    },
    "expert_steps": {
      shapes: ["m763.895 227"],
      textPrefixes: ["m779.693", "m785.677"]
    },
    "noise": {
      shapes: ["m763.895 435"],
      textPrefixes: ["m804.806"]
    },
    "inpainting": {
      shapes: ["m763.895 331"],
      textPrefixes: ["m789.791", "m820.912"]
    },
    "speedup": {
      shapes: ["m763.895 123"],
      textPrefixes: ["m794.673", "m791.447"]
    }
  };

  function updateStep() {
    if (!svg) return;
    const step = steps[currentStep];
    titleEl.textContent = step.title;
    descEl.textContent = step.description;
    
    svg.querySelectorAll("path").forEach(p => {
      const fill = p.getAttribute("fill");
      const stroke = p.getAttribute("stroke");
      const d = p.getAttribute("d") || "";
      let featureKey = null;

      for (const [key, cfg] of Object.entries(calloutMap)) {
        if ((cfg.shapes && cfg.shapes.some(s => d.startsWith(s))) || 
            (cfg.colors && (cfg.colors.includes(fill) || cfg.colors.includes(stroke))) ||
            (cfg.textPrefixes && cfg.textPrefixes.some(tp => d.startsWith(tp)))) {
          featureKey = key;
          break;
        }
      }

      let visible = false;
      if (!featureKey) {
        if (step.show.includes("base")) visible = true;
      } else {
        if (step.show.includes(featureKey)) visible = true;
      }
      p.style.opacity = visible ? "1" : "0.08";
      p.style.transition = "opacity 0.3s ease";
    });

    dotsContainer.querySelectorAll(".dot").forEach((dot, i) => {
      dot.classList.toggle("active", i === currentStep);
    });

    prevBtn.disabled = currentStep === 0;
    nextBtn.disabled = currentStep === steps.length - 1;
    prevBtn.style.opacity = prevBtn.disabled ? "0.3" : "1";
    nextBtn.style.opacity = nextBtn.disabled ? "0.3" : "1";
  }

  fetch("assets/winning-behavior-1k-challenge/interactive_model.svg")
    .then(r => r.text())
    .then(text => {
      container.innerHTML = text;
      svg = container.querySelector("svg");
      if (!svg) return;
      
      svg.style.width = "100%";
      svg.style.height = "auto";
      svg.style.maxHeight = "70vh";

      // Create dots
      steps.forEach((_, i) => {
        const dot = document.createElement("div");
        dot.className = "dot";
        dot.addEventListener("click", () => { currentStep = i; updateStep(); });
        dotsContainer.appendChild(dot);
      });

      updateStep();
    });

  prevBtn.addEventListener("click", () => { if (currentStep > 0) { currentStep--; updateStep(); } });
  nextBtn.addEventListener("click", () => { if (currentStep < steps.length - 1) { currentStep++; updateStep(); } });
});
</script>


## Results

On the held-out evaluation, our approach achieves **q-score ~0.26** (including partial successes) with minimal public–private gap.

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
      <li>Average task success rate varies significantly. Some tasks are almost solved, except under particularly tricky initial conditions, while in others, the model was unsuccessful across all 10 trials.</li>
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
  <h3>Examples of 100% Successful Episodes</h3>
  <p class="subtitle">Select an episode to show 10X-speed clip</p>
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
      
      const activeData = data.find(d => d.label.toLowerCase() === activeFilter);
      const descBox = document.getElementById('failure-desc');
      if (descBox) {
        descBox.innerHTML = activeData ? `<strong>${activeData.label}:</strong> ${activeData.desc}` : '<em>Select a reason to see the explanation and video examples.</em>';
      }
    };

    window.toggleFailureFilter = (reason) => {
      const failBlock = document.getElementById('fail-evals');
      if (!failBlock || !failBlock.filterItems) return;
      
      if (activeFilter === reason) {
        activeFilter = null;
        failBlock.filterItems(null);
      } else {
        activeFilter = reason;
        // Handle potential naming mismatches for filtering
        const filterKey = (reason === 'whole-body coordination') ? 'whole-body' : reason;
        failBlock.filterItems(item => item.reason && item.reason.toLowerCase() === filterKey);
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

To analyze why the robot fails, we labeled failures on a subset of tasks (15/50). *Select a reason to see the explanation and video examples:*

<style>
.hbar-chart { margin: 1.5rem auto; max-width: 720px; }
.hbar-chart .row { display: flex; align-items: center; margin-bottom: 6px; transition: opacity 0.2s; }
.hbar-chart .row:hover { background: #f0f4f8; border-radius: 4px; }
.hbar-chart .lbl { width: 160px; text-align: right; padding-right: 12px; font-size: 13px; color: #444; flex-shrink: 0; }
.hbar-chart .bar { height: 16px; border-radius: 3px; min-width: 6px; background: #ccc; }
</style>

<div class="hbar-chart" id="failure-chart"></div>

<div id="failure-desc" style="max-width: 720px; margin: 0 auto 2rem; padding: 1rem; background: #f0f6ff; border-radius: 6px; border-left: 4px solid #0366d6; font-size: 0.95rem; min-height: 3em;">
  <em>Select a reason to see the explanation and video examples.</em>
</div>


<div class="eval-block" id="fail-evals" data-tabbed-json="assets/winning-behavior-1k-challenge/failure_examples.json" data-tabbed-type="fail">
  <h3>Examples of Failure Episodes</h3>
  <p class="subtitle">Select an episode to show 5X-speed clip</p>
  <div class="eval-container">
    <div class="fail-tabs eval-tabs"></div>
    <div class="fail-panels eval-panels"></div>
  </div>
</div>

Please note that fail reason labeling is subjective, and is there to provide a big picture. Refer to all evaluation videos and scores [here](https://drive.google.com/drive/folders/12Wb21mQi6UP8_OMKPGNHOII_-MV3oxk-?usp=sharing).

### Recovery from cross-task learning

Training on all 50 tasks leads to **emergent behaviors**. Due to the data collection process, single-task models never recover from mistakes; the multi-task model trained sufficiently long learns to use skills from other tasks to recover (e.g., to pick up a dropped item).

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
      <video autoplay muted playsinline controls data-src="https://pub-a5638afed52c4226aac6a1e71ecc323c.r2.dev/behavior_report/recovery_picture_early_ex1.mp4"></video>
      <div class="label">Single task model: no attempt to recover</div>
    </div>
    <div>
      <video muted playsinline controls data-src="https://pub-a5638afed52c4226aac6a1e71ecc323c.r2.dev/behavior_report/recovery_picture_late_ex1.mp4"></video>
      <div class="label">Multi task model: non-trivial recovery</div>
    </div>
  </div>
</div>

<div class="recovery-panel" id="recovery-1">
  <div class="video-pair">
    <div>
      <video autoplay muted playsinline controls data-src="https://pub-a5638afed52c4226aac6a1e71ecc323c.r2.dev/behavior_report/recovery_picture_early_ex2.mp4"></video>
      <div class="label">Single task model: no attempt to recover</div>
    </div>
    <div>
      <video muted playsinline controls data-src="https://pub-a5638afed52c4226aac6a1e71ecc323c.r2.dev/behavior_report/recovery_picture_late_ex2.mp4"></video>
      <div class="label">Multi task model: non-trivial recovery</div>
    </div>
  </div>
</div>

<script>
function showRecovery(index) {
  document.querySelectorAll('.recovery-tabs button').forEach((btn, i) => btn.classList.toggle('active', i === index));
  document.querySelectorAll('.recovery-panel').forEach((panel, i) => {
    const isActive = i === index;
    panel.classList.toggle('active', isActive);
    const videos = panel.querySelectorAll('video');
    videos.forEach(vid => {
      if (isActive) {
        if (vid.dataset.src && !vid.src) {
          vid.src = vid.dataset.src;
          vid.load();
        }
        vid.play().catch(() => {});
      } else {
        vid.pause();
      }
    });
  });
}

// Lazy load recovery videos when they come into view
document.addEventListener('DOMContentLoaded', () => {
  const recoverySection = document.querySelector('.recovery-tabs');
  if (recoverySection && typeof IntersectionObserver !== 'undefined') {
    const observer = new IntersectionObserver((entries) => {
      if (entries[0].isIntersecting) {
        showRecovery(0); // Load/play first tab
        observer.disconnect();
      }
    }, { threshold: 0.1 });
    observer.observe(recoverySection);
  }
});
</script>
## Summary

* The dominant failure modes closely reflect real-world challenges faced by imitation-learning-based robotics models, supporting the role of the BEHAVIOR benchmark as a valuable test bed for evaluating new approaches.
* Evidence from cross-task learning reinforces the view that imitation-learning-based models benefit significantly from training on diverse datasets.
* Due to limited resources for thorough ablation studies, we cannot precisely identify which components were critical. Nonetheless, the combined approach proves robust and outperforms comparable Pi0.5-based methods despite having a smaller training budget.

---


<style>
.team-grid { display: flex; justify-content: center; gap: 3rem; flex-wrap: wrap; margin: 2rem 0; }
.team-member { text-align: center; }
.team-member img { width: 140px; height: 140px; border-radius: 50%; object-fit: cover; border: 3px solid #ddd; transition: transform 0.2s; }
.team-member img:hover { transform: scale(1.05); }
.team-member p { margin-top: 0.75rem; font-weight: 500; color: #333; }
</style>

<div class="team-grid">
  <div class="team-member">
    <img src="assets/winning-behavior-1k-challenge/image3.jpg" alt="Ilia Larchenko">
    <p><a href="https://www.linkedin.com/in/larchenko/">Ilia Larchenko</a></p>
  </div>
  <div class="team-member">
    <img src="assets/winning-behavior-1k-challenge/1736204199843.jpeg" alt="Gleb Zarin">
    <p><a href="https://www.linkedin.com/in/zaringleb/">Gleb Zarin</a></p>
  </div>
  <div class="team-member">
    <img src="assets/winning-behavior-1k-challenge/photo_2025-12-22_20-08-47.jpg" alt="Akash Karnatak">
    <p><a href="https://www.linkedin.com/in/akash-karnatak-9027371a0/">Akash Karnatak</a></p>
  </div>
</div>

### Acknowledgments
This work was made possible by the generous support of [Nebius](https://nebius.com/), who provided the high-performance cloud GPU compute resources required to train our models.

<a href="https://nebius.com/" target="_blank">
  <img src="assets/winning-behavior-1k-challenge/idOGJsi66K_logos.webp" alt="Nebius Logo" style="height: 32px; width: auto; margin: 10px 0;">
</a>

We would also like to thank the following people for their help and
support: [Vladimir Ershov](https://www.linkedin.com/in/vladimir-ershov-33559466/), [Justyna Ilczuk](https://www.linkedin.com/in/justynailczuk/), [Andrey Mulenkov](https://www.linkedin.com/in/andrey-mulenkov/).

Interested in Robot Learning? Join our [Discord](https://discord.gg/Jr8tcnVNGw) to discuss the Challenge results and collaboration.