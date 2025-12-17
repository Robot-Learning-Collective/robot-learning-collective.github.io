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
</style>

<div class="top-buttons">
  <a href="https://arxiv.org/abs/2512.06951" target="_blank"><img src="https://img.shields.io/badge/arXiv-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://github.com/IliaLarchenko/behavior-1k-solution" target="_blank"><img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a>
  <a href="https://huggingface.co/IliaLarchenko/behavior_submission" target="_blank"><img src="https://img.shields.io/badge/🤗_Hugging_Face-ffd21e?style=for-the-badge" alt="Hugging Face"></a>
  <a href="https://discord.gg/Jr8tcnVNGw" target="_blank"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
</div>

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

## Task Examples

*Videos shown at 8× speed.*

<style>
.video-tabs { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1rem; }
.video-tabs button { padding: 0.5rem 1rem; border: 1px solid #ccc; background: #f5f5f5; cursor: pointer; border-radius: 4px; font-size: 0.85rem; }
.video-tabs button.active { background: #0366d6; color: white; border-color: #0366d6; }
.video-panel { display: none; }
.video-panel.active { display: block; }
</style>

<div class="video-tabs">
  <button class="active" onclick="showTab(0)">Task 1: Picking Up Trash</button>
  <button onclick="showTab(1)">Task 12: Preparing Lunch Box</button>
  <button onclick="showTab(2)">Task 16: Moving Boxes</button>
  <button onclick="showTab(3)">Task 32: Wash Baseball Cap</button>
  <button onclick="showTab(4)">Task 38: Spraying For Bugs</button>
  <button onclick="showTab(5)">Task 42: Chop An Onion</button>
</div>

<div class="video-panel active" id="panel-0">
  <video width="640" height="360" autoplay muted playsinline controls><source src="https://pub-a5638afed52c4226aac6a1e71ecc323c.r2.dev/behavior_report/episode_00010120_fast.mp4" type="video/mp4"></video>
  <p><small>Put the three cans of soda from the living room inside the trash can in the kitchen.</small></p>
</div>
<div class="video-panel" id="panel-1">
  <video width="640" height="360" autoplay muted playsinline controls><source src="https://pub-a5638afed52c4226aac6a1e71ecc323c.r2.dev/behavior_report/episode_00120030_fast.mp4" type="video/mp4"></video>
  <p><small>Put apple halves, sandwich, and cookie into the packing box, then add tea from the fridge.</small></p>
</div>
<div class="video-panel" id="panel-2">
  <video width="640" height="360" autoplay muted playsinline controls><source src="https://pub-a5638afed52c4226aac6a1e71ecc323c.r2.dev/behavior_report/episode_00160020_fast.mp4" type="video/mp4"></video>
  <p><small>Move two storage containers from living room to garage and stack them.</small></p>
</div>
<div class="video-panel" id="panel-3">
  <video width="640" height="360" autoplay muted playsinline controls><source src="https://pub-a5638afed52c4226aac6a1e71ecc323c.r2.dev/behavior_report/episode_00320050_fast.mp4" type="video/mp4"></video>
  <p><small>Wash two baseball caps on the countertop using the washer until clean.</small></p>
</div>
<div class="video-panel" id="panel-4">
  <video width="640" height="360" autoplay muted playsinline controls><source src="https://pub-a5638afed52c4226aac6a1e71ecc323c.r2.dev/behavior_report/episode_00380020_fast.mp4" type="video/mp4"></video>
  <p><small>Pick up pesticide atomizer and spray both potted plants in the garden.</small></p>
</div>
<div class="video-panel" id="panel-5">
  <video width="640" height="360" autoplay muted playsinline controls><source src="https://pub-a5638afed52c4226aac6a1e71ecc323c.r2.dev/behavior_report/episode_00420030_fast.mp4" type="video/mp4"></video>
  <p><small>Dice the onion on the chopping board, put it in the bowl, then wash the knife and board.</small></p>
</div>

<script>
function showTab(index) {
  document.querySelectorAll('.video-tabs button').forEach((btn, i) => btn.classList.toggle('active', i === index));
  document.querySelectorAll('.video-panel').forEach((panel, i) => panel.classList.toggle('active', i === index));
}
</script>

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

<small>Top 5 teams on the held-out test set. [Leaderboard](https://behavior.stanford.edu/challenge/leaderboard.html)</small>

<div class="lightbox-trigger" onclick="openLightbox('assets/winning-behavior-1k-challenge/per_task_and_eps_score.png', 'Per-task and per-episode scores. Green = success; red = failure.')">
  <img src="assets/winning-behavior-1k-challenge/per_task_and_eps_score.png" alt="Per-task scores" style="max-height: 400px; width: auto; border-radius: 4px; cursor: zoom-in;">
</div>
<small style="display: block; color: #666;">Per-task and per-episode scores. Green = success; red = failure. *Click to enlarge.*</small>

<div id="lightbox" class="lightbox" onclick="closeLightbox(event)">
  <div class="lightbox-content">
    <img id="lightbox-img" alt="" onclick="event.stopPropagation()">
    <div id="lightbox-caption"></div>
  </div>
</div>

<style>
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
</style>

<div class="eval-block">
  <h3>Example of 100% Succesful Episodes</h3>
  <p class="subtitle">Select an episode to show 10X-speed clip and its notes.</p>
  <div class="eval-container">
    <div class="eval-tabs"></div>
    <div class="eval-panels"></div>
  </div>
</div>

<script>
function showEvalTab(index) {
  document.querySelectorAll('.eval-tabs button').forEach((btn, i) => btn.classList.toggle('active', i === index));
  document.querySelectorAll('.eval-panel').forEach((panel, i) => panel.classList.toggle('active', i === index));
}

async function renderEvalClips() {
  const tabs = document.querySelector('.eval-tabs');
  const panels = document.querySelector('.eval-panels');
  if (!tabs || !panels) return;

  try {
    const resp = await fetch('assets/winning-behavior-1k-challenge/eval_clips.json');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const evalClips = await resp.json();

    evalClips.forEach((clip, idx) => {
      const btn = document.createElement('button');
      btn.textContent = clip.label;
      btn.addEventListener('click', () => showEvalTab(idx));
      tabs.appendChild(btn);

      const panel = document.createElement('div');
      panel.className = 'eval-panel';
      panel.id = `eval-${idx}`;
      panel.innerHTML = `
        <video autoplay muted playsinline controls>
          <source src="${clip.src}" type="video/mp4">
        </video>
        ${clip.instruction ? `<div class="eval-instruction"><strong>Instruction:</strong> ${clip.instruction}</div>` : ''}
        <div class="eval-note"><strong>Note:</strong> ${clip.note}</div>
      `;
      panels.appendChild(panel);
    });

    showEvalTab(0);
  } catch (err) {
    tabs.innerHTML = '<small>Failed to load clips.</small>';
    console.error('Failed to load eval clips', err);
  }
}

document.addEventListener('DOMContentLoaded', renderEvalClips);
</script>

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
    chart.innerHTML = data.map(d => {
      const widthPct = (d.value / max) * 100;
      const color = d.color || '#888';
      return `<div class="row">
        <span class="lbl">${d.label}</span>
        <div class="bar" style="width:${widthPct}%; background:${color};"></div>
        <span class="val">${d.value}</span>
      </div>`;
    }).join('');
  } catch (err) {
    chart.innerHTML = '<small>Failed to load failure reasons.</small>';
    console.error('Failed to load failure reasons', err);
  }
}

document.addEventListener('DOMContentLoaded', renderFailureChart);
</script>


### Failure Analysis

We labeled failure reasons on a subset of tasks (15/50):

<style>
.hbar-chart { margin: 1.5rem auto; max-width: 720px; }
.hbar-chart .row { display: flex; align-items: center; margin-bottom: 6px; }
.hbar-chart .lbl { width: 160px; text-align: right; padding-right: 12px; font-size: 13px; color: #444; flex-shrink: 0; }
.hbar-chart .bar { height: 16px; border-radius: 3px; min-width: 6px; background: #ccc; }
.hbar-chart .val { margin-left: 8px; font-size: 12px; color: #666; flex-shrink: 0; }
</style>

<div class="hbar-chart" id="failure-chart"></div>

<small>Dexterity is the dominant failure mode (~33%), validating our VLA-based approach.</small>

## Recovery from cross-task learning

Training on all 50 tasks leads to **emergent recovery behaviors**. Single-task models never recover from mistakes; the multi-task model learns to pick up fallen items and retry.

<style>
.video-pair { display: flex; gap: 1rem; margin: 1rem 0; }
.video-pair > div { flex: 1; }
.video-pair video { width: 100%; height: auto; display: block; }
.video-pair .label { font-size: 0.9rem; margin-top: 0.5rem; color: #555; }
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

---

## Team

<style>
.team-grid { display: flex; justify-content: center; gap: 3rem; flex-wrap: wrap; margin: 2rem 0; }
.team-member { text-align: center; }
.team-member img { width: 120px; height: 120px; border-radius: 50%; object-fit: cover; border: 3px solid #ddd; }
.team-member p { margin-top: 0.75rem; font-weight: 500; color: #333; }
</style>

<div class="team-grid">
  <div class="team-member">
    <img src="assets/winning-behavior-1k-challenge/1707732561897.jpeg" alt="Ilia Larchenko">
    <p>Ilia Larchenko</p>
  </div>
  <div class="team-member">
    <img src="assets/winning-behavior-1k-challenge/1736204199843.jpeg" alt="Gleb Zarin">
    <p>Gleb Zarin</p>
  </div>
  <div class="team-member">
    <img src="assets/winning-behavior-1k-challenge/1670851469680.jpeg" alt="Akash Karnatak">
    <p>Akash Karnatak</p>
  </div>
</div>
