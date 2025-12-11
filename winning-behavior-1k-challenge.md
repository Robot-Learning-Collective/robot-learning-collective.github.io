---
layout: default
title: Winning the BEHAVIOR-1K Challenge
---

The Robot Learning Collective took **1st place** in the 2025 **BEHAVIOR-1K Challenge**, a large-scale benchmark of 50 long-horizon household tasks in photo-realistic simulation. Each episode spans minutes of bimanual manipulation and navigation, with a single policy expected to generalize across diverse activities such as turning on a radio, cooking a hotdog, or tidying a room.

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
<td><img src="media/popcorn1.png" alt="Grasping door handle"><br><small>Grasping the microwave door</small></td>
<td><img src="media/popcorn2.png" alt="Pressing start button"><br><small>Pressing the start button</small></td>
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

<small>Top 5 teams on the held-out test set. [Full leaderboard →](https://behavior.stanford.edu/challenge/leaderboard.html)</small>

<a href="media/per_task_and_eps_score.png" target="_blank">
  <img src="media/per_task_and_eps_score.png" alt="Per-task scores" style="max-height: 400px; width: auto; border-radius: 4px; cursor: zoom-in;">
</a>
<small style="display: block; color: #666;">Per-task and per-episode scores. Green = success; red = failure. *Click to enlarge.*</small>

### Failure Analysis

We labeled failure reasons on a subset of tasks (15/50):

<style>
.hbar-chart { margin: 1.5rem 0; max-width: 500px; }
.hbar-chart .row { display: flex; align-items: center; margin-bottom: 5px; }
.hbar-chart .lbl { width: 140px; text-align: right; padding-right: 10px; font-size: 12px; color: #444; }
.hbar-chart .bar { height: 16px; border-radius: 2px; min-width: 4px; }
.hbar-chart .val { margin-left: 6px; font-size: 11px; color: #666; }
</style>

<div class="hbar-chart">
  <div class="row"><span class="lbl">Dexterity</span><div class="bar" style="width: 255px; background: #a08070;"></div><span class="val">51</span></div>
  <div class="row"><span class="lbl">Order</span><div class="bar" style="width: 130px; background: #d4a04a;"></div><span class="val">26</span></div>
  <div class="row"><span class="lbl">Confusion</span><div class="bar" style="width: 115px; background: #c76b8f;"></div><span class="val">23</span></div>
  <div class="row"><span class="lbl">Robot fell</span><div class="bar" style="width: 75px; background: #aaa;"></div><span class="val">15</span></div>
  <div class="row"><span class="lbl">Reasoning</span><div class="bar" style="width: 70px; background: #e07060;"></div><span class="val">14</span></div>
  <div class="row"><span class="lbl">Search</span><div class="bar" style="width: 45px; background: #d4a04a;"></div><span class="val">9</span></div>
  <div class="row"><span class="lbl">Robot stuck</span><div class="bar" style="width: 25px; background: #b99ad6;"></div><span class="val">5</span></div>
  <div class="row"><span class="lbl">Head camera</span><div class="bar" style="width: 20px; background: #e07060;"></div><span class="val">4</span></div>
  <div class="row"><span class="lbl">Navigation</span><div class="bar" style="width: 15px; background: #aaa;"></div><span class="val">3</span></div>
  <div class="row"><span class="lbl">Unknown</span><div class="bar" style="width: 15px; background: #aaa;"></div><span class="val">3</span></div>
  <div class="row"><span class="lbl">Reachability</span><div class="bar" style="width: 10px; background: #d4a04a;"></div><span class="val">2</span></div>
  <div class="row"><span class="lbl">Catastrophic</span><div class="bar" style="width: 5px; background: #7ab87a;"></div><span class="val">1</span></div>
  <div class="row"><span class="lbl">Whole-body</span><div class="bar" style="width: 5px; background: #b99ad6;"></div><span class="val">1</span></div>
</div>

<small>**Dexterity** (grasping/releasing) is the dominant failure mode (~33%), validating our VLA-based approach.</small>

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
    <img src="media/1707732561897.jpeg" alt="Ilia Larchenko">
    <p>Ilia Larchenko</p>
  </div>
  <div class="team-member">
    <img src="media/1736204199843.jpeg" alt="Gleb Zarin">
    <p>Gleb Zarin</p>
  </div>
  <div class="team-member">
    <img src="media/1670851469680.jpeg" alt="Akash Karnatak">
    <p>Akash Karnatak</p>
  </div>
</div>
