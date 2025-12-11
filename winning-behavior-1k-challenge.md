---
layout: default
title: Winning the BEHAVIOR-1K Challenge
---

The Robot Learning Collective took **1st place** in the 2025 **BEHAVIOR-1K Challenge**, a large-scale benchmark of 50 long-horizon household tasks in photo-realistic simulation. Each episode spans minutes of bimanual manipulation and navigation, with a single policy expected to generalize across diverse activities such as turning on a radio, cooking a hotdog, or tidying a room.

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

![Per-task scores](media/per_task_and_eps_score.png)
*Per-task and per-episode scores. Green = success; red = failure.*

## Recovery from cross-task learning

Training on all 50 tasks leads to **emergent recovery behaviors**. Single-task models never recover from mistakes; the multi-task model learns to pick up fallen items and retry.

**Example 1:**

<table><tr>
<td><iframe width="320" height="180" src="https://www.youtube.com/embed/uAjF1_p9kJc" frameborder="0" allowfullscreen></iframe><br><small>Before: fails to recover</small></td>
<td><iframe width="320" height="180" src="https://www.youtube.com/embed/2Xi3uqARchw" frameborder="0" allowfullscreen></iframe><br><small>After: robust recovery</small></td>
</tr></table>

**Example 2:**

<table><tr>
<td><iframe width="320" height="180" src="https://www.youtube.com/embed/VSazcAkIEGI" frameborder="0" allowfullscreen></iframe><br><small>Before: error cascades to failure</small></td>
<td><iframe width="320" height="180" src="https://www.youtube.com/embed/i6cF_g20njg" frameborder="0" allowfullscreen></iframe><br><small>After: recovers and completes task</small></td>
</tr></table>

## Links

- Code: [behavior-1k-solution](https://github.com/IliaLarchenko/behavior-1k-solution)
- Checkpoints: [🤗 Hugging Face](https://huggingface.co/IliaLarchenko/behavior_submission)
