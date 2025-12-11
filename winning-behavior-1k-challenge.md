---
layout: default
title: Winning the BEHAVIOR-1K Challenge
---

The Robot Learning Collective took **1st place** in the 2025 **BEHAVIOR-1K Challenge**, a large-scale benchmark of 50 long-horizon household tasks in photo-realistic simulation. Each episode spans minutes of bimanual manipulation and navigation, with a single policy expected to generalize across diverse activities such as turning on a radio, cooking a hotdog, or tidying a room.

## Challenge

50 tasks etc.

## Task Examples

<table>
<tr>
<td><video width="320" height="180" controls><source src="https://pub-a5638afed52c4226aac6a1e71ecc323c.r2.dev/behavior_report/episode_00010120_fast.mp4" type="video/mp4"></video><br><small><b>Task 1: Picking Up Trash</b><br>Put the three cans of soda from the living room inside the trash can in the kitchen.</small></td>
<td><video width="320" height="180" controls><source src="https://pub-a5638afed52c4226aac6a1e71ecc323c.r2.dev/behavior_report/episode_00120030_fast.mp4" type="video/mp4"></video><br><small><b>Task 12: Preparing Lunch Box</b><br>Put apple halves, sandwich, and cookie into the packing box, then add tea from the fridge.</small></td>
</tr>
<tr>
<td><video width="320" height="180" controls><source src="https://pub-a5638afed52c4226aac6a1e71ecc323c.r2.dev/behavior_report/episode_00160020_fast.mp4" type="video/mp4"></video><br><small><b>Task 16: Moving Boxes To Storage</b><br>Move two storage containers from living room to garage and stack them.</small></td>
<td><video width="320" height="180" controls><source src="https://pub-a5638afed52c4226aac6a1e71ecc323c.r2.dev/behavior_report/episode_00320050_fast.mp4" type="video/mp4"></video><br><small><b>Task 32: Wash A Baseball Cap</b><br>Wash two baseball caps on the countertop using the washer until clean.</small></td>
</tr>
<tr>
<td><video width="320" height="180" controls><source src="https://pub-a5638afed52c4226aac6a1e71ecc323c.r2.dev/behavior_report/episode_00380020_fast.mp4" type="video/mp4"></video><br><small><b>Task 38: Spraying For Bugs</b><br>Pick up pesticide atomizer and spray both potted plants in the garden.</small></td>
<td><video width="320" height="180" controls><source src="https://pub-a5638afed52c4226aac6a1e71ecc323c.r2.dev/behavior_report/episode_00420030_fast.mp4" type="video/mp4"></video><br><small><b>Task 42: Chop An Onion</b><br>Dice the onion on the chopping board, put it in the bowl, then wash the knife and board.</small></td>
</tr>
</table>

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
<td><video width="320" height="180" controls><source src="https://pub-a5638afed52c4226aac6a1e71ecc323c.r2.dev/behavior_report/recovery_picture_early_ex1.mp4" type="video/mp4"></video><br><small>Before: fails to recover</small></td>
<td><video width="320" height="180" controls><source src="https://pub-a5638afed52c4226aac6a1e71ecc323c.r2.dev/behavior_report/recovery_picture_late_ex1.mp4" type="video/mp4"></video><br><small>After: robust recovery</small></td>
</tr></table>

**Example 2:**

<table><tr>
<td><video width="320" height="180" controls><source src="https://pub-a5638afed52c4226aac6a1e71ecc323c.r2.dev/behavior_report/recovery_picture_early_ex2.mp4" type="video/mp4"></video><br><small>Before: error cascades to failure</small></td>
<td><video width="320" height="180" controls><source src="https://pub-a5638afed52c4226aac6a1e71ecc323c.r2.dev/behavior_report/recovery_picture_late_ex2.mp4" type="video/mp4"></video><br><small>After: recovers and completes task</small></td>
</tr></table>

## Links

- Code: [behavior-1k-solution](https://github.com/IliaLarchenko/behavior-1k-solution)
- Checkpoints: [🤗 Hugging Face](https://huggingface.co/IliaLarchenko/behavior_submission)
