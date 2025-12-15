---
layout: default
title: Easy VLA
description: What it takes for a VLM to control the robot
---

## TL;DR
- Small VLM + simple action tokens is enough for LIBERO success.
- Everything is open: code, model, config, and eval video.

## Introduction
Teaching robots to manipulate objects remains a challenging problem. Recent work asks vision-language models to generate action numbers the same way they generate text. The VLA-0 paper by Goyal et al. [1] took Qwen2-VL-3B with almost no architectural changes and hit 94.7% on LIBERO—beating π₀.₅-KI, OpenVLA-OFT, and SmolVLA. No special heads or tokenization; just predicting actions as tokens.

This raises a question: which design choices actually matter? VLA-0 trained 100k steps at 5×10⁻⁶, used system prompts, and ensembles. Which of these are critical versus incidental?

## What We Did
We ran systematic ablations on PushT to isolate what matters for training VLAs. We tested:
- Learning rate sensitivity: 5×10⁻⁶, 1×10⁻⁵, 5×10⁻⁵, 1×10⁻⁴.
- Vision encoder fine-tuning vs. freezing.
- State/action representations: absolute vs. relative; with/without state tokens.
- System prompts: whether structured prompts help after fine-tuning.

We made two pragmatic choices:
- Use SmolVLM2 (sub-billion params) to keep experiments on consumer GPUs.
- Start with PushT for faster iteration and clearer signal than full LIBERO.

## Method

### Model Architecture
- Base: SmolVLM2 (vision encoder + connector + language model).
- Actions treated as discrete tokens (512 bins for actions and states).
- Model autoregressively predicts action sequences as if they were text.

### Action and State Representation
- Actions in PushT are 2D continuous end-effector positions, discretized into 512 bins over [-1, 1].
- Absolute actions: direct positions in workspace coordinates.
- Relative actions: deltas = actions - current_state; deltas discretized and added back during inference.
- State vector (end-effector + T-block pose) optionally added to the prompt as space-separated integers.

### Grammar Enforcement
Small datasets can yield the wrong token count. We enforce output grammar (format and sequence length) to block invalid trajectories.

### Training Setup
We use the LeRobot framework. Base configuration:

```json
{
  "batch_size": 16,
  "training_steps": 30000,
  "optimizer": "AdamW",
  "betas": [0.9, 0.95],
  "weight_decay": 0.01,
  "epsilon": 1e-8,
  "warmup_steps": 1000,
  "gradient_clip_norm": 1.0,
  "chunk_size_H": 10,
  "action_steps": 5,
  "observation_steps": 1
}
```

The chunk size of 10 means we predict 10 future actions per step but execute only the first 5 during rollout, using a single observation frame.

During training, inputs are formatted as a user/assistant conversation: the user message has images plus the task (and optionally state); the assistant replies with space-separated action integers. Loss is cross-entropy on action tokens only. For prompt experiments, we prepend the VLA-0 system prompt:

"Analyze the input image and predict robot actions for the next H timesteps. Each action has D dimensions. Output a single sequence of H×D integers (0 - B each), representing the H timesteps sequentially. Provide only space-separated numbers. Nothing else."

where H=10, D=2, and B=512.


### Evaluation
We evaluate all models purely on success rate—the percentage of episodes where the T-block reaches the target pose within tolerance. We do not compare models based on training loss, as we found that loss does not always correlate with downstream task performance. Each evaluation runs 64 episodes and we report the success rate at the final checkpoint unless otherwise specified.

### Results
#### Task: PushT
PushT is a 2D robotic manipulation task where an agent must push a T-shaped block to match a target pose (position and orientation). The task is performed in a planar workspace, making it simpler than 3D manipulation while still requiring the agent to learn coordinated pushing motions. The agent observes the scene through RGB images and controls a 2D end-effector position. Success is measured by whether the T-block's final pose matches the target within a specified tolerance.

![PushT demo](/assets/easy-vla/687474703a2f2f72656d69636164656e652e636f6d2f6173736574732f6769662f70757368745f646966667573696f6e2e676966.gif)


#### Task: Libero
### Fails

<div style="display: flex; gap: 12px; flex-wrap: wrap; align-items: flex-start;">
  <figure style="flex: 1 1 320px; margin: 0;">
    <video controls playsinline muted loop style="width: 100%; height: auto;">
      <source src="/assets/easy-vla/eval_episode_0.mp4" type="video/mp4" />
      Your browser does not support the video tag.
    </video>
    <figcaption style="margin-top: 6px;">Episode 0 (failure)</figcaption>
  </figure>
  <figure style="flex: 1 1 320px; margin: 0;">
    <video controls playsinline muted loop style="width: 100%; height: auto;">
      <source src="/assets/easy-vla/eval_episode_9.mp4" type="video/mp4" />
      Your browser does not support the video tag.
    </video>
    <figcaption style="margin-top: 6px;">Episode 9 (failure)</figcaption>
  </figure>
</div>
