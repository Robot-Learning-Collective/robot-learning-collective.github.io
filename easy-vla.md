---
layout: default
title: Easy VLA
description: What it takes for a VLM to control the robot
header_links:
  - text: View on GitHub
    url: https://github.com/Robot-Learning-Collective/lerobot-experiments
  - text: Model weights
    url: https://huggingface.co/zaringleb/pusht_smolandfast_20k
---

# TL;DR
We present Easy-VLA: compact (500M) and simple VLA models with comaprable performance to SOTA models on simulation benchmarks
- Light weighted VLM (SmolVLM) + actions-as-text tokens is enough for robot success in simulation.
- Everything is open: code, model, config, and eval results.

[-> Jump to model training](#what-we-did) or continue to understand our motivation and research path.

# Motivation
While working with existing **VLA models**, we noticed that they **are often overly complex** for many tasks. This complexity appears both in feature design and in model scale. While such complexity is required and intended for some problems, for many others, especially within the learning community, it is clear overkill. For example, using a 3B model to pick up a toy cube.

In terms of scale, we already know that models as small as 50M parameters, such as ACT, can be surprisingly capable for simple tasks. Similarly, in terms of architectural complexity, models like Pi0 are not only hard to build on top of, but even community efforts to reproduce them often lead to errors.

This led us to a central question: how compact, easy to use, and simple can VLA models be while still retaining their core benefits:
* **VLM pretraining** on internet data that enables zero-shot generalisation (even if success rates are low) — see the [RT-2](https://robotics-transformer2.github.io/assets/rt2.pdf)
* **VLA pretraining** on robot data that significantly reduces the number of required demonstrations [TRI](https://toyotaresearchinstitute.github.io/lbm1/)
* **Language instructions** that provide a flexible interface for multi-task settings.

To address this question, we are building Easy-VLA. This release presents our first **milestone: achieving parity with the best available VLA models on simulation benchmarks**, while reducing both complexity and scale by an order of magnitude.

Our second motivation relates to our view on how **open-source projects** in robotics and AI should be developed. We believe that impactful open-source projects should not stop at releasing weights and inference code. They should include the full training pipeline, exact configs to reproduce results on concrete tasks, and honest notes about what failed and why certain directions were chosen over others.

This level of openness makes it much easier for others to understand what was done, why it was done, and how to build on top of it.

That is why **we release everything**: model weights, code, training pipelines, configuration files, Weights and Biases (wandb, an experiment tracking tool) runs, failure analysis, and even internal documentation. The goal is for anyone to be able to follow our thinking step by step.

## Intro

<details markdown="1">
  <summary style="font-size:1.rem; font-weight:700; cursor:pointer;">What is VLA? <em>(click to expand)</em></summary>

  **Vision-Language-Action models** are, put simply, an attempt to enable something like ChatGPT to control a robot. In this sense, they represent a continuation of the expansion of Large Language Models into new domains.

  The first step in this expansion was vision. By encoding images into the same representation space as language token embeddings, we got Vision Language Models (VLMs). The next step was Action: enabling these models to output robot commands, such as a robo-arm pose or joint positions for the next time step. Adding this action output gives us VLA models.

  If you want more background on how AI and robotics gradually came together to reach this point, check our [previous post](https://robot-learning.notion.site/robot-learnig-onboarding).

  To understand what really changed in the VLA world in 2025, we suggest watching two talks. One looks closely at how **pretraining affects action models**. The other tells the story of Physical Intelligence, one of the leading companies pushing **large-scale models for robotics**.
  <div style="display:flex; gap:1rem; flex-wrap:wrap; margin:1rem 0; justify-content:center;">
    <a href="https://www.youtube.com/watch?v=TN1M6vg4CsQ" title="Pretraining effects (opens on YouTube)" style="flex:1 1 320px; max-width:520px; text-decoration:none; border:0;">
      <img src="https://img.youtube.com/vi/TN1M6vg4CsQ/hqdefault.jpg" alt="Pretraining effects video" style="width:100%; height:240px; object-fit:cover; display:block; border:0; border-radius:4px;">
    </a>
    <iframe title="Physical Intelligence story" src="https://www.youtube.com/embed/a8-QsBHoH94" style="flex:1 1 320px; height:240px; border:0; max-width:520px; border-radius:4px;" allowfullscreen></iframe>
  </div>
</details>

### VLA architecture and choice of action space

What is the best way to close the gap between industry and open-source? It is to take the best available model, adapt it to a smaller scale, and remove secondary features. So, naturally, our first choice was Pi0.5 from [Physical Intelligence](https://www.physicalintelligence.company/).

We were especially inspired by the line of work culminating in [Knowledge Insulation](https://www.physicalintelligence.company/research/knowledge_insulation), which shows that quantised (rounded into discrete bins) actions treated as an LLM’s native tokens (as opposed to a diffusion or flow matching "expert") preserve knowledge acquired from VLM pretraining on internet data much better. As a result, the model is much better at zero-shot (no task-specific training) and language steerability (following language instructions reliably).

This direction also feels especially promising because it leads to deeper integration with mainstream AI development. The model is not only "adapted" to robotics needs (as was done, for example, using a ResNet in ACT, or building VLA models on top of VLMs), but can co-develop with them. That is, adding robotics data does not "break" the VLM, and may even improve its capabilities in the visual domain and intuitive physics.

So our first choice was to train a quantised compact VLA with [FAST](https://www.physicalintelligence.company/research/fast) action tokens.

**FAST action tokens.** Robot actions are time series with high correlation, which is bad for token learning. FAST proposes that instead of quantising actions for each DoF (degree of freedom, a.k.a. robot joint) at each timestamp (the robot predicts around ~1 second into the future, for example 30 steps), we represent each action chunk differently:
- Each action is treated as a signal and processed with Discrete Cosine Transform (DCT), (you can think of it as similar to a Fourier transform) to obtain coefficients representing leading frequency amplitudes.
- These coefficients are rounded (quantised). This is the only lossy step.
- The rounded coefficients are represented as symbols and written as a string combining all DoFs. This string represents all actions in the chunk.
- Then the string is encoded via byte-pair encoding (BPE) in the exactly same way individual letters are combined into tokens for LLMs.
- The resulting tokens (both the ones representing initial coefficients and their frequent combinations) overwrite N least-frequent LLM tokens.
- The reverse procedure, from tokens predicted by the LLM back to robot actions, is the exact opposite.

This sounds good, but our experience with FAST tokens was not great. In a low-data regime (typical for the learning community), the FAST scheme struggles to provide enough data points per token for the model to learn which tokens are appropriate to use. This is made worse by the fact that these tokens lose semantic meaning, so a single mistake can lead to a catastrophic decoding error. In our intended use case, we would be okay with poor model precision, but we need it to be safe for a robot to use.

To illustrate the problem, see the histogram of FAST tokens on the PushT dataset (using a pretrained "universal" tokenizer). Or even more fundamentally, look at the histogram of Fourier coefficients on the same dataset: the majority of values lie around zero, which means uniform bins in this space do not get enough data points per coefficient for any meaningful precision.

<div style="display:flex; flex-wrap:wrap; gap:1rem; justify-content:center; align-items:flex-start;">
  <figure style="flex:1 1 320px; max-width:520px; margin:0;">
    <img src="/assets/easy-vla/token_freq_physical-intelligence-fast.png" alt="FAST token frequency on PushT" style="width:100%; height:320px; object-fit:contain; background:#f8f8f8; padding:8px; border-radius:4px;">
    <figcaption style="margin-top:0.5rem; text-align:center;">FAST token frequency on PushT (pretrained universal tokenizer).</figcaption>
  </figure>
  <figure style="flex:1 1 320px; max-width:520px; margin:0;">
    <img src="/assets/easy-vla/dct_tokens_bar.png" alt="Histogram of DCT coefficients" style="width:100%; height:320px; object-fit:contain; background:#f8f8f8; padding:8px; border-radius:4px;">
    <figcaption style="margin-top:0.5rem; text-align:center;">Histogram of DCT coefficients showing most mass near zero.</figcaption>
  </figure>
</div>

Additionally, we think both DCT and BWT try to solve the same correlation-across-time problem (although in different spaces), and therefore may be redundant. Also, the whole time-correlation problem is most prominent with high-frequency control, which does not make that much sense for cheap/open-source hardware solutions.

So we decided to further research tokenisation schemes. We tried three common choices: uniform bins, learned quantised encodings, and representing actions as text native to LLMs ([VLA-0](https://arxiv.org/abs/2510.13054)). We achieved comparable results with all of them, but picked the last one as the simplest and most native-to-LLM choice.

The next step was to adapt this design for a small model, try to reproduce the same results, and find which design choices are crucial.


# Design choices oblation and finding best params for VLA-0

## What We Did
We ran systematic ablations on PushT to isolate what matters for training VLAs. We tested:
- Learning rate sensitivity: 5×10⁻⁶, 1×10⁻⁵, 5×10⁻⁵, 1×10⁻⁴.
- Vision encoder fine-tuning vs. freezing.
- State/action representations: absolute vs. relative; with/without state tokens.
- System prompts: whether structured prompts help after fine-tuning.

We made two pragmatic choices:
- Use SmolVLM2 (sub-billion params) to keep experiments on consumer GPUs.
- Start with PushT for faster iteration and clearer signal than full LIBERO.

# Method

## Model Architecture
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


## Evaluation
We evaluate all models purely on success rate—the percentage of episodes where the T-block reaches the target pose within tolerance. We do not compare models based on training loss, as we found that loss does not always correlate with downstream task performance. Each evaluation runs 64 episodes and we report the success rate at the final checkpoint unless otherwise specified.

## Results
### Task: PushT
<div style="display:flex; flex-wrap:wrap; gap:1rem; align-items:flex-start;">
  <div style="flex:1 1 320px; min-width:280px; max-width:640px;" markdown="1">
**PushT** is a 2D robotic manipulation task where an agent must push a T-shaped block to match a target pose (position and orientation). The task is performed in a planar workspace, making it simpler than 3D manipulation while still requiring the agent to learn coordinated pushing motions. The agent observes the scene through RGB images and controls a 2D end-effector position. Success is measured by whether the T-block's final pose matches the target within a specified tolerance.
  </div>
  <div style="flex:1 1 320px; min-width:280px; max-width:480px; text-align:center;">
    <img src="/assets/easy-vla/687474703a2f2f72656d69636164656e652e636f6d2f6173736574732f6769662f70757368745f646966667573696f6e2e676966.gif" alt="PushT demo" style="width:100%; height:auto; border-radius:4px;">
  </div>
</div>

### What We Learned
...

### Task: Libero

<div style="display:flex; flex-wrap:wrap; gap:1rem; align-items:flex-start;">
  <div style="flex:1 1 320px; min-width:280px; max-width:640px;" markdown="1">
LIBERO (Lifelong Robot Learning Benchmark) is a 3D robotic manipulation suite designed to evaluate how agents transfer and retain skills across a wide range of tasks. The agent operates a 7-DOF robotic arm in kitchen or tabletop environments, observing the scene through RGB images from both a workspace and a wrist-mounted camera. Success is measured by the agent's ability to complete long-horizon tasks and its performance on previously learned tasks after acquiring new ones. Following the methodology of the VLA0 paper, we compare our model across four task suites:

- `Libero_object`: Varies the types of objects manipulated to test generalizable object recognition (e.g., "pick up the ketchup/milk/juice and put it in the basket").

- `Libero_spatial`: Focuses on spatial relationships and layouts (e.g., "put the bowl on the plate" with varying object positions).

- `Libero_goal`: Keeps objects the same but changes the end goal (e.g., "open the drawer" vs. "put the mug in the drawer").

- `Libero_10`: Focuses on long-horizon and procedural sequences (e.g., "open the microwave and put the bowl in it"), requiring the agent to chain multiple behaviors together to complete complex, multi-step goals.

</div>
  <div style="flex:1 1 320px; min-width:280px; max-width:480px; text-align:center;">
    <img src="/assets/easy-vla/687474703a2f2f72656d69636164656e652e636f6d2f6173736574732f6769662f70757368745f646966667573696f6e2e676966.gif" alt="Libero demo" style="width:100%; height:auto; border-radius:4px;">
  </div>
</div>

### Results

For training on LIBERO, we used the optimal learning rate determined by our best results on the PushT task. We trained for 60 epochs with a batch size of 192. Additionally, we applied image cropping and action masking. To optimize performance, we trained the model in mixed precision using the bf16 format.

**_Reproducibility:_** : To reproduce our results, you can use the vla0-libero-final.yaml configuration file.

We opted not to use relative positions; while an ablation study for the PushT task showed they provided a significant boost in success rate, they were difficult to apply to the LIBERO input data. Furthermore, while ensembling did not improve performance on PushT, we achieved our best metrics on LIBERO by ensembling across 8 tokens ahead.

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

# Discussions

## Reflection and future steps
....
\[Draft\] Does reimplementing all features from VLA-0 undermined our initial goal of obtaining minimal model for future open-source development? No, 1. our model is much smoller, 2. we know relative importance of main design choises and will simplify it mooving into real robots with denoising experts.

---

# Team

<style>
.team-grid { display: flex; justify-content: center; gap: 3rem; flex-wrap: wrap; margin: 2rem 0; }
.team-member { text-align: center; }
.team-member img { width: 120px; height: 120px; border-radius: 50%; object-fit: cover; border: 3px solid #ddd; }
.team-member p { margin-top: 0.75rem; font-weight: 500; color: #333; }
</style>

<div class="team-grid">
  <div class="team-member">
    <img src="assets/easy-vla/1650104310604.jpeg" alt="Oleg Balakhnov">
    <p><a href="https://www.linkedin.com/in/oleg-balakhnov-a591a1233/">Oleg Balakhnov</a></p>
  </div>
  <div class="team-member">
    <img src="assets/easy-vla/1740353302751.jpeg" alt="Sergei Skvortsov">
    <p><a href="https://www.linkedin.com/in/sergei-skvortsov/">Sergei Skvortsov</a></p>
  </div>
  <div class="team-member">
    <img src="assets/easy-vla/1736204199843.jpeg" alt="Gleb Zarin">
    <p><a href="https://www.linkedin.com/in/zaringleb/">Gleb Zarin</a></p>
  </div>
</div>
