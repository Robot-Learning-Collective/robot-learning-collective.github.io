---
layout: default
title: VLA-0-Smol
description: A Reproducible Recipe for High-Performance, Sub-Billion Parameter VLAs
header_links:
  - text: View on GitHub
    url: https://github.com/Robot-Learning-Collective/lerobot-experiments/tree/balakhnov/rebase_vla0
  - text: Model weights
    url: https://huggingface.co/olegbalakhnov/libero_vla0_final
---

# VLA-0-Smol: A Reproducible Recipe for High-Performance, Sub-Billion Parameter VLAs
We present a compact VLA model (500M params) that achieves a high score on simulation benchmarks with a clear recipe to reproduce results and a detailed ablation study showing the impact of key design choices.

# Intro

Growing interest in Vision Language Action models for robotics applications currently faces a very high barrier to entry due to their complexity. This is especially hard for the learning community, where the requirement to rent cloud GPUs and set up distributed training poses an additional hurdle on top of existing challenges such as robot hardware, data collection, and model training. As a result, we observed an unfortunate trend where users obtain significantly worse results compared to much more basic models like ACT, which at least exhibit predictive spatial generalisation.

To address this issue, we decided to implement a VLA model based on a compact Vision Language Model that can be fine tuned and run for inference on a consumer grade GPU. In particular, our requirement to keep the model below 1B parameters left us with an almost unchallenged choice of SmolVLM2. In a sense, we are following in the steps of SmolVLA, but with a different set of design choices.

Another gap we aimed to address is the lack of clear recipes for reproducing claimed results. Combined with the unclear impact of various design choices, this creates a situation that is difficult to untangle.

To this end, we implemented one of the most straightforward action discretisation mechanisms from VLA 0 and conducted a careful study of key design choices on established simulation benchmarks. Our resulting model achieved one of the highest scores on the LIBERO benchmark, which mostly reflects current limitations of simulation benchmarks rather than the unique capabilities of our model. We plan to address this issue and adapt our model to real world problems in future work.


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

This chapter describes the key design components of VLA-0 and highlights the specific modifications we introduced in our implementation. Rather than re-deriving the full method, we focus on the architectural choices, input representations, training augmentations, and system-level considerations that are most relevant for understanding how our setup differs from the original approach. These design decisions provide the context needed to interpret the ablation results and performance analyses presented in the following sections.

## Model Architecture
The VLA-0 authors use `Qwen-VL-2.5-3B` as the vision–language backbone. In contrast, we chose a smaller model to enable inference on a laptop equipped with a consumer-grade GPU, with the goal of eventually evaluating the system on a real robot. For this reason, we focused on sub-billion-parameter models and selected `SmolVLM2-500M`. This model offers solid documentation and an accompanying paper that clearly explains its design, and it has been evaluated for robotic applications in the SmolVLA work.

We use the standard `SmolVLM2` implementation from the Transformers library with one key modification. In the original setup, the authors employ an image tiling strategy in which both the original image and an upscaled version split into 16 tiles are passed to the image encoder. While effective, this approach significantly increases computational cost and is impractical for real-time robotic applications. Instead, we pass only the original image to the image encoder. This simplification matches the approach used in SmolVLA and makes the model more suitable for deployment on resource-constrained robotic systems.

### Model Input
We largely follow the approach of the original paper, with minor modifications to accommodate the `SmolVLM2` chat template.

Specifically, we construct a chat-style prompt that includes the image observation along with task description, explicit state and action tokens.

When used, the state is discretized into N bins and appended to the input as a sequence of integer tokens represented in text form. Similarly, a chunk of actions is discretized into M bins, flattened into a time-ordered sequence, and provided to the model also as text.

The original authors use a system prompt to describe the task and guide the model’s behaviour. Since it was unclear how critical this component is, we ablated it by evaluating the model both with and without the system prompt.

### Masked Action Augmentation

As part of our training pipeline, we use masked action augmentation, a data augmentation technique in which random characters in the target action string are masked during training.

Previous work on VLA-0 demonstrated that action masking significantly improves success rates on the Libero benchmark. We adopted this technique with a modification: instead of masking individual characters, we mask the entire action value. This augmentation aims to mitigate the model's reliance on auto-completing numerical sequences, thereby enforcing stronger dependence on visual observations. We hypothesized that character-level masking might still allow for implicit completion; whole-action masking eliminates this possibility.

### Grammar Enforcing

Although the model generally converges on the correct output format, the limited size of usual robotics datasets can occasionally cause the policy to generate an incorrect number of tokens, leading to trajectory failure. To mitigate this, we implemented grammar constraints that strictly enforce the output format and sequence length, preventing the generation of invalid tokens.

### Performance

We discovered that our model is primarily CPU bound. This means the overhead of placing a CUDA kernel call takes more time than the actual execution on the GPU. Model compilation is typically the best solution for this; however, we encountered compatibility issues when using it within the LeRobot framework alongside the Accelerate library. 

[picture from profiler]

- **Mixed Precision:** We found that training in `float16` leads to gradient explosions. Switching to `bfoat16` completely resolved this issue.
- **Loss Calculation:** Calculating loss in mixed precision resulted in slight performance degradation. To fix this, we convert our logits to full precision before computing the loss. This approach provides nearly the same accuracy as full-precision training but is almost twice as fast.
- **FlashAttention:** We chose not to use FlashAttention as it provides no performance boost. This is due to our relatively small sequence lengths (usually under 512 tokens), where standard attention implementation is already highly efficient.

Having outlined the core components of VLA-0 and the modifications introduced in our implementation, we now turn to a systematic evaluation of these design choices. In the next chapter, we use controlled ablation studies to quantify how individual decisions—such as optimization settings and model freezing—affect performance. These experiments provide empirical grounding for the design choices discussed above and clarify which components are most critical for successful learning.

## Ablations

<div style="display:flex; flex-wrap:wrap; gap:1rem; align-items:flex-start;">
  <div style="flex:1 1 320px; min-width:280px; max-width:640px;" markdown="1">
For ablations we were using PushT task. The task is performed in a 2D space, making it simpler than 3D manipulation while still requiring the agent to learn accurate pushing motions. The agent observes the scene through RGB images and controls a 2D end-effector position. Success is measured by whether the T-block’s final pose matches the target within a specified tolerance.
  </div>
  <div style="flex:1 1 320px; min-width:280px; max-width:480px; text-align:center;">
    <img src="/assets/vla-0-smol/687474703a2f2f72656d69636164656e652e636f6d2f6173736574732f6769662f70757368745f646966667573696f6e2e676966.gif" alt="PushT demo" style="width:100%; height:auto; border-radius:4px;">
  </div>
</div>

We evaluate all models purely on success rate: the percentage of episodes where the T-block reaches the target pose within tolerance. Each evaluation runs 64 episodes and we report the success rate at the best checkpoint unless otherwise specified.

For completeness and reproducibility, we provide the full set of hyperparameters and training details for every experiment in our public Weights & Biases workspace ([wandb](https://wandb.ai/sergeyskv/rlc_public?nw=nwusersergeyskv)).

### Learning Rate

Learning rate is often treated as a hyperparameter to tune, but its importance for VLA fine-tuning was unclear. We tested four learning rates `5×10⁻⁶`, `1×10⁻⁵`, `5×10⁻⁵`, and `1×10⁻⁴`, keeping all other parameters constant (relative actions, no state, full model training).

The results show that learning rate has a dramatic impact on performance. The two smallest learning rates (5×10⁻⁶ and 1×10⁻⁵) completely failed to learn, achieving 0% success rate throughout training. This suggests that VLA-0's choice of 5×10⁻⁶ may have worked for their 100,000 training steps but is too conservative for our 30,000 step training regime.

At 5×10⁻⁵, we observed the best performance with a final success rate of 57.8%. The highest learning rate (1×10⁻⁴) showed slightly worse performance, suggesting we may be approaching the upper bound of stable learning rates for this task.

### Vision Encoder Freezing

A common question in fine-tuning large models is whether you can get away with freezing some components. For VLMs, the vision encoder is an obvious candidate—it's large, and freezing it would speed up training and reduce memory usage. However, vision encoders in VLMs are typically trained on natural images, not robotics scenes, so it's unclear if their frozen features are sufficient for manipulation tasks.

We compared training the full model (vision encoder + connector + language model) against freezing the vision encoder and only training the connector and language model.

**Freezing the vision encoder drops success rate from 57.8% to 25%** — a 32 percentage point decrease. This suggests that the features learned by pre-trained vision encoders on natural images are not directly suitable for robotic manipulation, and the model benefits significantly from adapting the visual representations during fine-tuning.

This result is somewhat disappointing from a practical standpoint. We had hoped that frozen vision encoders would be sufficient, which would make training faster and potentially preserve performance on other vision-language tasks. However, the data clearly shows that vision encoder fine-tuning is necessary for good performance on manipulation tasks.

### State and Action Representations

The robotics literature shows considerable variation in how actions and states are represented. We tested four combinations to understand their individual and combined effects:

| Configuration	| State	| Actions	| Success Rate |
| :--- | :--- | :--- | :--- |
| Baseline	| No	| Absolute	| 29.7% |
| + Relative actions	| No	| Relative	| 57.8% |
| + State	| Yes	| Absolute	| 45.3% |
| Full	| Yes	| Relative |	70.3% |

The results show clear trends:

- **Relative vs. Absolute Actions**: Switching from absolute to relative actions provided the largest single improvement. This aligns with common practice in robot learning—relative actions are often easier to learn because they're invariant to the absolute position in the workspace and typically have smaller magnitudes.
- **Adding State Information**: Including the discretized state vector in the prompt also improved performance, though the effect was less dramatic than the action representation choice. This is somewhat surprising—we expected the model might be able to infer the relevant state information from the image alone. The improvement suggests that explicit state information helps the model, even when that information should theoretically be extractable from visual observations.
- **Combined Effect**: The best performance came from using both relative actions and state information, suggesting these design choices are complementary rather than redundant.


### System Prompt

VLA-0 uses a structured system prompt that explicitly describes the output format. We tested whether this prompt provides any benefit after fine-tuning, since one might expect that during fine-tuning, the model learns the desired output format from the data itself.

We compared training with and without the system prompt, keeping all other parameters constant. **The results show no difference in performance**—both conditions achieved the same success rate. This suggests that after fine-tuning on task-specific data, the system prompt becomes redundant. The model learns the output format from the training examples themselves.

### Masked Action Augmentation

Adding action masking improved the success rate from **57.8% to 78.1%**. This significant jump validates the hypothesis we set out to test: that preventing the model from relying on character-level auto-completion results in a stronger policy. It also confirms that the benefits of action masking, originally reported by the VLA-0 authors, transfer effectively to the smaller SmolVLM2 architecture and our specific whole-action masking strategy.

### Ensemble Prediction

VLA0 employs temporal ensembling across $N$ tokens to smooth out action sequences. For the PushT task, we experimented with a standard $n$-tokens ahead approach, where we predicted and executed a fixed chunk of $N=5$ tokens.

When comparing these two methods, we found that temporal ensembling yielded no significant performance boost for PushT. Therefore, to optimize computational efficiency and inference speed, we opted for the 5-tokens ahead approach.

### What We Learned

Our ablation study on PushT reveals a clear hierarchy of importance for VLA training decisions:

**Critical choices** (large impact on performance):

- **Learning rate**: Getting this wrong means complete failure (0% success) or suboptimal performance. For 30,000 training steps, 5×10⁻⁵ worked best, but this likely needs adjustment for different training budgets.
- **Vision encoder fine-tuning**: Freezing the vision encoder cuts performance by more than half. Pre-trained visual features from natural images aren't sufficient for manipulation tasks.
- **Masked Action Augmentation:** This proved to be a major performance driver, boosting success rates from 57.8% to 78.1%. Preventing the model from relying on numerical pattern matching significantly improves policy robustness.
- **Action representation**: Relative actions substantially outperform absolute actions, consistent with broader robotics literature.

**Helpful but secondary**:

- **State information**: Provides a meaningful boost, even though the information should be present in the images. The explicit state signal seems to help the model learn faster or more reliably.

**Doesn't matter**:

- **System prompts**: After fine-tuning, the model learns output formats from data. Carefully crafted prompts add no value during training.
- **Temporal Ensembling:** Temporal ensembling did not improve performance on PushT, suggesting it is unnecessary for short-horizon, planar manipulation tasks.


## LIBERO

<div style="display:flex; flex-wrap:wrap; gap:1rem; align-items:flex-start;">
  <div style="flex:1 1 320px; min-width:280px; max-width:640px;" markdown="1">
Having identified the most impactful design choices on PushT, we next ask whether these conclusions transfer to a substantially more challenging setting. While PushT enables rapid iteration, it lacks the visual complexity, long-horizon structure, and embodiment constraints of real-world manipulation. To test the robustness of our findings, we applied the best-performing configuration from our ablation study directly to **LIBERO**, a standard benchmark for evaluating vision–language–action models in 3D environments.
  </div>
  <div style="flex:1 1 320px; min-width:280px; max-width:480px; text-align:center;">
    <img src="/assets/vla-0-smol/media_videos_eval_video_100000_90766c2977b1eb305cd3.gif" alt="PushT demo" style="width:100%; height:auto; border-radius:4px;">
  </div>
</div>


**LIBERO** is a 3D robotic manipulation suite designed to evaluate how agents transfer and retain skills across a diverse range of tasks. In this environment, an agent operates a 7-DOF robotic arm within kitchen or tabletop settings, observing the scene via RGB images from both a workspace camera and a wrist-mounted camera.

Success is measured by the agent's ability to complete long-horizon tasks and maintain its performance on previously learned tasks after acquiring new ones. Following the VLA0 paper, we compared our model across four specific task suites:

- **Object:** Tests generalizable object recognition by varying the types of objects manipulated (e.g., "pick up the [ketchup/milk/juice] and put it in the basket").
- **Spatial:** Focuses on spatial relationships and layouts (e.g., "put the bowl on the plate" with varying initial object positions).
- **Goal:** Maintains the same object set but varies the end goal (e.g., "open the drawer" vs. "put the mug in the drawer").
- **Long:** Emphasizes long-horizon, procedural sequences (e.g., "open the microwave and put the bowl in it"). This requires the agent to chain multiple behaviors together to complete complex, multi-step goals.


### Training Setup

Our LIBERO experiments largely follow the configuration identified as optimal on PushT. We used the same learning rate, enabled action masking, and applied image cropping. The model was trained for **100,000 steps** (approximately 70 epochs) with a **batch size of 192**, using mixed-precision training in **bf16**.

One notable deviation concerns **action representation**. Although relative actions provided a significant performance boost on PushT, we found them difficult to apply reliably to LIBERO due to differences in input structure and task dynamics. As a result, all LIBERO experiments were conducted using absolute actions.

### Results

We evaluated the final model using two inference-time protocols. Following VLA-0, we ensemble predictions across **8 future action tokens**. For comparison, we also report results using a standard **n-tokens-ahead prediction** with n=8, matching the evaluation setup used in our PushT ablations.

Without ensembling, the average success rate on LIBERO drops to **90.7%**, a **3.4 percentage point decrease** relative to the ensembled evaluation. This differs from our PushT results, where temporal ensembling had no measurable impact on performance.

These results indicate that while temporal ensembling is unnecessary for PushT, it provides a clear benefit on LIBERO. In contrast, other high-impact design choices identified on PushT — such as learning rate, vision encoder fine-tuning, and action masking — transfer cleanly to the LIBERO setting.

| Model | Params | Object | Spatial | Goal | Long | Avg |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Diffusion Policy | 0.15B | 78.3 | 92.5 | 68.3 | 50.5 | 72.4 |
| pi0-FAST | 3B | 87.0 | 63.0 | 89.0 | 48.0 | 71.8 |
| SmolVLA | 0.25B | 87.0 | 93.0 | 88.0 | 63.0 | 82.8 |
| SmolVLA | 2.25B | 93.0 | 94.0 | 91.0 | 77.0 | 88.8 |
| OpenVLA-OFT | 7B | 94.3 | 95.2 | 91.7 | 86.5 | 91.9 |
| pi0.5 - KI | 3.3B | 96.6 | 97.2 | 94.6 | 85.8 | 93.3 |
| VLA0 | 3B | 97.0 | **97.8** | **96.2** | 87.6 | **94.7** |
| **VLA-0-Smol (Ours)** | **0.5B** | **97.2** | 92.2 | 95.6 | **91.2** | 94.1 |

### Analysis

A central question of this work is whether conclusions drawn from fast ablation on PushT transfer to a more complex benchmark. We find that most high-impact design choices identified on PushT generalize well to LIBERO. In particular, the learning rate, vision encoder fine-tuning, and masked action augmentation all remained necessary for strong performance when scaled to 3D, long-horizon tasks.
However, not all conclusions transferred directly. Temporal ensembling, which had no measurable effect on PushT, provided a clear improvement during LIBERO evaluation. This indicates that while PushT is effective for identifying many critical training decisions, it does not fully capture the requirements of long-horizon task execution.

Our analysis of the LIBERO results highlighted two primary areas of interest:

- **Overfitting.** During evaluation, we observed that the model performs near-perfectly on some tasks while failing on others that are visually and logically similar. This suggests that the model may be overfitting to specific task trajectories rather than learning fully generalizable behaviors.

<div style="display: flex; gap: 12px; flex-wrap: wrap; align-items: flex-start;">
  <figure style="flex: 1 1 320px; margin: 0;">
    <video controls playsinline muted loop style="width: 100%; height: auto;">
      <source src="/assets/vla-0-smol/eval_episode_0.mp4" type="video/mp4" />
      Your browser does not support the video tag.
    </video>
  </figure>
  <figure style="flex: 1 1 320px; margin: 0;">
    <video controls playsinline muted loop style="width: 100%; height: auto;">
      <source src="/assets/vla-0-smol/eval_episode_9.mp4" type="video/mp4" />
      Your browser does not support the video tag.
    </video>
  </figure>
</div>


- **Model size.** Our 0.5B parameter model achieved performance close to the 3B parameter model used in the original VLA-0 work. This result suggests that, for LIBERO, careful design and training choices can compensate for reduced model capacity. While larger models are typically expected to generalize better, our findings indicate diminishing returns from scale alone under this evaluation setup. We discuss broader implications of this observation in the Discussion section.

## Conclusion
TBD

## Discussions

### Future steps
TBD

### Performance

Since we are utilizing a very small model, our goal is to enable it to run locally on consumer devices and mid-range GPUs. Specifically, we aim to optimize the model for **real-time inference** on user-grade GPUs and hardware such as the **SO100 ARM** robotic platform.

Additionally, we believe that action prediction can be significantly accelerated by applying **speculative decoding**. Because inference is currently in a **memory-bound regime** and actions typically consist of only a few tokens followed by a space, these sequences can be efficiently predicted by smaller **draft models** to reduce latency.

### LIBERO

Recent research from [*LIBERO-Plus*](https://arxiv.org/abs/2510.13626) and [*LIBERO-PRO*](https://arxiv.org/abs/2510.03827) confirms what we suspected: high scores on LIBERO don't always mean the model is "smart". These papers found that many models, even huge ones, are often just memorizing the training data instead of truly understanding the task. When researchers made small changes—like moving objects slightly, changing the lighting, or even messing up the text instructions—the models often failed completely. In some cases, the models ignored the language instructions entirely and just repeated the exact same movement they learned during training.

This explains why our tiny 0.5B model was able to perform just as well as the big 3B model. It seems the standard LIBERO benchmark is mostly testing memory, not skill. Since even small models can memorize things well, the difference in model size didn't matter as much as we expected. We know this means our high scores might not fully prove our model is ready for the real world just yet. To fix this, we want to move beyond simulation and start experimenting on a real robot, where the "messiness" of the real world will act as the ultimate test of our model's true skills

## Acknowledgments
<a href="https://nebius.com/" target="_blank">
  <img src="assets/winning-behavior-1k-challenge/idOGJsi66K_logos.webp" alt="Nebius Logo" style="height: 32px; width: auto; margin: 10px 0;">
</a>

We would like to thank [Nebius](https://nebius.com) for providing the GPU resources that made this research possible. Their support was instrumental in training `VLA-0-Smol` and conducting the extensive evaluations across the LIBERO and PushT benchmarks.

## Citation information 

Please cite as:

```jsx
Balakhnov et al., "VLA-0-Smol: A Reproducible Recipe for High-Performance, Sub-Billion Parameter VLAs", Robot Learning Collective blog, 2025.
```

BibTeX citation:
```jsx
@article{balakhnov2025vla0smol,
  title={VLA-0-Smol: A Reproducible Recipe for High-Performance, Sub-Billion Parameter VLAs},
  author={Balakhnov, Oleg and Skvortsov, Sergei and Zarin, Gleb},
  year={2025},
  journal={Robot Learning Collective blog},
  note={}
}
```

# Team

<style>
.team-grid { display: flex; justify-content: center; gap: 3rem; flex-wrap: wrap; margin: 2rem 0; }
.team-member { text-align: center; }
.team-member img { width: 120px; height: 120px; border-radius: 50%; object-fit: cover; border: 3px solid #ddd; }
.team-member p { margin-top: 0.75rem; font-weight: 500; color: #333; }
</style>

<div class="team-grid">
  <div class="team-member">
    <img src="assets/vla-0-smol/1650104310604.jpeg" alt="Oleg Balakhnov">
    <p><a href="https://www.linkedin.com/in/oleg-balakhnov-a591a1233/">Oleg Balakhnov</a></p>
  </div>
  <div class="team-member">
    <img src="assets/vla-0-smol/1740353302751.jpeg" alt="Sergei Skvortsov">
    <p><a href="https://www.linkedin.com/in/sergei-skvortsov/">Sergei Skvortsov</a></p>
  </div>
  <div class="team-member">
    <img src="assets/vla-0-smol/1736204199843.jpeg" alt="Gleb Zarin">
    <p><a href="https://www.linkedin.com/in/zaringleb/">Gleb Zarin</a></p>
  </div>
</div>
