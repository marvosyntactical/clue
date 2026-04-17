# CLUE on LIBERO: Experiment Specification

**Purpose.** Validate CLUE (SLAO + Fisher-weighted B merging, `fisher_merge_beta=0.5`) as a continual learning method for Vision-Language-Action models. Existing Llama-2-7B results show CLUE achieves +1.0% BWT on the Large Number of Tasks benchmark vs. SAPT-LoRA at −2.9%. This plan tests whether that gain transfers to VLA continual manipulation.

**Timeline.** 3 weeks to FR8 Demo Day. Budget ~30 A100-days on AWS.

---

## 1. Research Questions

**RQ1 (primary).** Does CLUE's Fisher-weighted asymmetric merging reduce catastrophic forgetting on sequential VLA fine-tuning compared to standard CL baselines?

**RQ2 (architectural transfer).** Does the EWC Fisher estimation, derived in the SLAO paper for cross-entropy LLM heads, remain meaningful when applied to OpenVLA's discrete action-token head (cross-entropy-compatible)? If we scale to OpenVLA-OFT (L1 regression), does it still hold, or does the Fisher need rederivation?

**RQ3 (scaling).** Does CLUE's BWT advantage grow with sequence length? LLM results showed the gap widening from 5 to 15 tasks. Cross-suite LIBERO evaluation reaches 40 tasks.

A negative result on RQ1 is itself informative — it reshapes the demo-day narrative toward "CL for LLMs" rather than "CL for agents."

---

## 2. Benchmarks

### Primary: LIBERO (Liu et al., 2023)

Four 10-task manipulation suites in MuJoCo:

- **LIBERO-Object** — primary benchmark. ~300-step horizon, faster rollouts. Best cost/signal ratio.
- **LIBERO-Long** — confirmation run for headline number. ~600-step horizon, 10 long-horizon tasks, most forgetting-sensitive.
- **LIBERO-Spatial, LIBERO-Goal** — reserve for post-demo-day paper.

Cross-suite protocol (40 tasks): train Object → Goal → Spatial → Long sequentially for RQ3.

### Fast iteration loop: Meta-World ML10 (Yu et al., 2019)

State-based, no vision. Full BWT matrix in <1 hour per seed. Use for debugging CL mechanics (Fisher damping, β tuning, EWC γ) before committing LIBERO compute. Not reported as a result — diagnostic only.

### Stretch: LIBERO-PRO (2025)

Perturbed LIBERO testing true robustness. Post-demo-day.

---

## 3. Base Model

**OpenVLA-7B** (Kim et al., 2024) is the primary base model.

Rationale:
- Llama-2-7B backbone — identical to the SLAO/CLUE validation regime, so the Fisher estimation carries over with minimal adaptation.
- Discrete action tokens with cross-entropy loss — directly compatible with CLUE's current Fisher code.
- LoRA finetuning is the documented adaptation path (rank 32, 1.4% of params matches full FT performance).
- Published baselines exist for every suite.

**Stretch: OpenVLA-OFT** if time permits. L1 regression head requires rederiving Fisher (Laplace likelihood → absolute-value gradient magnitudes) — this is a research risk, budget 3-5 days.

---

## 4. Methods to Compare

All methods fine-tune LoRA adapters (rank r=32, targeting q_proj and v_proj; optionally all linear layers as ablation) on sequentially presented tasks.

| Method | Description | Source |
|---|---|---|
| **Sequential FT** | Naive sequential LoRA fine-tuning, no protection. Forgetting floor. | — |
| **Experience Replay** | Mix 10% buffer of old-task data into new-task training. Strong simple baseline. | Chaudhry et al., 2019 |
| **EWC** | Diagonal Fisher penalty, γ=0.9 online. | Kirkpatrick et al., 2017 |
| **SeqLoRA** | SLAO paper's sequential baseline (no merging). | Qiao & Mahdavi, 2025 |
| **IncLoRA** | One LoRA per task, summed at inference. Linear memory growth. | Qiao & Mahdavi, 2025 |
| **O-LoRA** | Orthogonal LoRA subspaces. | Wang et al., 2023 |
| **TAIL** | Task-specific adapters for VLA CL. Most direct VLA CL comparison. | Liu et al., 2024 |
| **LOTUS** | Lifelong skill discovery. Reports on LIBERO-Long. | Wan et al., 2024 |
| **SLAO (ours)** | Baseline of CLUE: asymmetric A/B, uniform merge. | Qiao & Mahdavi, 2025 |
| **CLUE (ours)** | SLAO + `fisher_merge_beta=0.5`, `fisher_lambda=0.5`, `fisher_gamma=0.9`, LoRA+ ratio=4. | This work |
| **Multi-task joint** | Oracle: train on all tasks simultaneously. Upper bound. | — |

---

## 5. Published SOTA Reference (Non-CL Setting)

**Critical framing:** these numbers are from *single-task* or *multi-task joint* training, NOT the sequential CL protocol. They are NOT directly comparable to our CL numbers. Reviewers must not confuse the regimes.

| Method | LIBERO avg | LIBERO-Long | Source |
|---|---|---|---|
| OpenVLA base (Kim et al., 2024) | ~76.5% | ~53.7% | [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) |
| π0 (Black et al., 2024) | mid-90s | 85.2% | Physical Intelligence |
| OpenVLA-OFT (Kim et al., 2025) | **97.1%** | — | [arXiv:2502.19645](https://arxiv.org/abs/2502.19645) |
| SimpleVLA-RL (Li et al., 2025) | **99.0%** | **98.5%** | [arXiv:2509.09674](https://arxiv.org/abs/2509.09674) |
| PLD / Self-Improving VLA (2025) | ~99% | — | [paper](https://wenlixiao.com/self-improve-VLA-PLD/) |

**LIBERO is saturated in the non-CL regime.** Our playing field is the CL regime where sequential FT drops to 30-50% on LIBERO-Long with strongly negative BWT.

### Expected CL-regime numbers (from literature + our LLM experience)

| Method | LIBERO-Long avg SR | BWT |
|---|---|---|
| Sequential FT | 30-50% | strongly negative |
| Experience Replay | 60-70% | mildly negative |
| EWC standalone | 55-65% | negative |
| O-LoRA | 65-75% | near −4 |
| TAIL | 75-85% | near zero |
| LOTUS | 70-80% | varies |
| **SLAO (expected)** | **75-82%** | **~−3** |
| **CLUE (target)** | **>TAIL** | **positive or near-zero** |
| Multi-task oracle | 86% (π0) to 98% (SimpleVLA-RL) | n/a |

---

## 6. Metrics

Following SLAO protocol:

**Average Success Rate (AA):**
```
AA = (1/T) · Σ_{i=1..T} a_{i,T}
```
where `a_{i,T}` is the success rate on task i after training on all T tasks.

**Backward Transfer (BWT):** the headline metric.
```
BWT = (1/(T-1)) · Σ_{i=1..T-1} (a_{i,T} - a_{i,i})
```

**Forward Transfer (FWT):** did prior tasks help? Compute against random-init baseline.

**Sample Efficiency:** success rate on new task after K demonstrations (K ∈ {10, 50, full}). Robotics audiences care about this more than BWT.

**Full accuracy matrix:** T×T matrix where entry (i,j) = SR on task i after training task j. Required for all above metrics. Build incrementally — evaluate tasks {1..i} after training task i.

---

## 7. Evaluation Protocol

Rollout parameters:
- **50 rollouts per task per eval point** — standard LIBERO protocol, but we use **20 rollouts during iteration**, 50 for final numbers. SD at n=20 is ~10% for 50%-SR tasks; acceptable for ranking, not for publication.
- **Max horizon**: 300 steps (Object/Spatial/Goal), 600 steps (Long).
- **Success criterion**: LIBERO default (object in target region at episode end).
- **Parallelization**: 8 parallel envs per A100 (CPU-bound physics). Near-linear speedup — this is the biggest practical cost reduction.

Evaluation cadence:
- Iteration runs (Week 1-2): evaluate after tasks {2, 5, 10}. Cuts eval cost ~4x vs full schedule.
- Final runs (Week 3): evaluate after every task for complete BWT matrix.

---

## 8. CLUE Configuration

Default CLUE config for LIBERO:

```bash
python train.py \
    --method slao \
    --base_model openvla/openvla-7b \
    --benchmark libero_object \
    --task_order sequential \
    --lora_rank 32 \
    --lora_target q_proj,v_proj \
    --lr 1e-4 \
    --lora_plus_ratio 4 \
    --fisher_lambda 0.5 \
    --fisher_gamma 0.9 \
    --fisher_samples 256 \
    --fisher_merge_beta 0.5 \
    --riemannian \
    --gpm_threshold 0.0 \
    --epochs_per_task 3 \
    --batch_size 16 \
    --grad_accum 4 \
    --eval_rollouts 20 \
    --seed 42
```

**Ablation grid for RQ1 (LIBERO-Object, 1 seed first, then 3 seeds on winner):**

| Config | `fisher_merge_beta` | `fisher_lambda` | `lora_plus_ratio` | `riemannian` | Purpose |
|---|---|---|---|---|---|
| SLAO | 0 | 0 | 1 | off | Ablation base |
| SLAO + LoRA+ | 0 | 0 | 4 | off | Isolate LoRA+ |
| SLAO + EWC | 0 | 0.5 | 1 | off | Isolate Fisher regularization |
| SLAO + Fisher-merge | 0.5 | 0 | 1 | off | Isolate Fisher-weighted merging (the CLUE innovation) |
| CLUE (full) | 0.5 | 0.5 | 4 | on | Full method |
| CLUE − Fisher-merge | 0 | 0.5 | 4 | on | Key ablation: does Fisher-merging add value on top of EWC? |

This isolates whether CLUE's specific contribution (Fisher-weighted B merging, β>0) adds value beyond what EWC regularization alone provides.

---

## 9. Compute Budget

Per-task LoRA finetune on OpenVLA-7B: ~2-4 hours A100.
Per-eval rollout (20 episodes, parallelized): ~8 min.

| Run | A100-days |
|---|---|
| Meta-World iteration (Week 1) | ~1 |
| LIBERO-Object ablation (6 configs × 1 seed) | ~6 |
| LIBERO-Object winning configs (2 × 3 seeds) | ~6 |
| LIBERO-Long final (3 seeds × 3 methods) | ~15 |
| Baseline re-runs where published protocol differs | ~3 |
| **Total** | **~31** |

Within AWS Activate budget.

---

## 10. Timeline

**Week 1.** Meta-World ML10 iteration. Port CLUE from LLM repo to VLA setting. Debug Fisher estimation on OpenVLA's action-token head (sanity check: Fisher magnitudes should be non-trivial and non-uniform across LoRA params). Decision gate at end of week: if CLUE doesn't beat replay + EWC on ML10, pivot — either revert to LLM story for demo day and treat VLA as future work, or investigate Fisher rederivation.

**Week 2.** LIBERO-Object with OpenVLA-7B. Run full ablation grid at 1 seed, 20 rollouts. Parallelize rollouts (8 envs). Reproduce published OpenVLA LIBERO-Object single-task number (~88%) as infrastructure sanity check before running CL experiments. If reproduction takes >5 days, scope down.

**Week 3.** LIBERO-Long confirmation run on winning configuration (CLUE) and top 2 baselines (TAIL, Experience Replay), 3 seeds each, 50 rollouts, full BWT matrix. Prepare demo-day figures:
- BWT comparison bar chart (CLUE vs. baselines, error bars across seeds)
- Accuracy matrix heatmap showing forgetting pattern
- Learning curve: AA on all-tasks-so-far vs. task index

---

## 11. Decision Rules (Pre-Committed)

1. If Meta-World ML10 shows CLUE not beating replay + EWC by end of Week 1 → pivot, don't burn LIBERO compute.
2. If OpenVLA-7B LIBERO-Object reproduction fails by end of Week 2 day 3 → scope to a smaller base model (SmolVLA) or revert to LLM-only story.
3. If CLUE beats SLAO by <0.5% BWT on LIBERO-Object (within noise) → the Fisher-merge contribution is not robust across architectures; report honestly and frame as LLM-specific in demo day.
4. If CLUE beats SLAO by >1% BWT on LIBERO-Object → run LIBERO-Long confirmation.
5. If CLUE beats TAIL (current CL-for-VLA SOTA) → headline result for demo day.

---

## 12. Known Risks

**Fisher estimation on action tokens.** OpenVLA discretizes actions into 256 bins per dimension. The resulting cross-entropy loss has very peaked distributions (near-deterministic optimal actions), which can make empirical Fisher estimates noisy or degenerate. Mitigation: estimate Fisher from the top-k nearest action bins, not just the argmax. Budget 1 day to validate Fisher sanity.

**LoRA-on-vision-encoder vs. LoRA-on-LLM.** OpenVLA's LoRA can target just the LLM backbone (standard) or include vision encoder adapters. CLUE was validated only on LLM adapters. We apply to LLM only for the primary run; vision-encoder adaptation is a post-demo-day extension.

**LIBERO task similarity.** Tasks within a suite are visually and semantically similar. Forgetting might be mild even with sequential FT, compressing our delta over baselines. Mitigation: cross-suite evaluation (Object → Goal → Spatial → Long) for RQ3, where distribution shift is real.

**Protocol mismatch with published baselines.** TAIL, LOTUS report on slightly different LIBERO configurations. We re-run them under our protocol where feasible and flag asymmetries where not.

---

## 13. References

**Benchmarks and base models**
- Liu, Zhu, Chen, Fan, Biza, Zhu, Martín-Martín. "LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning." 2023. [arXiv:2306.03310](https://arxiv.org/abs/2306.03310)
- Yu, Quillen, He, Julian, Hausman, Finn, Levine. "Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning." 2019. [arXiv:1910.10897](https://arxiv.org/abs/1910.10897)
- Kim, Pertsch, Karamcheti, Xiao, Balakrishna, Nair, Rafailov, Foster, Lam, Sanketi, Vuong, Kollar, Burchfiel, Tedrake, Sadigh, Levine, Liang, Finn. "OpenVLA: An Open-Source Vision-Language-Action Model." 2024. [arXiv:2406.09246](https://arxiv.org/abs/2406.09246)
- Kim, Karamcheti, et al. "Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success" (OpenVLA-OFT). 2025. [arXiv:2502.19645](https://arxiv.org/abs/2502.19645) · [project](https://openvla-oft.github.io/)
- Black, Brown, Driess, et al. "π0: A Vision-Language-Action Flow Model for General Robot Control." 2024. [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)

**CL-for-VLA baselines**
- Liu, Zhu, et al. "TAIL: Task-specific Adapters for Imitation Learning with Large Pretrained Models." 2024. [arXiv:2310.05905](https://arxiv.org/abs/2310.05905)
- Wan, Zhu, et al. "LOTUS: Continual Imitation Learning for Robot Manipulation via Unsupervised Skill Discovery." 2024. [arXiv:2311.02058](https://arxiv.org/abs/2311.02058)

**Continual learning methods**
- Kirkpatrick, Pascanu, Rabinowitz, et al. "Overcoming catastrophic forgetting in neural networks." PNAS 2017. [arXiv:1612.00796](https://arxiv.org/abs/1612.00796)
- Schwarz, Czarnecki, Luketina, et al. "Progress & Compress: A scalable framework for continual learning." ICML 2018. [arXiv:1805.06370](https://arxiv.org/abs/1805.06370)
- Chaudhry, Rohrbach, Elhoseiny, et al. "On Tiny Episodic Memories in Continual Learning." 2019. [arXiv:1902.10486](https://arxiv.org/abs/1902.10486)
- Wang, Liu, et al. "Orthogonal Subspace Learning for Language Model Continual Learning" (O-LoRA). 2023. [arXiv:2310.14152](https://arxiv.org/abs/2310.14152)
- Saha, Garg, Roy. "Gradient Projection Memory for Continual Learning." ICLR 2021. [arXiv:2103.09762](https://arxiv.org/abs/2103.09762)

**LoRA methods**
- Hu, Shen, Wallis, et al. "LoRA: Low-Rank Adaptation of Large Language Models." 2021. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Hayou, Ghosh, Yu. "LoRA+: Efficient Low Rank Adaptation of Large Models." 2024. [arXiv:2402.12354](https://arxiv.org/abs/2402.12354)
- Prakash, Zhao, et al. "Riemannian Preconditioned LoRA for Fine-Tuning Foundation Models." 2024. [arXiv:2402.02347](https://arxiv.org/abs/2402.02347)
- Qiao, Mahdavi. "Merge before Forget: A Single LoRA Continual Learning via Continual Merging" (SLAO). 2025. [arXiv:2512.23017](https://arxiv.org/abs/2512.23017)

**SOTA reference**
- Li, Chen, et al. "SimpleVLA-RL: Scaling VLA Training via RL." 2025. [arXiv:2509.09674](https://arxiv.org/abs/2509.09674)
- Xiao, et al. "Self-Improving Vision-Language-Action Models with Data Generation" (PLD). 2025. [project](https://wenlixiao.com/self-improve-VLA-PLD/)
- "LIBERO-PRO: Towards Robust and Fair Evaluation of VLA Models." 2025. [arXiv:2510.03827](https://arxiv.org/abs/2510.03827)