# VLA HOWTO: Benchmarking CLUE on LIBERO with OpenVLA

This document is for the Claude Code instance on a fresh pod. Read it
before doing anything. The VLA pipeline in `vla/` was **scaffolded but
never run end-to-end**. There are unverified pieces that will need
debugging. Don't trust the code until you've validated each stage.

---

## Pre-flight: what the pod needs

### Hardware
- **GPU:** A40 (48 GB) or larger. Qwen-7B in 4-bit was the previous
  workload; OpenVLA-7B at LoRA rank 32 with batch 16 needs more memory
  than that — expect ~25-35 GB peak.
- **Disk:** **60 GB minimum.** The 20 GB pod we used previously was too
  small. Breakdown:
  - 15 GB OpenVLA-7B weights
  - 10 GB LIBERO RLDS data (one suite is ~3 GB; budget 10 GB for safety
    + multiple suites)
  - 5 GB Python packages (openvla, libero, mujoco, robosuite, tfds)
  - 5 GB working space (LoRA checkpoints, eval logs, video renders)
  - 25 GB headroom (LIBERO/MuJoCo create simulation artifacts and dataset
    caches that grow during long runs)

### Software stack expected
- PyTorch 2.4+ (older versions lack `nn.Module.set_submodule` which
  `transformers >= 5.x` needs; on the previous pod we downgraded
  `transformers==4.46.3` to work around this — do that here too)
- `bitsandbytes` 0.49+ (for 4-bit quantization)
- `peft` 0.10+
- New for VLA: `openvla`, `libero`, `mujoco>=2.3`, `robosuite`,
  `tensorflow-datasets`

---

## Step 1: Disk and environment setup

```bash
# Verify disk
df -h /
# Expect at least 50 GB free

# Verify GPU
nvidia-smi
# Expect 40+ GB VRAM

# Clone openvla and libero
cd /workspace
git clone https://github.com/openvla/openvla.git
pip install -e openvla/

git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO/
pip install -r LIBERO/requirements.txt

# MuJoCo + robosuite come as deps but verify
python -c "import mujoco; print(mujoco.__version__)"
python -c "import robosuite; print(robosuite.__version__)"
python -c "import libero; print(libero.__file__)"
```

If any import fails, fix it before proceeding. `mujoco` in particular
requires GL libraries on the host. On RunPod templates that work for
robotics, this is usually pre-installed; on minimal templates you'll
need `apt install -y libgl1-mesa-glx libosmesa6-dev libglew-dev`.

### Sanity check the LIBERO benchmark API

The first thing we couldn't verify on the previous pod was whether
LIBERO's Python API matches what `vla/eval_vla.py` assumes. Run:

```python
from libero.libero import benchmark
b = benchmark.get_benchmark("libero_object")()
print(b.get_task(0).bddl_file)  # should print a file path
```

If `get_benchmark` doesn't exist or has a different signature, the
hardcoded usage in `_create_libero_env` (in `vla/eval_vla.py`) needs
updating. The LIBERO repo has changed its API at least once.

### Download the RLDS data

```bash
mkdir -p /data
cd /data
# Use HuggingFace CLI to clone (~3 GB for libero_object, ~10 GB for all suites)
huggingface-cli download openvla/modified_libero_rlds \
    --local-dir /data/libero_rlds \
    --include "libero_object_no_noops/*"
```

Verify by loading via tfds:

```python
import tensorflow_datasets as tfds
b = tfds.builder_from_directory("/data/libero_rlds/libero_object_no_noops")
ds = b.as_dataset(split="train")
for ep in ds.take(1):
    print(list(ep.keys()))  # should include 'steps'
    break
```

---

## Step 2: Reproduce single-task OpenVLA performance (sanity check)

Before any continual learning, verify the model + env stack works on a
single task. This catches 80% of pipeline bugs.

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 \
    openvla/vla-scripts/finetune.py \
    --vla_path openvla/openvla-7b \
    --data_root_dir /data/libero_rlds \
    --dataset_name libero_object_no_noops \
    --lora_rank 32 \
    --batch_size 16 \
    --learning_rate 5e-4 \
    --max_steps 1000 \
    --save_freq 1000
```

Then run an eval rollout. If you can't get ~80% success on
`libero_object` task 0 with this config, **stop and debug** — there's
a setup problem. Don't move on.

---

## Step 3: Smoke-test our CL pipeline (small)

Once the standalone OpenVLA setup is working, validate the code in
`vla/` end-to-end on a tiny scale. **Do not start a full benchmark
before this passes.**

```bash
cd /workspace/clue
python -m vla.train_vla \
    --suite libero_object \
    --method seq_lora \
    --epochs 1 \
    --batch_size 8 \
    --grad_accum 2 \
    --lora_rank 16 \
    --lora_target_modules all-linear \
    --n_rollouts 3 \
    --eval_tasks 0,2,4 \
    --data_root /data/libero_rlds \
    --output_dir outputs/vla_smoketest \
    --seed 42
```

This trains 5 tasks (0..4) sequentially with no CL protection
(`seq_lora`), evaluates with 3 rollouts after tasks {0, 2, 4}.
Expected wall-clock: ~30-60 min if everything works. Each task's
training is short (1 epoch, small batch).

### Things that will probably break (in order of likelihood)

1. **`LiberoTaskDataset._load_task_data`** — the RLDS reading code in
   `vla/data/libero_loader.py` has three fallback paths (tfds,
   load_from_disk, HF Hub). The first one is the right path for the
   official RLDS data, but the field names (`agentview_image`,
   `natural_language_instruction`, `joint_pos`) are guesses based on
   typical RLDS schemas. Check what the actual fields are in the
   loaded episode and update `_parse_step` and `_extract_instruction`
   accordingly.

2. **OpenVLA forward pass** — `_forward_vla` in `vla/train_vla.py`
   manually builds the `input_ids` by appending discretized action
   tokens. This may not match how the real OpenVLA wants its inputs
   (it has its own `PaddedCollatorForActionPrediction` that we're
   bypassing). If loss is `nan` or doesn't decrease, replace our
   manual tokenization with OpenVLA's collator.

3. **Action discretization offset** — `_discretize_actions` uses
   `action_token_offset=32000`. OpenVLA's actual offset depends on
   the tokenizer and may differ. Read OpenVLA's source for the
   correct value, or import it.

4. **LIBERO env creation** — `_create_libero_env` in `vla/eval_vla.py`
   uses `OffScreenRenderEnv(bddl_file_name=...)`. If LIBERO's API has
   changed, this needs updating. Look at LIBERO's example scripts.

5. **Rollout action prediction** — `predict_action` in `eval_vla.py`
   tries to call `model.get_action()` if it exists, else manually
   decodes tokens. The manual decode is almost certainly wrong;
   OpenVLA has a specific decode path with action normalization that
   we're not using. Either find OpenVLA's `get_action` method on the
   model class, or import their decode utilities.

6. **PEFT + OpenVLA model class** — OpenVLA may not be a standard
   `Vision2Seq` model class. The `AutoModelForVision2Seq` call in
   `model_server`-equivalent (well, the `load_openvla` function in
   `train_vla.py`) may need to be `AutoModel.from_pretrained` with
   `trust_remote_code=True` and `processor` loaded separately.

7. **LoRA on Vision encoder** — `target_modules="all-linear"` will
   apply LoRA to OpenVLA's DINOv2 vision encoder too, which we may
   not want. The standard OpenVLA fine-tuning recipe applies LoRA
   only to the LLM backbone. If memory or quality is bad, restrict
   to the LLM submodule explicitly.

---

## Step 4: Hyperparameters to use

These are derived from `bench_fmerge05_O4_sXX` (our LLM CL benchmark
at AA=80.3%, BWT=+1.0% averaged across 9 runs):

```yaml
method: slao
lora_rank: 32                # OpenVLA default; higher than LLM CL because VLA tasks carry more diverse info
lora_alpha: 16               # OpenVLA default — note 16/32 = 0.5 scaling, intentional
lora_target_modules: all-linear   # Best result on LLM was all-modules; matches QLoRA default
optimizer: adamw
lr: 5e-4                     # OpenVLA default
batch_size: 16
grad_accum: 4                # Effective batch 64
epochs: 3                    # Per task; longer than LLM CL because action data is denser
max_grad_norm: 1.0
torch_dtype: bfloat16
image_aug: true              # OpenVLA default (random 90% crop)

# CLUE knobs (the headline contribution from EXTENSIONS.md)
fisher_merge_beta: 0.5       # The Fisher-weighted B merge — this is the CLUE secret sauce
fisher_lambda: 0.0           # No EWC penalty during training (best LLM result didn't use it)
fisher_gamma: 1.0            # Standard EWC accumulation (unused since lambda=0)
fisher_samples: 256          # More than LLM (action distributions are peakier → noisier Fisher)

# Don't enable for the headline run
riemannian: false
lora_plus_ratio: 1.0
gpm_threshold: 0.0
bayesian_merge: false
```

### Eval protocol

| Stage | n_rollouts | eval_after | Cost |
|---|---|---|---|
| Smoke test | 3 | tasks 0, 2, 4 | 30-60 min |
| Iteration | 20 | tasks 0, 4, 9 | 4-8 hours |
| Final (one seed) | 50 | every task | 12-20 hours |
| Final (3 seeds, headline number) | 50 | every task | ~2-3 days |

For Demo Day: **iteration-quality numbers** (20 rollouts, eval at 3
points) is enough to tell whether CLUE beats SLAO. Don't burn 3 days
on 50-rollout final numbers until iteration looks good.

---

## Step 5: Full benchmark

Once the smoke test passes:

```bash
# Headline CLUE run
python -m vla.train_vla \
    --suite libero_object \
    --method slao \
    --fisher_merge_beta 0.5 \
    --fisher_samples 256 \
    --epochs 3 \
    --lr 5e-4 \
    --batch_size 16 \
    --grad_accum 4 \
    --lora_rank 32 \
    --lora_target_modules all-linear \
    --n_rollouts 20 \
    --eval_tasks all \
    --data_root /data/libero_rlds \
    --output_dir outputs/vla_clue_libero_object_s42 \
    --seed 42

# SLAO baseline (no Fisher merging)
python -m vla.train_vla \
    --suite libero_object \
    --method slao \
    --fisher_merge_beta 0.0 \
    --epochs 3 --lr 5e-4 --batch_size 16 --grad_accum 4 \
    --lora_rank 32 --lora_target_modules all-linear \
    --n_rollouts 20 --eval_tasks all \
    --data_root /data/libero_rlds \
    --output_dir outputs/vla_slao_libero_object_s42 --seed 42

# Sequential FT baseline (forgetting floor)
python -m vla.train_vla \
    --suite libero_object \
    --method seq_lora \
    --epochs 3 --lr 5e-4 --batch_size 16 --grad_accum 4 \
    --lora_rank 32 --lora_target_modules all-linear \
    --n_rollouts 20 --eval_tasks all \
    --data_root /data/libero_rlds \
    --output_dir outputs/vla_seqft_libero_object_s42 --seed 42
```

The headline comparison: CLUE vs SLAO vs SeqFT. The key result for the
paper/CV is **CLUE > SLAO** (ablating Fisher merging), and **CLUE
beats SeqFT by a large BWT margin**.

---

## Files in `vla/` and their roles

```
vla/
├── data/libero_loader.py    # RLDS → per-task datasets. Has three fallback loading paths;
│                              the field names are guessed and need verification.
├── configs/libero_object.yaml  # Suite definitions (task names, hyperparams)
│                              Mostly informational; train_vla.py uses CLI args.
├── eval_vla.py              # MuJoCo rollouts. _create_libero_env, predict_action, and
│                              _run_single_episode all need verification against the actual
│                              LIBERO/OpenVLA APIs.
└── train_vla.py             # Main CL loop. Reuses methods/slao.py + methods/fisher.py
                              from clue/. The only VLA-specific bits are _forward_vla and
                              estimate_fisher_vla, which call the OpenVLA model with
                              image+text inputs.
```

What's reused unchanged:
- `methods/slao.py` — orthogonal init, asymmetric merge, Fisher-weighted merge
- `methods/fisher.py` — diagonal Fisher (works for any model with gradients)
- `models/lora.py` — extract/set LoRA state, merge functions
- `eval/metrics.py` — AccuracyMatrix, AA, BWT (just takes float "accuracy" values
  which here are success rates)

---

## Specific things that previously broke or were never validated

1. **Disk:** the previous pod ran out of disk during the Qwen-7B
   experiments. Plan ahead. Move HF cache to the volume disk if you're
   on a tight container disk.

2. **`bitsandbytes`:** previous install was corrupt. If you see
   `ImportError: cannot import name 'research'`, do
   `rm -rf /usr/local/lib/python3.11/dist-packages/bitsandbytes* && pip install --no-cache-dir bitsandbytes`.

3. **`transformers` version:** if the OpenVLA model class needs
   `set_submodule`, you need PyTorch 2.5+. If you have PyTorch 2.4 like
   the previous pod, downgrade `transformers` to 4.46.x and use
   `torch_dtype=` instead of `dtype=` in `from_pretrained`.

4. **`gradient_checkpointing`:** essential for OpenVLA-7B fine-tuning
   at batch 16. Don't disable it (we did for the chat demo because
   rank 8 fit comfortably; rank 32 + 7B + activations does not).

5. **Action token offset:** the value 32000 in `_discretize_actions` is
   a guess. Verify with OpenVLA's tokenizer:
   ```python
   from transformers import AutoTokenizer
   tok = AutoTokenizer.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
   # Find where action tokens live in the vocabulary
   print(tok.vocab_size)  # base vocab
   # OpenVLA action tokens are typically the last 256 tokens
   ```

6. **Reward / success criterion:** LIBERO's success check varies by
   task. The default in `_run_single_episode` checks
   `info.get("success", False)` and falls back to `reward > 0`. Verify
   that `info["success"]` actually appears in the env step output for
   LIBERO tasks; if not, you'll need a per-task success checker.

7. **Action normalization:** `ActionNormStats` computes mean/std from
   training data. OpenVLA's official protocol uses `q01`/`q99`
   percentile normalization with a per-dataset `unnorm_key`. Match
   theirs if results are off.

---

## Decision tree for when things go wrong

**Smoke test fails to load model:** OpenVLA install is broken. `pip
install -e openvla/` again, check Python version compat.

**Smoke test loads but `_forward_vla` returns NaN loss:** action
tokenization is wrong. Replace our manual tokenization with OpenVLA's
collator (import from openvla package).

**Training runs but rollout success rate is 0%:** action decoding is
wrong. The model is producing tokens but they're being decoded to
nonsense actions. Use OpenVLA's `model.predict_action()` if it exists.

**Training runs, rollout works, but task 0 SR is well below 80%:** LoRA
config is wrong (probably target_modules excluding important layers,
or rank too low). Match OpenVLA's published LoRA config exactly first;
optimize after baseline works.

**Everything works for one task but multi-task fails:** the SLAO
weight extraction (`extract_lora_state`) is iterating over modules
that don't have `lora_A`/`lora_B` attributes the way it expects. PEFT
may have a different module structure for OpenVLA than for plain
Llama. Print all modules with `lora_` in the name first to verify.

---

## What success looks like

A good full run produces, per method, an output directory containing:
- `config.json` — full hyperparameters
- `results.json` — full T×T accuracy matrix, AA, BWT
- `task_*/eval_results.json` — per-task results matrix snapshot

The headline numbers go in a table:

| Method | LIBERO-Object AA | BWT |
|---|---|---|
| SeqFT (forgetting floor) | ~40-55% | ~-15% |
| SLAO (no Fisher merge) | ~70-78% | ~-3% |
| **CLUE** (Fisher merge β=0.5) | **target: > SLAO** | **target: BWT >= 0** |
| Multi-task (oracle, optional) | ~85-95% | n/a |

Beating SLAO by 1-2% AA and BWT is the demo win. Beating it by 5%+ is
a paper win.

---

## What to NOT do

- Don't run the full 50-rollout × 10-task × 3-seed protocol before the
  smoke test passes.
- Don't try to also benchmark LIBERO-Long, Spatial, Goal in the same
  pass. Get LIBERO-Object working end-to-end first.
- Don't enable `--riemannian` or `--gpm_threshold` for the headline
  CLUE run; those are research ablations and didn't help on LLM CL.
- Don't change the LoRA target_modules to anything other than
  `all-linear` unless OpenVLA can't fit at that setting; the all-linear
  result is materially better than `q_proj v_proj` for LLM CL and
  should also be for VLA.
- Don't add a vision encoder LoRA on the headline run; that's a
  separate ablation and adds a lot of complexity. Note in the writeup
  that vision-encoder adaptation is future work.

---

## Final note for the next Claude Code instance

The previous instance (me) was honest about what's verified and what's
not. The `vla/` code is well-structured but has several pieces that
were guessed rather than tested. Treat the smoke test as a debugging
exercise, not a check. Expect 2-5 bugs before it runs cleanly. Budget
4-8 hours for the smoke test phase before starting the real benchmark.

If you find the OpenVLA API has shifted significantly since this code
was written (April 2026 — check the openvla repo's last commit date),
or if LIBERO has had a major version bump, **propose updates to the
human before plowing ahead** with what may be obsolete glue code.

Good luck.
