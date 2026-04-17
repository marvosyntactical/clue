"""LIBERO rollout evaluation for VLA continual learning.

Runs parallel MuJoCo rollouts for a given LIBERO task and returns the
success rate. This replaces the text-generation evaluation in eval/evaluate.py
— the interface to AccuracyMatrix is identical (pass success rate as accuracy).

Usage:
    sr = evaluate_libero_task(model, processor, "libero_object", task_idx=0)
    acc_matrix.update(eval_task_idx=0, trained_up_to=current_task, accuracy=sr)
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vla.data.libero_loader import (
    ActionNormStats,
    get_libero_task_order,
    LIBERO_TASK_ORDERS,
)


# ---------------------------------------------------------------------------
# LIBERO environment creation
# ---------------------------------------------------------------------------

def _create_libero_env(suite_name: str, task_idx: int):
    """Create a single LIBERO environment for the given task.

    Returns:
        env: LIBERO MuJoCo environment instance.
        task_spec: The LIBERO task specification object.
    """
    from libero.libero import benchmark

    # Map suite name to LIBERO benchmark class
    suite_map = {
        "libero_object": "libero_object",
        "libero_spatial": "libero_spatial",
        "libero_goal": "libero_goal",
        "libero_long": "libero_long",
    }
    bench = benchmark.get_benchmark(suite_map[suite_name])()
    task_spec = bench.get_task(task_idx)

    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=task_spec.bddl_file,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="agentview",
        use_camera_obs=True,
        camera_heights=256,
        camera_widths=256,
        reward_shaping=False,
    )
    env.seed(0)
    return env, task_spec


def _create_libero_envs(suite_name: str, task_idx: int, n_envs: int):
    """Create multiple LIBERO environments for parallel rollouts.

    Returns list of (env, task_spec) tuples.
    """
    envs = []
    for i in range(n_envs):
        env, task_spec = _create_libero_env(suite_name, task_idx)
        env.seed(i)
        envs.append((env, task_spec))
    return envs


# ---------------------------------------------------------------------------
# Single-episode rollout
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_single_episode(
    model: nn.Module,
    processor: Any,
    env,
    task_instruction: str,
    action_norm_stats: ActionNormStats | None,
    task_idx: int,
    max_steps: int = 300,
    device: str = "cuda",
) -> bool:
    """Run one episode and return whether it succeeded.

    Args:
        model: OpenVLA model (with LoRA applied).
        processor: OpenVLA processor for image/text tokenization.
        env: LIBERO environment instance.
        task_instruction: Natural language instruction for this task.
        action_norm_stats: Per-task normalization stats. None to skip.
        task_idx: For selecting the right normalization stats.
        max_steps: Max environment steps before timeout.
        device: Torch device string.

    Returns:
        True if the episode reached the success condition.
    """
    obs = env.reset()
    for step_i in range(max_steps):
        # Extract observation image
        image = obs["agentview_image"]
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Predict action via OpenVLA
        action = predict_action(
            model, processor, image, task_instruction, device=device
        )

        # Unnormalize if stats are available
        if action_norm_stats is not None and task_idx in action_norm_stats.stats:
            action = action_norm_stats.unnormalize(action, task_idx)

        # Step environment
        obs, reward, done, info = env.step(action)

        # Check success
        if done or info.get("success", False):
            return info.get("success", reward > 0)

    return False


# ---------------------------------------------------------------------------
# Action prediction
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_action(
    model: nn.Module,
    processor: Any,
    image: np.ndarray,
    instruction: str,
    device: str = "cuda",
) -> np.ndarray:
    """Predict a 7-DoF action from an image observation and instruction.

    This wraps OpenVLA's inference protocol:
    1. Process image + instruction through the VLA processor
    2. Generate action tokens via the model
    3. Decode action tokens to continuous action values

    Args:
        model: OpenVLA model.
        processor: OpenVLA processor.
        image: (H, W, 3) uint8 observation.
        instruction: Task instruction string.
        device: Torch device.

    Returns:
        action: (7,) float32 array — 7-DoF action.
    """
    from PIL import Image

    # Convert numpy image to PIL
    pil_image = Image.fromarray(image)

    # Format prompt as OpenVLA expects
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"

    # Process inputs
    inputs = processor(prompt, pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate action tokens
    # OpenVLA generates 7 tokens (one per action dimension), each from a
    # vocabulary of 256 bins
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=7,
            do_sample=False,
        )

    # Decode generated token IDs to continuous actions
    # OpenVLA uses a discretization scheme: 256 uniform bins per dimension
    # The action vocabulary starts after the text vocabulary
    action_token_ids = generated[0, -7:]  # last 7 tokens

    # Try using OpenVLA's built-in decode if available
    if hasattr(model, "get_action"):
        # Some OpenVLA versions have a convenience method
        action = model.get_action(generated)
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        return action.astype(np.float32)

    # Manual decode: map token IDs to [-1, 1] range
    # OpenVLA's action token offset (action tokens start at this vocab index)
    ACTION_TOKEN_OFFSET = 32000  # typical for Llama-based VLAs
    N_BINS = 256

    action_bins = (action_token_ids - ACTION_TOKEN_OFFSET).clamp(0, N_BINS - 1)
    action = (action_bins.float() / (N_BINS - 1)) * 2.0 - 1.0  # [-1, 1]
    return action.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_libero_task(
    model: nn.Module,
    processor: Any,
    suite_name: str,
    task_idx: int,
    action_norm_stats: ActionNormStats | None = None,
    n_rollouts: int = 20,
    max_steps: int = 300,
    n_parallel: int = 8,
    device: str = "cuda",
    verbose: bool = True,
) -> float:
    """Evaluate a single LIBERO task via rollouts.

    Runs n_rollouts episodes in batches of n_parallel and returns the
    success rate.

    Args:
        model: OpenVLA model with LoRA applied.
        processor: OpenVLA processor.
        suite_name: LIBERO suite name (e.g. "libero_object").
        task_idx: Task index within the suite (0-9).
        action_norm_stats: Per-task action normalization stats.
        n_rollouts: Number of rollout episodes.
        max_steps: Max steps per episode.
        n_parallel: Number of parallel environments.
        device: Torch device string.
        verbose: Print per-episode results.

    Returns:
        Success rate in [0, 1].
    """
    model.eval()
    task_order = get_libero_task_order(suite_name)
    task_instruction = task_order[task_idx].replace("_", " ")

    successes = 0
    total = 0

    # Process rollouts in batches of n_parallel
    # (sequential within each batch — MuJoCo envs are not trivially
    # parallelizable in-process, but we create separate instances)
    remaining = n_rollouts
    while remaining > 0:
        batch_size = min(remaining, n_parallel)
        envs = _create_libero_envs(suite_name, task_idx, batch_size)

        for env_idx, (env, task_spec) in enumerate(envs):
            try:
                success = _run_single_episode(
                    model=model,
                    processor=processor,
                    env=env,
                    task_instruction=task_instruction,
                    action_norm_stats=action_norm_stats,
                    task_idx=task_idx,
                    max_steps=max_steps,
                    device=device,
                )
                if success:
                    successes += 1
                total += 1

                if verbose:
                    print(
                        f"  Rollout {total}/{n_rollouts}: "
                        f"{'SUCCESS' if success else 'FAIL'} "
                        f"(running SR: {successes}/{total} = {successes/total:.1%})"
                    )
            finally:
                env.close()

        remaining -= batch_size

    success_rate = successes / max(total, 1)
    if verbose:
        print(
            f"Task {task_idx} ({task_instruction}): "
            f"SR = {successes}/{total} = {success_rate:.1%}"
        )
    return success_rate


def evaluate_all_tasks(
    model: nn.Module,
    processor: Any,
    suite_name: str,
    up_to_task: int,
    action_norm_stats: ActionNormStats | None = None,
    n_rollouts: int = 20,
    max_steps: int = 300,
    n_parallel: int = 8,
    device: str = "cuda",
) -> dict[int, float]:
    """Evaluate tasks 0..up_to_task and return per-task success rates.

    Returns:
        Dict mapping task_idx -> success_rate.
    """
    results = {}
    for task_idx in range(up_to_task + 1):
        sr = evaluate_libero_task(
            model=model,
            processor=processor,
            suite_name=suite_name,
            task_idx=task_idx,
            action_norm_stats=action_norm_stats,
            n_rollouts=n_rollouts,
            max_steps=max_steps,
            n_parallel=n_parallel,
            device=device,
        )
        results[task_idx] = sr
    return results
