"""Per-task LIBERO data loading for OpenVLA continual learning.

Wraps RLDS-formatted LIBERO datasets (from HuggingFace) into per-task
PyTorch datasets compatible with OpenVLA's training collator.

Data source: openvla/modified_libero_rlds on HuggingFace.
Each suite (e.g. libero_object_no_noops) contains demonstrations for
all 10 tasks, keyed by the natural-language instruction string.

Usage:
    tasks = get_libero_task_order("libero_object")
    ds = LiberoTaskDataset("libero_object", task_idx=0, data_root="/data/libero_rlds")
    loader = DataLoader(ds, batch_size=16, collate_fn=ds.collate)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Task orderings per suite (LIBERO default ordering, indices 0-9)
# ---------------------------------------------------------------------------

LIBERO_TASK_ORDERS: dict[str, list[str]] = {
    "libero_object": [
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        "pick_up_the_cream_cheese_and_place_it_in_the_bowl",
        "pick_up_the_butter_and_place_it_in_the_basket",
        "pick_up_the_wine_bottle_and_place_it_in_the_rack",
        "pick_up_the_orange_juice_and_place_it_in_the_basket",
        "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
        "pick_up_the_ketchup_and_place_it_in_the_basket",
        "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
        "pick_up_the_milk_and_place_it_in_the_basket",
        "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
    ],
    "libero_spatial": [
        "pick_up_the_alphabet_soup_and_place_it_in_the_top_drawer_of_the_cabinet",
        "pick_up_the_cream_cheese_and_place_it_in_the_bowl",
        "pick_up_the_butter_and_place_it_in_the_tray",
        "pick_up_the_wine_bottle_and_place_it_in_the_front_of_the_cabinet",
        "pick_up_the_orange_juice_and_place_it_in_the_top_drawer_of_the_cabinet",
        "pick_up_the_bbq_sauce_and_place_it_in_the_basket_on_the_left",
        "pick_up_the_ketchup_and_place_it_in_the_front_of_the_cabinet",
        "pick_up_the_tomato_sauce_and_place_it_on_the_plate",
        "pick_up_the_milk_and_place_it_in_the_bowl",
        "pick_up_the_chocolate_pudding_and_place_it_in_the_tray",
    ],
    "libero_goal": [
        "open_the_top_drawer_of_the_cabinet",
        "put_the_bowl_on_the_plate",
        "put_the_cream_cheese_in_the_bowl",
        "turn_on_the_stove",
        "put_the_wine_bottle_on_the_rack",
        "turn_on_the_green_light",
        "put_the_bowl_on_top_of_the_cabinet",
        "push_the_plate_to_the_front_of_the_stove",
        "put_the_cream_cheese_in_the_top_drawer_of_the_cabinet",
        "turn_off_the_stove",
    ],
    "libero_long": [
        "pick_up_the_alphabet_soup_and_place_it_in_the_basket_then_push_the_basket_to_the_target",
        "pick_up_the_cream_cheese_and_place_it_in_the_bowl_then_push_the_bowl_to_the_target",
        "pick_up_the_butter_and_place_it_in_the_basket_then_push_the_basket_to_the_target",
        "pick_up_the_wine_bottle_and_place_it_in_the_rack_then_push_the_rack_to_the_target",
        "pick_up_the_orange_juice_and_place_it_in_the_basket_then_push_the_basket_to_the_target",
        "pick_up_the_bbq_sauce_and_place_it_in_the_basket_then_push_the_basket_to_the_target",
        "pick_up_the_ketchup_and_place_it_in_the_basket_then_push_the_basket_to_the_target",
        "pick_up_the_tomato_sauce_and_place_it_in_the_basket_then_push_the_basket_to_the_target",
        "pick_up_the_milk_and_place_it_in_the_basket_then_push_the_basket_to_the_target",
        "pick_up_the_chocolate_pudding_and_place_it_in_the_basket_then_push_the_basket_to_the_target",
    ],
}

# Map suite name to the HuggingFace RLDS dataset name
SUITE_TO_DATASET: dict[str, str] = {
    "libero_object": "libero_object_no_noops",
    "libero_spatial": "libero_spatial_no_noops",
    "libero_goal": "libero_goal_no_noops",
    "libero_long": "libero_long_no_noops",
}


def get_libero_task_order(suite_name: str) -> list[str]:
    """Return the ordered list of task instruction strings for a suite."""
    if suite_name not in LIBERO_TASK_ORDERS:
        raise ValueError(
            f"Unknown suite '{suite_name}'. "
            f"Available: {list(LIBERO_TASK_ORDERS.keys())}"
        )
    return LIBERO_TASK_ORDERS[suite_name]


# ---------------------------------------------------------------------------
# Per-task dataset
# ---------------------------------------------------------------------------

class LiberoTaskDataset(Dataset):
    """Single LIBERO task's demonstrations, formatted for OpenVLA training.

    Loads the RLDS dataset for the given suite, filters to the specified task
    (by instruction string match), and yields (image, instruction, action)
    tuples suitable for OpenVLA's action-prediction training.

    Each item is a single timestep from a demonstration trajectory.
    """

    def __init__(
        self,
        suite_name: str,
        task_idx: int,
        data_root: str,
        image_size: tuple[int, int] = (224, 224),
        image_aug: bool = True,
    ):
        self.suite_name = suite_name
        self.task_idx = task_idx
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.image_aug = image_aug

        task_order = get_libero_task_order(suite_name)
        self.task_instruction = task_order[task_idx]
        self.dataset_name = SUITE_TO_DATASET[suite_name]

        # Load and filter the RLDS dataset for this task
        self.samples = self._load_task_data()

    def _load_task_data(self) -> list[dict[str, Any]]:
        """Load RLDS dataset and extract timesteps for this task.

        Returns a flat list of dicts, each containing:
            - image: np.ndarray (H, W, 3), uint8
            - instruction: str
            - action: np.ndarray (7,), float32 (7-DoF: dx,dy,dz,drx,dry,drz,gripper)
            - proprio: np.ndarray (7,), float32 (joint positions)
        """
        dataset_path = self.data_root / self.dataset_name

        # Try loading via tensorflow_datasets (RLDS format)
        try:
            import tensorflow_datasets as tfds
            builder = tfds.builder_from_directory(str(dataset_path))
            ds = builder.as_dataset(split="train")
        except Exception:
            # Fallback: try loading as a HuggingFace dataset
            try:
                from datasets import load_from_disk
                ds = load_from_disk(str(dataset_path))
            except Exception:
                # Final fallback: load from HuggingFace Hub
                from datasets import load_dataset
                ds = load_dataset(
                    f"openvla/modified_libero_rlds",
                    self.dataset_name,
                    split="train",
                )

        samples = []
        for episode in ds:
            # RLDS episodes contain a "steps" field with timestep data
            steps = episode.get("steps", episode)

            # Check if this episode belongs to our task by matching instruction
            episode_instruction = self._extract_instruction(steps)
            if not self._instruction_matches(episode_instruction):
                continue

            # Extract each timestep as a training sample
            for step in steps:
                sample = self._parse_step(step, episode_instruction)
                if sample is not None:
                    samples.append(sample)

        if not samples:
            raise RuntimeError(
                f"No data found for task {self.task_idx} "
                f"('{self.task_instruction}') in {self.dataset_name}. "
                f"Check data_root={self.data_root} and task instruction matching."
            )

        return samples

    def _extract_instruction(self, steps) -> str:
        """Extract the instruction string from an episode's steps."""
        # RLDS format: instruction is in each step's observation
        if hasattr(steps, "__iter__"):
            for step in steps:
                obs = step.get("observation", step)
                if "natural_language_instruction" in obs:
                    inst = obs["natural_language_instruction"]
                    if hasattr(inst, "numpy"):
                        inst = inst.numpy()
                    if isinstance(inst, bytes):
                        inst = inst.decode("utf-8")
                    return inst.strip().lower()
                # Some formats put instruction at the episode level
                if "language_instruction" in step:
                    inst = step["language_instruction"]
                    if hasattr(inst, "numpy"):
                        inst = inst.numpy()
                    if isinstance(inst, bytes):
                        inst = inst.decode("utf-8")
                    return inst.strip().lower()
        return ""

    def _instruction_matches(self, episode_instruction: str) -> bool:
        """Check if an episode's instruction matches this task."""
        target = self.task_instruction.replace("_", " ").lower()
        episode = episode_instruction.replace("_", " ").lower()
        # Allow fuzzy match: target contained in episode or vice versa
        return target in episode or episode in target

    def _parse_step(
        self, step: dict, instruction: str
    ) -> dict[str, Any] | None:
        """Parse a single RLDS timestep into a training sample."""
        obs = step.get("observation", step)

        # Extract image (primary camera)
        image = None
        for key in ["image", "agentview_rgb", "agentview_image", "rgb"]:
            if key in obs:
                image = obs[key]
                break
        if image is None:
            return None

        # Convert to numpy if needed
        if hasattr(image, "numpy"):
            image = image.numpy()
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        image = np.asarray(image, dtype=np.uint8)

        # Extract action (7-DoF)
        action = step.get("action", None)
        if action is None:
            return None
        if hasattr(action, "numpy"):
            action = action.numpy()
        action = np.asarray(action, dtype=np.float32)

        # Extract proprioceptive state if available
        proprio = None
        for key in ["proprio", "joint_pos", "robot_state", "state"]:
            if key in obs:
                proprio = obs[key]
                break
        if proprio is not None:
            if hasattr(proprio, "numpy"):
                proprio = proprio.numpy()
            proprio = np.asarray(proprio, dtype=np.float32)
        else:
            proprio = np.zeros(7, dtype=np.float32)

        return {
            "image": image,
            "instruction": instruction,
            "action": action,
            "proprio": proprio,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        image = sample["image"].copy()

        # Random crop augmentation (OpenVLA default: 90% crop)
        if self.image_aug:
            image = self._random_crop(image, crop_ratio=0.9)

        # Resize to target size
        image = self._resize(image, self.image_size)

        # Convert to float tensor, normalize to [0, 1]
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0

        return {
            "image": image_tensor,
            "instruction": sample["instruction"],
            "action": torch.from_numpy(sample["action"]),
            "proprio": torch.from_numpy(sample["proprio"]),
        }

    @staticmethod
    def _random_crop(image: np.ndarray, crop_ratio: float = 0.9) -> np.ndarray:
        """Random crop with given ratio, then resize back."""
        h, w = image.shape[:2]
        ch, cw = int(h * crop_ratio), int(w * crop_ratio)
        top = np.random.randint(0, h - ch + 1)
        left = np.random.randint(0, w - cw + 1)
        return image[top : top + ch, left : left + cw]

    @staticmethod
    def _resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """Resize image to (H, W) using PIL for quality."""
        from PIL import Image
        pil_img = Image.fromarray(image)
        pil_img = pil_img.resize((size[1], size[0]), Image.BILINEAR)
        return np.asarray(pil_img)

    def collate(self, batch: list[dict]) -> dict[str, Any]:
        """Collate function for DataLoader.

        Stacks images and actions into batched tensors.
        Instructions are passed as a list of strings (for tokenization
        by the model's processor).
        """
        return {
            "image": torch.stack([b["image"] for b in batch]),
            "instruction": [b["instruction"] for b in batch],
            "action": torch.stack([b["action"] for b in batch]),
            "proprio": torch.stack([b["proprio"] for b in batch]),
        }


# ---------------------------------------------------------------------------
# Action normalization statistics per task
# ---------------------------------------------------------------------------

class ActionNormStats:
    """Per-task action normalization statistics.

    OpenVLA uses per-dataset unnormalization at inference time. In CL mode
    each task needs its own stats, stored at training time and selected by
    task identity at eval time.
    """

    def __init__(self):
        self.stats: dict[int, dict[str, np.ndarray]] = {}

    def compute_from_dataset(self, dataset: LiberoTaskDataset, task_idx: int):
        """Compute action mean/std from a task's training data."""
        actions = np.stack([s["action"] for s in dataset.samples])
        self.stats[task_idx] = {
            "mean": actions.mean(axis=0).astype(np.float32),
            "std": actions.std(axis=0).astype(np.float32),
            "q01": np.quantile(actions, 0.01, axis=0).astype(np.float32),
            "q99": np.quantile(actions, 0.99, axis=0).astype(np.float32),
        }

    def normalize(self, action: np.ndarray, task_idx: int) -> np.ndarray:
        """Normalize an action using this task's statistics."""
        s = self.stats[task_idx]
        return (action - s["mean"]) / np.clip(s["std"], 1e-6, None)

    def unnormalize(self, action: np.ndarray, task_idx: int) -> np.ndarray:
        """Unnormalize a predicted action back to robot space."""
        s = self.stats[task_idx]
        return action * np.clip(s["std"], 1e-6, None) + s["mean"]

    def save(self, path: str):
        """Save stats to disk."""
        import json
        out = {}
        for task_idx, s in self.stats.items():
            out[str(task_idx)] = {
                k: v.tolist() for k, v in s.items()
            }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)

    def load(self, path: str):
        """Load stats from disk."""
        import json
        with open(path) as f:
            data = json.load(f)
        for task_idx_str, s in data.items():
            self.stats[int(task_idx_str)] = {
                k: np.array(v, dtype=np.float32) for k, v in s.items()
            }
