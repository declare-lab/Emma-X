"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType
from prismatic.vla.datasets.rlds_dataset import bridge_v2_dataset
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase


# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        # TODO: Edit here
        labels[: -(len(action) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, dataset_name=dataset_name)


class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),
            load_depth=False,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=0,                        # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)


class DummyDatasetDiscrete(Dataset):
    def __init__(
        self,
        # action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        # self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        self.dataset_statistics = {
            "dummy_discrete_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return 10000

    def generate_movement_list(self, num_instructions):
        rand_num = random.random()
        if rand_num < 0.1:
            return [("gripper", 0)]
        if rand_num < 0.2:
            return [("gripper", 1)]
        directions = ["left", "right", "forward", "backward", "up", "down"]

        movement_list = []

        for _ in range(num_instructions):
            direction = random.choice(directions)
            steps = random.randint(1, 5)
            movement_list.append((f"move {direction}", steps))

        return movement_list

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        # action = np.asarray(np.random.rand(7), dtype=np.float32)
        action_list = self.generate_movement_list(random.randint(1, 4))

        instruction = "put the object at the destination"
        gpt_output = f"plan: \n1. Go to object \n2. Pick up object\n\nCurrent Action Plan:\n{action_list}"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"Instruction: {instruction}?"},
            # {"from": "gpt", "value": self.action_tokenizer(action)},
            {"from": "gpt", "value": gpt_output},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        # labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)


class FastDatasetDiscrete_31_sep(Dataset):
    def __init__(
        self,
        data_root_dir: Path,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
        file_name: str = "",
        data_mix: str = "",
        mask_inst: bool = False,
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.mask_inst = mask_inst
        self.prompt_display_count = 2
        self.data_mix = data_mix
        if "train" in file_name:
            self.split = "train"
        elif "val" in file_name:
            self.split = "val"
        elif "test" in file_name:
            self.split = "test"
        else:
            assert False
        with open(f"{data_root_dir}/{file_name}", "r") as file:
            self.data = json.load(file)
        self.dataset_statistics = {
            "fast_dataset": {
                # TODO: Update this
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image = Image.open(item["current_image_path"]).resize((256, 256), Image.Resampling.LANCZOS)
        pixel_values = self.image_transform(image)

        if self.data_mix == "movement_only":
            prompt_str = item["user"] + "\n" + item["assistant_reason_level"] + item["assistant_position_level"]
            gpt_output = item["assistant_movement_level"]
        elif self.data_mix == "reason_position_movement":
            prompt_str = item["user"]
            gpt_output = (
                item["assistant_reason_level"]
                + "\n"
                + item["assistant_position_level"]
                + "\n"
                + item["assistant_movement_level"]
            )
        elif self.data_mix == "action_policy_only":
            prompt_str = item["user"]
            action_policies = self.action_tokenizer(item["assistant_action_policy"])
            gpt_output = "Next action policies:\n" + ";".join(action_policies)
        elif self.data_mix == "pred_all":
            prompt_str = item["user"]
            action_policies = self.action_tokenizer(item["assistant_action_policy"])
            gpt_output = (
                item["assistant_reason_level"]
                + "\n"
                + item["assistant_position_level"]
                + "\n"
                + item["assistant_movement_level"]
                + "\n"
                + "Next action policies:\n"
                + ";".join(action_policies)
                + "\n"
            )

        prompt_builder = self.prompt_builder_fn("openvla")
        if self.split == "train":
            conversation = [
                {"from": "human", "value": prompt_str},
                {"from": "gpt", "value": gpt_output},
            ]
        else:
            conversation = [
                {"from": "human", "value": prompt_str},
                {"from": "gpt", "value": ""},
            ]

        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        if self.split == "train":
            labels = list(input_ids)
            # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
            #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
            input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
            if self.mask_inst:
                label_prompt_builder = self.prompt_builder_fn("openvla")
                label_conversation = [
                    {"from": "human", "value": prompt_str},
                    {"from": "gpt", "value": ""},
                ]
                for turn in label_conversation:
                    label_prompt_builder.add_turn(turn["from"], turn["value"])

                label_input_ids = self.base_tokenizer(
                    label_prompt_builder.get_prompt(), add_special_tokens=True
                ).input_ids
                labels[: len(label_input_ids) - 2] = IGNORE_INDEX

        else:
            labels = gpt_output
            input_ids = torch.tensor(input_ids)[:-2]

        if self.prompt_display_count > 0:
            print(prompt_builder.get_prompt())
            self.prompt_display_count -= 1
            print(labels)
            print(input_ids)

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)


class FastDatasetDiscrete(Dataset):
    def __init__(
        self,
        data_root_dir: Path,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
        file_name: str = "",
        data_mix: str = "",
        mask_inst: bool = False,
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.mask_inst = mask_inst
        self.prompt_display_count = 2
        self.data_mix = data_mix
        self.split = "train"
        dataset_train, dataset_len, self.dataset_statistics = bridge_v2_dataset(split="train")
        dataset_val, dataset_len, self.dataset_statistics = bridge_v2_dataset(split="val")
        self.build_dataset(dataset_train, dataset_val)
        self.data_root_dir = data_root_dir

        with open(f"{data_root_dir}/{file_name}", "r") as file:
            self.data = json.load(file)

    def build_dataset(self, dataset_train, dataset_val):
        print("Finished loading dataset in TF")
        self.info = dict()
        for i, traj in enumerate(dataset_train.as_numpy_iterator()):
            key = traj["file_path"][0].decode("utf-8") + "|" + str(traj["episode_id"][0])
            self.info[key] = traj
        for i, traj in enumerate(dataset_val.as_numpy_iterator()):
            key = traj["file_path"][0].decode("utf-8") + "|" + str(traj["episode_id"][0])
            self.info[key] = traj
        print("Finish building dataset")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        policy_prefix = "POLICIES:\n"
        movment_prefix = "MOVEMENT:\n"
        item = self.data[idx]
        key = "/".join(item["current_image_path"].split("/")[:-1])
        current_image_index = int(item["current_image_path"].split("/")[-1].split("_")[-1].split(".")[0])
        info = self.info[key]
        image = tf.io.decode_image(
            info["observation"]["image_primary"][current_image_index][0], expand_animations=False, dtype=tf.uint8
        ).numpy()
        image = Image.fromarray(image)
        pixel_values = self.image_transform(image)

        if self.data_mix == "movement_only":
            prompt_str = item["user"] + "\n" + item["assistant_reason_level"] + item["assistant_position_level"]
            gpt_output = item["assistant_movement_level"]
        elif self.data_mix == "reason_position_movement":
            prompt_str = item["user"]
            gpt_output = (
                item["assistant_reason_level"]
                + "\n"
                + item["assistant_position_level"]
                + "\n"
                + item["assistant_movement_level"]
            )
        elif self.data_mix == "action_policy_only":  # for multiple policies only
            prompt_str = item["user"]
            action_policies = self.action_tokenizer(item["assistant_action_policy"])
            gpt_output = policy_prefix + ";".join(action_policies) + "\n"
        elif self.data_mix == "openvla":  # for single policy only
            prompt_str = item["user"].split("CURRENT GRIPPER")[0]
            action_policies = self.action_tokenizer(item["assistant_action_policy"])
            gpt_output = policy_prefix + ";".join(action_policies) + "\n"
        elif self.data_mix == "pred_all":
            prompt_str = item["user"]
            action_policies = self.action_tokenizer(item["assistant_action_policy"])
            if "norm" in self.data_root_dir.name:
                movement = movment_prefix + self.action_tokenizer(item["delta_full_state_norm"])
            else:
                movement = item["assistant_movement_level"]
            gpt_output = (
                item["assistant_reason_level"]
                + "\n"
                + item["assistant_position_level"]
                + "\n"
                + movement
                + "\n"
                + policy_prefix
                + ";".join(action_policies)
                + "\n"
            )
        elif self.data_mix == "no_movement":
            prompt_str = item["user"]
            action_policies = self.action_tokenizer(item["assistant_action_policy"])
            gpt_output = (
                item["assistant_reason_level"]
                + "\n"
                + item["assistant_position_level"]
                + "\n"
                + policy_prefix
                + ";".join(action_policies)
                + "\n"
            )
        elif self.data_mix == "movement_policy":
            prompt_str = item["user"]
            action_policies = self.action_tokenizer(item["assistant_action_policy"])
            movement = item["assistant_movement_level"]
            gpt_output = (
                movement
                + "\n"
                + policy_prefix
                + ";".join(action_policies)
                + "\n"
            )
        elif self.data_mix == "no_position":
            prompt_str = item["user"]
            action_policies = self.action_tokenizer(item["assistant_action_policy"])
            movement = item["assistant_movement_level"]
            gpt_output = (
                item["assistant_reason_level"]
                + "\n"
                + movement
                + "\n"
                + policy_prefix
                + ";".join(action_policies)
                + "\n"
            )
        elif self.data_mix == "no_reason":
            prompt_str = item["user"]
            action_policies = self.action_tokenizer(item["assistant_action_policy"])
            movement = item["assistant_movement_level"]
            gpt_output = (
                item["assistant_position_level"]
                + "\n"
                + movement
                + "\n"
                + policy_prefix
                + ";".join(action_policies)
                + "\n"
            )
        prompt_builder = self.prompt_builder_fn("openvla")
        if self.split == "train":
            conversation = [
                {"from": "human", "value": prompt_str},
                {"from": "gpt", "value": gpt_output},
            ]
        else:
            conversation = [
                {"from": "human", "value": prompt_str},
                {"from": "gpt", "value": ""},
            ]

        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        if self.split == "train":
            labels = list(input_ids)
            # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
            #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
            input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
            if self.mask_inst:
                label_prompt_builder = self.prompt_builder_fn("openvla")
                label_conversation = [
                    {"from": "human", "value": prompt_str},
                    {"from": "gpt", "value": ""},
                ]
                for turn in label_conversation:
                    label_prompt_builder.add_turn(turn["from"], turn["value"])

                label_input_ids = self.base_tokenizer(
                    label_prompt_builder.get_prompt(), add_special_tokens=True
                ).input_ids
                labels[: len(label_input_ids) - 2] = IGNORE_INDEX

        else:
            labels = gpt_output
            input_ids = torch.tensor(input_ids)[:-2]

        if self.prompt_display_count > 0:
            print(prompt_builder.get_prompt())
            self.prompt_display_count -= 1
            print(labels)
            print(input_ids)

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
