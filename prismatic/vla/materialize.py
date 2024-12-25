"""
materialize.py

Factory class for initializing Open-X RLDS-backed datasets, given specified data mixture parameters; provides and
exports individual functions for clear control flow.
"""

from pathlib import Path
from typing import Tuple, Type

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import PaddedCollatorForActionPrediction, ValPaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import EpisodicRLDSDataset, FastDatasetDiscrete, RLDSBatchTransform, RLDSDataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


def get_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
    predict_stop_token: bool = True,
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    episodic: bool = False,
    image_aug: bool = False,
) -> Tuple[Dataset, ActionTokenizer, PaddedCollatorForActionPrediction]:
    """Initialize RLDS Dataset (wraps TFDS), ActionTokenizer, and initialize transform/collation functions."""
    action_tokenizer = ActionTokenizer(tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer, tokenizer, image_transform, prompt_builder_fn, predict_stop_token=predict_stop_token
    )
    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length, tokenizer.pad_token_id, padding_side=padding_side
    )
    # Build RLDS Iterable Dataset
    cls = RLDSDataset if not episodic else EpisodicRLDSDataset
    dataset = cls(
        data_root_dir,
        data_mix,
        batch_transform,
        resize_resolution=default_image_resolution[1:],
        shuffle_buffer_size=shuffle_buffer_size,
        train=train,
        image_aug=image_aug,
    )

    return dataset, action_tokenizer, collator


def get_discrete_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
    predict_stop_token: bool = True,
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    episodic: bool = False,
    image_aug: bool = False,
):
    # action_tokenizer = ActionTokenizer(tokenizer)
    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length, tokenizer.pad_token_id, padding_side=padding_side
    )

    val_collator = ValPaddedCollatorForActionPrediction(
        tokenizer.model_max_length, tokenizer.pad_token_id, padding_side="left"
    )
    action_tokenizer = ActionTokenizer(tokenizer)
    dataset_noiter = FastDatasetDiscrete(
        data_root_dir=data_root_dir,
        action_tokenizer=action_tokenizer,
        base_tokenizer=tokenizer,
        image_transform=image_transform,
        prompt_builder_fn=prompt_builder_fn,
        file_name="second_version.json",
        data_mix=data_mix,
        mask_inst=True,
    )
    # val_dataset_noiter = FastDatasetDiscrete(
    #     data_root_dir=data_root_dir,
    #     action_tokenizer=action_tokenizer,
    #     base_tokenizer=tokenizer,
    #     image_transform=image_transform,
    #     prompt_builder_fn=prompt_builder_fn,
    #     file_name="first_version_val.json",
    #     data_mix=data_mix,
    # )
    val_dataset_noiter = dataset_noiter
    print("Shuffled Val, Train lengths:", len(val_dataset_noiter), len(dataset_noiter))
    # dataset = DiscreteIterableDataset(dataset=dataset_noiter)
    # val_dataset = DiscreteIterableDataset(dataset=val_dataset_noiter)
    return dataset_noiter, val_dataset_noiter, collator, val_collator, action_tokenizer
