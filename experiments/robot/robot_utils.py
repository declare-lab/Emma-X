"""Utils for evaluating robot policies in various environments."""

import time

import numpy as np
import torch
from experiments.robot.openvla_utils import get_seq_action
from prismatic import load_vla
from experiments.robot.openvla_utils import (
    get_vla,
    get_vla_action,
)

from pathlib import Path


# get_vla  # get_vla_action,

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_model(cfg, proprio_norm_stats, wrap_diffusion_policy_for_droid=False):
    """Load model for evaluation."""
    if cfg.model_family == "openvla":
        model = get_vla(cfg)
    elif cfg.model_family == "pred-all":
        hf_token = Path(".hf_token").read_text().strip()
        model = load_vla(model_id_or_path=cfg.model_pretrained_checkpoint, hf_token=hf_token, proprio_norm_stats=proprio_norm_stats)
        device = torch.device("cuda")
        # model.to(device, dtype=torch.bfloat16)
        model.to(device, dtype=torch.float16)
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    print(f"Loaded model: {type(model)}")
    return model


def get_image_resize_size(cfg):
    """
    Gets image resize size for a model class.
    If `resize_size` is an int, then the resized image will be a square.
    Else, the image will be a rectangle.
    """
    if cfg.model_family == "openvla":
        resize_size = 224
    elif cfg.model_family == "pred-all":
        resize_size = 224
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return resize_size


def get_action(cfg, model, obs, task_label, processor=None, type='act'):
    """Queries the model to get an action."""
    if cfg.model_family == "openvla":
        action = get_vla_action(
            model, processor, cfg.pretrained_checkpoint, obs, task_label, cfg.unnorm_key, center_crop=cfg.center_crop
        )
        assert action.shape == (ACTION_DIM,)
        return [action], None
    elif cfg.model_family == "pred-all":
        assert type in ['pos', 'act']
        delta, generated_text = get_seq_action(
            model, processor, cfg.pretrained_checkpoint, obs, task_label, cfg.unnorm_key, type=type, center_crop=cfg.center_crop
        )
        return delta, generated_text
    else:
        raise ValueError("Unexpected `model_family` found in config.")


def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action
