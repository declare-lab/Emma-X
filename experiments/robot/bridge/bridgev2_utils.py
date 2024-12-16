"""Utils for evaluating policies in real-world BridgeData V2 environments."""

import json
import os
import re
import time

import imageio
import numpy as np
import tensorflow as tf
import torch
from experiments.robot.bridge.widowx_env import WidowXGym
from PIL import Image
from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs


# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
BRIDGE_PROPRIO_DIM = 7
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})


def get_widowx_env_params(cfg):
    """Gets (mostly default) environment parameters for the WidowX environment."""
    env_params = WidowXConfigs.DefaultEnvParams.copy()
    env_params["override_workspace_boundaries"] = cfg.bounds
    env_params["camera_topics"] = cfg.camera_topics
    env_params["return_full_image"] = True
    return env_params


def get_widowx_env(cfg, model=None):
    """Get WidowX control environment."""
    # Set up the WidowX environment parameters
    env_params = get_widowx_env_params(cfg)
    start_state = np.concatenate([cfg.init_ee_pos, cfg.init_ee_quat])
    env_params["start_state"] = list(start_state)
    # Set up the WidowX client
    widowx_client = WidowXClient(host=cfg.host_ip, port=cfg.port)
    widowx_client.init(env_params)
    env = WidowXGym(
        widowx_client,
        cfg=cfg,
        blocking=cfg.blocking,
    )
    return env


def get_next_task_label(task_label):
    """Prompt the user to input the next task."""
    if task_label == "":
        user_input = ""
        while user_input == "":
            user_input = input("Enter the task name: ")
        task_label = user_input
    else:
        user_input = input("Enter the task name (or leave blank to repeat the previous task): ")
        if user_input == "":
            pass  # Do nothing -> Let task_label be the same
        else:
            task_label = user_input
    print(f"Task: {task_label}")
    return task_label


def get_next_save_name(save_name):
    """Prompt the user to input the next save name."""

    def increment_round_number(text):
        pattern = r"(r)(\d+)"

        def replace_func(match):
            number = int(match.group(2))
            return f"{match.group(1)}{number + 1}"

        return re.sub(pattern, replace_func, text)

    save_name = increment_round_number(save_name)
    if save_name == "":
        user_input = ""
        while user_input == "":
            user_input = input("Enter the save name: ")
        save_name = user_input
    else:
        user_input = input(f"Enter the save name (or leave blank to use {save_name}): ")
        if user_input == "":
            pass  # Do nothing -> Let save_name be the same
        else:
            save_name = user_input
    print(f"Save name: {save_name}")
    return save_name


def save_rollout_video(rollout_images, save_name):
    """Saves an MP4 replay of an episode."""
    os.makedirs("./rollouts", exist_ok=True)
    if save_name == "":
        mp4_path = f"./rollouts/rollout-{DATE_TIME}.mp4"
    else:
        mp4_path = f"./rollouts/{save_name}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=5)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")


def save_rollout_text(generated_texts, save_name):
    os.makedirs("./rollouts", exist_ok=True)
    if save_name == "":
        text_path = f"./rollouts/rollout-{DATE_TIME}.json"
    else:
        text_path = f"./rollouts/{save_name}.json"
    with open(text_path, "w") as f:
        json.dump(generated_texts, f, indent=4)
    print(f"Saved rollout text at path {text_path}")


def save_rollout_data(rollout_orig_images, rollout_images, rollout_states, rollout_actions, idx):
    """
    Saves rollout data from an episode.

    Args:
        rollout_orig_images (list): Original rollout images (before preprocessing).
        rollout_images (list): Preprocessed images.
        rollout_states (list): Proprioceptive states.
        rollout_actions (list): Predicted actions.
        idx (int): Episode index.
    """
    os.makedirs("./rollouts", exist_ok=True)
    path = f"./rollouts/rollout-{DATE_TIME}-{idx+1}.npz"
    # Convert lists to numpy arrays
    orig_images_array = np.array(rollout_orig_images)
    images_array = np.array(rollout_images)
    states_array = np.array(rollout_states)
    actions_array = np.array(rollout_actions)
    # Save to a single .npz file
    np.savez(path, orig_images=orig_images_array, images=images_array, states=states_array, actions=actions_array)
    print(f"Saved rollout data at path {path}")


def save_image_data(rollout_orig_images, rollout_images, rollout_states, rollout_actions, idx):
    os.makedirs("./rollouts", exist_ok=True)

    for o in rollout_orig_images:
        img = Image.fromarray(o)
        img.save("./rollouts/img.png")


def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img


def get_preprocessed_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    obs["full_image"] = resize_image(obs["full_image"], resize_size)
    return obs["full_image"]


def refresh_obs(obs, env):
    """Fetches new observations from the environment and updates the current observations."""
    new_obs = env.get_observation()
    obs["full_image"] = new_obs["full_image"]
    obs["image_primary"] = new_obs["image_primary"]
    obs["proprio"] = new_obs["proprio"]
    if "eef_transform" in new_obs:
        obs["eef_transform"] = new_obs["eef_transform"]
    return obs
