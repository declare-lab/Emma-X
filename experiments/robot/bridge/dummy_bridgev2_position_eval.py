"""
run_bridge_eval.py

Runs a model in a real-world Bridge V2 environment.

Usage:
    # OpenVLA:
    python experiments/robot/bridge/run_bridge_eval.py --model_family openvla --pretrained_checkpoint openvla/openvla-7b
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union

import draccus
import numpy as np
from experiments.robot.bridge.bridgev2_utils import (
    get_next_task_label,
    get_preprocessed_image,
    get_widowx_env,
    refresh_obs,
    save_rollout_data,
    save_rollout_video,
)
from experiments.robot.bridge.tf_transformation import mat_to_pose, pose_to_mat
from experiments.robot.robot_utils import get_image_resize_size


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "pred-all" # openvla"                               # Model family
    pretrained_checkpoint: Union[str, Path] = "openvla/openvla-7b"                # Pretrained checkpoint path
    # model_pretrained_checkpoint: str = "/mnt/data1/emrys/openvla/Logs/prism-dinosiglip-224px+mx-pred-all-2gpu+n0+b16+x7/checkpoints/step-016945-epoch-04-loss=0.0474.pt"
    model_pretrained_checkpoint: str = "/mnt/data1/emrys/openvla/Logs/prism-dinosiglip-224px+mx-single-policy-wposition-4gpu+n0+b16+x7/checkpoints/step-014064-epoch-02-loss=0.0865.pt"
    # model_pretrained_checkpoint: str = "/mnt/data1/emrys/openvla/Logs/prism-dinosiglip-224px+mx-single-policy-withoutall-4gpu+n0+b16+x7/checkpoints/step-021807-epoch-01-loss=0.1959.pt"
    # model_pretrained_checkpoint: str = "/mnt/data1/emrys/openvla/Logs/prism-dinosiglip-224px+mx-single-policy-withall-2gpu+n0+b16+x7/checkpoints/step-015000-epoch-00-loss=0.2072.pt"


    load_in_8bit: bool = False                                  # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                                  # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = False                                   # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # WidowX environment-specific parameters
    #################################################################################################################
    host_ip: str = "172.17.240.173"
    # host_ip: str = "localhost"
    port: int = 5556

    # Note: Setting initial orientation with a 30 degree offset, which makes the robot appear more natural
    init_ee_pos: List[float] = field(default_factory=lambda: [0.3, -0.09, 0.26])
    init_ee_quat: List[float] = field(default_factory=lambda: [0, -0.259, 0, -0.966])
    bounds: List[List[float]] = field(default_factory=lambda: [
            [0.1, -0.20, -0.01, -1.57, 0],
            [0.45, 0.25, 0.30, 1.57, 0],
        ]
    )

    camera_topics: List[Dict[str, str]] = field(default_factory=lambda: [{"name": "/blue/image_raw"}])

    blocking: bool = False                                      # Whether to use blocking control
    max_episodes: int = 50                                      # Max number of episodes to run
    max_steps: int = 60                                         # Max number of timesteps per episode
    control_frequency: float = 5                                # WidowX control frequency

    #################################################################################################################
    # Utils
    #################################################################################################################
    save_data: bool = True                                     # Whether to save rollout data (images, actions, etc.)

    # fmt: on


@draccus.wrap()
def eval_model_in_bridge_env(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    assert not cfg.center_crop, "`center_crop` should be disabled for Bridge evaluations!"

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = "bridge_orig"

    # Initialize the WidowX environment
    env = get_widowx_env(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    task_label = ""
    episode_idx = 0
    while episode_idx < cfg.max_episodes:
        # Get task description from user
        task_label = get_next_task_label(task_label)

        # Reset environment
        obs, _ = env.reset()

        # Setup
        t = 0
        step_duration = 1.0 / cfg.control_frequency
        replay_images = []
        if cfg.save_data:
            rollout_images = []
            rollout_states = []
            rollout_actions = []

        # Start episode
        input(f"Press Enter to start episode {episode_idx+1}...")
        print("Starting episode... Press Ctrl-C to terminate episode early!")
        last_tstamp = time.time()
        while t < cfg.max_steps:
            # try:
            curr_tstamp = time.time()
            if curr_tstamp > last_tstamp + step_duration:
                print(f"t: {t}")
                print(f"Previous step elapsed time (sec): {curr_tstamp - last_tstamp:.2f}")
                last_tstamp = time.time()

                # Refresh the camera image and proprioceptive state
                obs = refresh_obs(obs, env)

                # Save full (not preprocessed) image for replay video
                replay_images.append(obs["full_image"])

                # Get preprocessed image for gripper size
                obs["full_image"] = get_preprocessed_image(obs, resize_size)
                delta_position = input("input 6 list position:")
                delta_position = eval(delta_position)
                # very important step here
                delta_position = np.array(delta_position)
                current_position = mat_to_pose(obs["eef_transform"])
                goal_mat = pose_to_mat(delta_position + current_position)

                if cfg.save_data:
                    rollout_images.append(obs["full_image"])
                    rollout_states.append(obs["proprio"])
                    # rollout_actions.append(goal_position)

                obs, _, _, _, _ = env.move(goal_mat, 1, duration=5)

                t += 1
                time.sleep(0.1)

            # except (KeyboardInterrupt, Exception) as e:
            #     if isinstance(e, KeyboardInterrupt):
            #         print("\nCaught KeyboardInterrupt: Terminating episode early.")
            #     else:
            #         print(f"\nCaught exception: {e}")
            #     break

        # Save a replay video of the episode
        save_rollout_video(replay_images, episode_idx)

        # [If saving rollout data] Save rollout data
        if cfg.save_data:
            # save_image_data(replay_images, rollout_images, rollout_states, rollout_actions, idx=episode_idx)
            save_rollout_data(replay_images, rollout_images, rollout_states, rollout_actions, idx=episode_idx)

        # Redo episode or continue
        if input("Enter 'r' if you want to redo the episode, or press Enter to continue: ") != "r":
            episode_idx += 1


if __name__ == "__main__":
    eval_model_in_bridge_env()
