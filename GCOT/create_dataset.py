"""
Clarification: 
(current_image, current_state) go through current_action -> (next_image, next_state)

Usage:
python create_dataset.py main --tag single_policy
python create_dataset.py main --tag multiple_policy
python create_dataset.py main --tag aug_multiple_policy
python create_dataset.py show_data --filepath xxx.json
"""

import ast
import json
import os
import random
import re
from collections import OrderedDict
from typing import Union

import fire
import numpy as np
from pydantic import BaseModel
from tqdm import tqdm

from rlds import bridge_v2_dataset
from utils import describe_move


input_template = (
    "What action should the robot take to achieve the instruction\n"
    "INSTRUCTION: \n{instruction}\n"
    # "Accomplished Actions: [{accomplished_actions}]\n"
    "CURRENT GRIPPER: {gripper_2d}\n"
    # "Current Orientation of the gripper: {gripper_orient}\n"
    # "Current Observation Image:"
)

reason_level_template = "REASONING: {reasoning}\n" "SUBTASK: {goal}\n"

position_level_template = (
    "NEXT GRIPPER: {gripper_2d_next}\n"
    # "Next orientation of the gripper: {gripper_orient_next}\n"
)

movement_level_template = "MOVEMENT:\n{movement}\n"


class RawSample(BaseModel, extra="allow", arbitrary_types_allowed=True):
    sample_dir: str
    instruction: str
    highlevel_plan: Union[str, dict]  # highlevel plan key starts from 1
    segments: list[int]  # segment starts from 1 [1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
    gripper_2d: list[list[int]]
    segment_index_to_image_index: dict = dict()
    full_state: np.ndarray
    action_policy: np.ndarray
    valid: bool = False

    def prepare_segments(self):
        """
        segment index starts from 1 {1: 0, 2: 6, 3: 11, 4: 20, 5: 28, 6: 33, -1: 38}
        the -1 index indicates the last frame index
        """
        count = 0
        init = None
        _segments = []
        for i, oseg in enumerate(self.segments):
            if oseg != init:
                init = oseg
                count += 1
                self.segment_index_to_image_index[count] = i
            _segments.append(count)

        self.segment_index_to_image_index[-1] = len(self.segments) - 1
        self.segments = _segments

    def check_valid(self):
        if self.highlevel_plan == "NA":
            return "no response"

        search_result = re.search(r"\{[\s\S]*\}", self.highlevel_plan)
        if search_result == None:
            return "no dict"
        try:
            match = ast.literal_eval(search_result.group(0))
        except Exception:
            return "no valid dict"

        # Check Format
        for k, v in match.items():
            if len(v) != 2:
                return "wrong format"

        self.highlevel_plan = match

        self.prepare_segments()

        if len(match) != max(self.segments):
            return "wrong segment number"

        self.valid = True

    def get_samples_multiple_policy(self):
        """
        Note: here we use the current plan, instead of the next plan, because the next plan is not executed yet
        """
        samples = []
        accomplished_actions = []
        highlevel_plan = tuple(OrderedDict(self.highlevel_plan).items())
        for plan_index in range(len(highlevel_plan)):
            segment_index, (goal, reason) = highlevel_plan[plan_index]
            if type(segment_index) == str:
                segment_index = int(re.findall(r"\d+", segment_index)[0])

            # the last segment index should use the -1, check self.prepare_segments
            # note: goal_next, reason_next is actually not needed
            if segment_index == len(highlevel_plan):
                segment_index_next, (goal_next, reason_next) = -1, (
                    "End",
                    "The instruction is completed",
                )
            elif plan_index < len(highlevel_plan) - 1:
                segment_index_next, (goal_next, reason_next) = highlevel_plan[
                    plan_index + 1
                ]
                if type(segment_index_next) == str:
                    segment_index_next = int(re.findall(r"\d+", segment_index_next)[0])
            image_index_next = self.segment_index_to_image_index[segment_index_next]

            # user
            image_index = self.segment_index_to_image_index[segment_index]
            image_path = os.path.join(self.sample_dir, f"im_{image_index}.jpg")
            user_input = input_template.format(
                instruction=self.instruction,
                accomplished_actions=" ".join(
                    f"{i + 1}. {act}" for i, act in enumerate(accomplished_actions)
                ),
                gripper_2d=self.get_gripper_position(image_index),
            )
            accomplished_actions.append(goal)

            # assistant
            reason_level = reason_level_template.format(
                reasoning=reason,
                goal=goal,
            )

            position_level = position_level_template.format(
                gripper_2d_next=self.get_gripper_position(image_index_next)
            )

            delta_full_state = self.get_position_change(image_index, image_index_next)
            movement_level = movement_level_template.format(
                movement=describe_move(delta_full_state)
            )

            sample = Sample(
                current_image_path=image_path,
                user=user_input,
                assistant_reason_level=reason_level,
                assistant_position_level=position_level,
                assistant_movement_level=movement_level,
                assistant_action_policy=self.action_policy[
                    image_index:image_index_next
                ].tolist(),
                delta_full_state=delta_full_state.tolist(),
            )
            samples.append(sample.dict())
        return samples

    def get_samples_single_policy(self):
        """
        Note: here we use the current plan, instead of the next plan, because the next plan is not executed yet
        """
        samples = []
        accomplished_actions = []
        highlevel_plan = tuple(OrderedDict(self.highlevel_plan).items())
        for index in range(len(self.segments) - 1):
            plan_index = self.segments[index] - 1
            segment_index, (goal, reason) = highlevel_plan[plan_index]
            if type(segment_index) == str:
                segment_index = int(re.findall(r"\d+", segment_index)[0])

            if segment_index == len(highlevel_plan):
                segment_index_next, (goal_next, reason_next) = -1, (
                    "End",
                    "The instruction is completed",
                )
            elif segment_index < len(highlevel_plan):
                segment_index_next, (goal_next, reason_next) = highlevel_plan[
                    plan_index + 1
                ]
                if type(segment_index_next) == str:
                    segment_index_next = int(re.findall(r"\d+", segment_index_next)[0])

            image_index = index
            image_index_next: int = image_index + 1
            image_index_next_segment = self.segment_index_to_image_index[
                segment_index_next
            ]

            if image_index_next >= len(self.segments):
                continue

            # user
            image_path = os.path.join(self.sample_dir, f"im_{image_index}.jpg")
            user_input = input_template.format(
                instruction=self.instruction,
                accomplished_actions=" ".join(
                    f"{i + 1}. {act}" for i, act in enumerate(accomplished_actions)
                ),
                gripper_2d=self.get_gripper_position(image_index),
            )
            accomplished_actions.append(goal)

            # assistant
            reason_level = reason_level_template.format(
                reasoning=reason,
                goal=goal,
            )
            position_level = position_level_template.format(
                gripper_2d_next=self.get_gripper_position(image_index_next)
                # gripper_2d_next=self.get_gripper_position(image_index_next_segment)
            )
            delta_full_state = self.get_position_change(
                image_index,
                image_index_next,
                # image_index_next_segment,
            )

            movement_level = movement_level_template.format(
                movement=describe_move(delta_full_state)
            )
            sample = Sample(
                current_image_path=image_path,
                user=user_input,
                assistant_reason_level=reason_level,
                assistant_position_level=position_level,
                assistant_movement_level=movement_level,
                assistant_action_policy=self.action_policy[
                    image_index:image_index_next
                ].tolist(),
                delta_full_state=delta_full_state.tolist(),
            )

            samples.append(sample.dict())

        return samples

    def get_samples_aug_multiple_policy(self):
        """
        Note: here we use the current plan, instead of the next plan, because the next plan is not executed yet
        """
        samples = []
        accomplished_actions = []
        highlevel_plan = tuple(OrderedDict(self.highlevel_plan).items())
        for frame_index in range(len(self.segments) - 1):
            plan_index = self.segments[frame_index] - 1
            segment_index, (goal, reason) = highlevel_plan[plan_index]
            if type(segment_index) == str:
                segment_index = int(re.findall(r"\d+", segment_index)[0])

            if segment_index == len(highlevel_plan):
                segment_index_next, (goal_next, reason_next) = -1, (
                    "End",
                    "The instruction is completed",
                )

            elif segment_index < len(highlevel_plan):
                segment_index_next, (goal_next, reason_next) = highlevel_plan[
                    plan_index + 1
                ]
                if type(segment_index_next) == str:
                    segment_index_next = int(re.findall(r"\d+", segment_index_next)[0])

            image_index_next = self.segment_index_to_image_index[segment_index_next]

            # user
            image_index = frame_index
            image_path = os.path.join(self.sample_dir, f"im_{image_index}.jpg")
            user_input = input_template.format(
                instruction=self.instruction,
                accomplished_actions=" ".join(
                    f"{i + 1}. {act}" for i, act in enumerate(accomplished_actions)
                ),
                gripper_2d=self.get_gripper_position(image_index),
            )
            accomplished_actions.append(goal)

            # assistant
            reason_level = reason_level_template.format(reasoning=reason, goal=goal)
            position_level = position_level_template.format(
                gripper_2d_next=self.get_gripper_position(image_index_next)
            )
            delta_full_state = self.get_position_change(image_index, image_index_next)

            movement_level = movement_level_template.format(
                movement=describe_move(delta_full_state)
            )
            assert image_index < image_index_next
            sample = Sample(
                current_image_path=image_path,
                user=user_input,
                assistant_reason_level=reason_level,
                assistant_position_level=position_level,
                assistant_movement_level=movement_level,
                assistant_action_policy=self.action_policy[
                    image_index:image_index_next
                ].tolist(),
                delta_full_state=delta_full_state.tolist(),
            )

            samples.append(sample.dict())

        return samples

    def get_position_change(self, image_index, image_index_next):
        delta_xyz_state = (
            self.full_state[image_index_next][:3] - self.full_state[image_index][:3]
        )
        rotat_state = self.full_state[image_index_next][3:6]
        gripper_action = self.action_policy[image_index_next][6]

        # only xyz have delta
        delta_full_state = np.concatenate(
            (delta_xyz_state, rotat_state, [gripper_action])
        )
        return delta_full_state

    def get_gripper_position(self, index):
        gripper2d = self.gripper_2d[index]
        rescaling = 224 / 256
        return [int(gripper2d[0] * rescaling), int(gripper2d[1] * rescaling)]


class Sample(BaseModel):  # one trajectory
    current_image_path: str  # multiple image path containing multiple camera angles
    user: str
    assistant_reason_level: str
    assistant_position_level: str
    assistant_movement_level: str
    assistant_action_policy: list
    delta_full_state: list
    delta_full_state_norm: list = []


def normalize_movement(tag, samples: Sample, overwrite=False):
    all_movements = []
    for sample in samples:
        all_movements.append(sample["delta_full_state"])
    all_movements = np.array(all_movements)
    mean = np.mean(all_movements, axis=0)
    std = np.std(all_movements, axis=0)
    low = np.percentile(all_movements, 1, axis=0)
    high = np.percentile(all_movements, 99, axis=0)

    percentiles = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "Q1": low.tolist(),  # Convert NumPy array to list for JSON compatibility
        "Q99": high.tolist(),
    }
    with open(f"dataset/{tag}/dataset_statistics.json", "w") as f:
        json.dump(percentiles, f, indent=4)

    all_movements_norm = 2 * (all_movements - low) / (high - low + 1e-8) - 1
    all_movements_norm = np.clip(all_movements_norm, -1, 1)

    for i, sample in enumerate(samples):
        sample["delta_full_state_norm"] = all_movements_norm[i].tolist()
        if overwrite:
            sample["assistant_movement_level"] = movement_level_template.format(
                movement=describe_move(sample.delta_full_state)
            )
    return samples


def create_dataset(gripper_position, split, tag="multiple_policy"):
    os.makedirs(f"dataset/{tag}", exist_ok=True)
    with open(f"plans/plans_{split}.json") as f:
        bridge_plans = json.load(f)
    print(f"you choose the tag: {tag}")

    dataset, dataset_len, statistics = bridge_v2_dataset(split=split)

    sample_list = []
    num_traj = 0
    for sample in tqdm(dataset.iterator(), total=dataset_len):
        file_path = sample["file_path"][0].decode("utf-8")
        episode_id = str(sample["episode_id"][0])
        key = file_path + "|" + episode_id
        if key in bridge_plans:
            instruction, segments, model_output = bridge_plans[key]
        else:
            print("notfound plans")
            continue
        full_state = sample["observation"]["proprio"][:, 0, [0, 1, 2, 3, 4, 5, 7]]
        gripper_2d = gripper_position.get(key, None)
        if gripper_2d is None:
            print("notfound gripper2d")
            continue
        action_policy = sample["action"][:, 0, :]
        raw_sample = RawSample(
            sample_dir=key,
            instruction=instruction,
            highlevel_plan=model_output,
            segments=segments,
            gripper_2d=gripper_2d,
            full_state=full_state,
            action_policy=action_policy,
        )

        raw_sample.check_valid()
        if not raw_sample.valid:
            continue

        tag = tag.replace("_", "")
        # This condition order is important, do not change
        if "single_policy".replace("_", "") in tag:
            sample_list.extend(raw_sample.get_samples_single_policy())
        elif "aug_multiple_policy".replace("_", "") in tag:
            sample_list.extend(raw_sample.get_samples_aug_multiple_policy())
        elif "multiple_policy".replace("_", "") in tag:
            sample_list.extend(raw_sample.get_samples_multiple_policy())
        else:
            raise AssertionError("wrong tag!")
        num_traj += 1
    print(split, "num_traj", num_traj, "num_dataset", len(sample_list))
    return sample_list


def main(tag="aug_multiple_policy"):
    """
    Tag: [single_policy, multiple_policy, aug_multiple_policy]
    """
    with open("dataset/embodied_features_bridge.json") as f:
        ecot_dataset = json.load(f)
    gripper_position = dict()
    for file_path, values in ecot_dataset.items():
        for episode_id, stats in values.items():
            gripper_position[file_path + "|" + episode_id] = stats["features"][
                "gripper_position"
            ]

    train_sample_list = create_dataset(
        gripper_position=gripper_position,
        split="train",
        tag=tag,
    )
    val_sample_list = create_dataset(
        gripper_position=gripper_position,
        split="val",
        tag=tag,
    )
    sample_list = train_sample_list + val_sample_list
    save_file_path = f"dataset/{tag}/second_version.json"
    # use unnormalized movement
    sample_list = normalize_movement(tag, sample_list, overwrite=False)
    # use normalized movement
    # sample_list = normalize_movement(tag, sample_list, overwrite=True)
    with open(save_file_path, "w") as f:
        json.dump(sample_list, f, indent=4)
    show_data(save_file_path)
    # filter_dataset(filepath)


def show_data(filepath):
    data = json.load(open(filepath))
    print("reading data from: ", filepath)
    print("dataset len", len(data))
    print(json.dumps(data[1], indent=4))
    num_frame_per_segment = []
    for o in data:
        num_frame_per_segment.append(len(o["assistant_action_policy"]))
    print("num_frame_per_segment", np.mean(num_frame_per_segment))
    breakpoint()


def filter_dataset(path):
    sampled = []
    val_ds = json.load(open(path))
    # random.shuffle(sampled)
    for i in tqdm(range(0, len(val_ds))):
        item = val_ds[i]
        if (
            "pick" in item["user"]
            or "place" in item["user"]
            or "put" in item["user"]
            or "wipe" in item["user"]
        ):
            sampled.append(item)
    print(json.dumps(sampled[1], indent=4))
    print("after sampled the length is: ", len(sampled))

    with open(
        "./dataset/sampled_w_gripper_position_train_aug_multiple_policy.json",
        "w",
    ) as f:
        json.dump(sampled, f, indent=4)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    fire.Fire()


"""
 ** DATE: 18 Nov
## multiple policy
 train num_traj 38453 num_dataset 181788
 val num_traj 5113 num_dataset 24193
 merged num_traj 43566 num_dataset 205981
 num_segment_per_traj = num_dataset / num_traj 4.7
 num_frame_per_segment 6.939601225355736


## single policy
 train num_traj 38453 num_dataset 1262244
 val num_traj 5113 num_dataset 167182
 merged num_traj 43566 num_dataset 1429426
 num_dataset / num_traj 32.8
 num_frame_per_segment 1.0


## aug multiple policy
 val num_traj 5113 num_dataset 167182
 train num_traj 38453 num_dataset 1262244
 merged num_traj 43566 num_dataset 1429426
 num_dataset / num_traj 32.8
 num_frame_per_segment 5.457
"""
