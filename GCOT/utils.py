import math
import os
import pickle
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mediapy
import numpy as np
import tensorflow as tf
from PIL import Image
from scipy.spatial.distance import euclidean
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity


def segment_traj(full_state, distance="euclidean", time_weight=1):

    def process_traj(segments):
        _segments = []
        previous = 0
        for seg in segments:
            if seg != -1:
                previous = seg
            _segments.append(previous)
        return _segments

    def spatio_temporal_distance(point1, point2):
        """
        Calculate a combined Euclidean and temporal distance between two points.
        point1, point2: Arrays or lists containing [x, y, z, time]
        time_weight: Factor that adjusts the influence of the time difference.
        """
        if distance == "euclidean":
            spatial_distance = euclidean(point1[:-1], point2[:-1])
        elif distance == "cosine":
            spatial_distance = cosine_similarity([point1[:-1]], [point2[:-1]])
        temporal_distance = time_weight * abs(point1[-1] - point2[-1])
        return spatial_distance + temporal_distance

    clustering = HDBSCAN(min_cluster_size=3, metric=spatio_temporal_distance)
    # x = [np.append(o, i / len(full_state)) for i, o in enumerate(full_state)]
    x = [np.append(o, i / 30) for i, o in enumerate(full_state)]
    segs = clustering.fit_predict(x)
    processed_segs = process_traj(segs)
    return processed_segs, segs


def segment_gripper(gripper_state):
    """
    Gripper State should only contain 0 or 1
    """
    previous_index = 0
    _segments = []
    for i, state in enumerate(gripper_state):
        # use round here because state have a slight chance(<0.01) of not being 0 or 1
        if round(state) != round(gripper_state[previous_index]):
            previous_index = i
        _segments.append(previous_index)
    return _segments


def get_soft_plus_gripper_segment(sample):
    """
    Design choices:
    1. delta action 171337
    2. delta state  45634
    3. action       92099
    4. state        42332
    """

    def get_delta(states):
        delta_states = [states[i] - states[i - 1] for i in range(1, len(states))]
        return [
            delta_states[0]
        ] + delta_states  # add first element to make it same length as states

    instruction = sample["task"]["language_instruction"][0].decode("utf-8")
    images = sample["observation"]["image_primary"]
    images = [
        tf.io.decode_image(
            images[i][0], expand_animations=False, dtype=tf.uint8
        ).numpy()
        for i in range(len(images))
    ]
    full_state = sample["observation"]["proprio"][:, 0, [0, 1, 2, 3, 4, 5]]
    assert sum(sample["observation"]["proprio"][:, 0, 6]) == 0
    # full_state = sample["action"][:, 0, [0, 1, 2, 3, 4, 5]]

    gripper_action = sample["action"][
        :, 0, -1
    ]  # here to use action instead of gripper state
    # because action only take value between 0, 1
    gripper_segment = np.array(segment_gripper(gripper_action))

    # spatial_state = get_delta([o[:3] for o in full_state])
    # orient_state = get_delta([o[3:-1] for o in full_state])
    # spatial_segment = np.array(segment_traj(spatial_state, distance="euclidean"))
    # orient_segment = np.array(segment_traj(orient_state, distance="euclidean"))
    # overall_segment = spatial_segment * 1e4 + orient_segment * 1e2 + gripper_segment

    # eff_pose = get_delta(full_state)
    eff_pose = full_state
    processed_segs, segs = segment_traj(eff_pose, distance="euclidean")
    pose_segment = np.array(processed_segs)
    overall_segment = pose_segment * 1e2 + gripper_segment

    key_frames, segment_count = get_key_frames(images, overall_segment)

    return (instruction, key_frames, segment_count), overall_segment


def get_soft_segment(sample):
    """
    Design choices:
    1. delta action 171337
    2. delta state  45634
    3. action       92099
    4. state        42332
    """

    def get_delta(states):
        delta_states = [states[i] - states[i - 1] for i in range(1, len(states))]
        return [
            delta_states[0]
        ] + delta_states  # add first element to make it same length as states

    instruction = sample["task"]["language_instruction"][0].decode("utf-8")
    images = sample["observation"]["image_primary"]
    images = [
        tf.io.decode_image(
            images[i][0], expand_animations=False, dtype=tf.uint8
        ).numpy()
        for i in range(len(images))
    ]
    full_state = sample["observation"]["proprio"][:, 0, [0, 1, 2, 3, 4, 5]]
    assert sum(sample["observation"]["proprio"][:, 0, 6]) == 0

    eff_pose = full_state
    processed_segs, segs = segment_traj(eff_pose, distance="euclidean")
    pose_segment = np.array(processed_segs)
    overall_segment = pose_segment * 1e2

    key_frames, segment_count = get_key_frames(images, overall_segment)

    return (instruction, key_frames, segment_count), overall_segment


def get_gripper_segment(sample):
    instruction = sample["task"]["language_instruction"][0].decode("utf-8")
    images = sample["observation"]["image_primary"]
    images = [
        tf.io.decode_image(
            images[i][0], expand_animations=False, dtype=tf.uint8
        ).numpy()
        for i in range(len(images))
    ]
    assert sum(sample["observation"]["proprio"][:, 0, 6]) == 0
    gripper_action = sample["action"][
        :, 0, -1
    ]  # here to use action instead of gripper state
    # because action only take value between 0, 1
    gripper_segment = np.array(segment_gripper(gripper_action))
    key_frames, segment_count = get_key_frames(images, gripper_segment)

    return (instruction, key_frames, segment_count), gripper_segment


def get_nstep_segment(sample, n=5):
    instruction = sample["task"]["language_instruction"][0].decode("utf-8")
    images = sample["observation"]["image_primary"]
    images = [
        tf.io.decode_image(
            images[i][0], expand_animations=False, dtype=tf.uint8
        ).numpy()
        for i in range(len(images))
    ]
    assert sum(sample["observation"]["proprio"][:, 0, 6]) == 0
    gripper_action = sample["action"][
        :, 0, -1
    ]  # here to use action instead of gripper state
    # because action only take value between 0, 1
    nstep_segment = np.repeat(np.arange(100), n)[: len(gripper_action)]
    key_frames, segment_count = get_key_frames(images, nstep_segment)

    return (instruction, key_frames, segment_count), nstep_segment


def get_key_frames(images, overall_segment):
    _images = []
    init = None
    count = 0
    for i, oseg in enumerate(overall_segment):
        if oseg != init:
            init = oseg
            count += 1
            _images.append(f"Segment {count}:")
        _images.append(Image.fromarray(images[i], "RGB"))
    return _images, count


def find_sample_folders(root_folder):
    sample_folders = []

    pattern = re.compile(r"^traj\d+$")

    for dirpath, dirnames, filenames in os.walk(root_folder):
        for dirname in dirnames:
            if pattern.match(dirname):
                sample_folders.append(os.path.join(dirpath, dirname))

    return sample_folders


def extract_image_number(filename):
    filename = str(filename).split("/")[-1]
    return int(filename.split("_")[1].split(".")[0])


def plot_traj(images, gripper_coordinates_2d=None, fps=5):
    if gripper_coordinates_2d is None:
        mediapy.show_video(images, fps=fps)
    elif gripper_coordinates_2d == []:
        mediapy.show_video(images, fps=fps)
    else:
        cmap = plt.get_cmap(
            "viridis"
        )  # You can change 'viridis' to other color maps like 'plasma', 'inferno', etc.

        norm = plt.Normalize(0, len(gripper_coordinates_2d))
        gripper_images = []
        for i, point in enumerate(gripper_coordinates_2d):
            rgb_color = cmap(norm(i))[:3]
            bgr_color = (
                int(rgb_color[2] * 255),
                int(rgb_color[1] * 255),
                int(rgb_color[0] * 255),
            )
            gripper_image = cv2.circle(
                images[i], point, color=bgr_color, radius=5, thickness=-1
            )
            gripper_images.append(gripper_image)
        mediapy.show_video(gripper_images, fps=fps)


def plot_traj_segments(
    images,
    full_state,
    action_policy=None,
    segments=None,
    gripper_coordinates_2d=None,
):
    if segments is None:
        segments = [0] * len(images)
    init_seg = segments[0]
    _images = []
    _gripper_coordinates_2d = [] if gripper_coordinates_2d else None
    init_state = full_state[0]
    if action_policy is not None:
        _action_policy = []
        _full_state = []
    for i, seg in enumerate(segments):
        if seg != init_seg:
            print(f"Segment: {init_seg}")
            print(describe_move(full_state[i] - init_state))
            if action_policy is not None:
                print("action policy", "\t\t\t\t\t", "state")
                for _ac_po, _state in zip(
                    np.round(_action_policy, 3), np.round(_full_state, 3)
                ):
                    print(_ac_po, _state)

            plot_traj(_images, _gripper_coordinates_2d)
            init_seg = seg
            _images = []
            if action_policy is not None:
                _action_policy = []
                _full_state = []
            _gripper_coordinates_2d = []
            init_state = full_state[i]
        _images.append(images[i])
        if action_policy is not None:
            _action_policy.append(action_policy[i])
            _full_state.append(full_state[i])
        if gripper_coordinates_2d:
            _gripper_coordinates_2d.append(gripper_coordinates_2d[i])

    print(f"Segment: {init_seg}")
    print(describe_move(full_state[i] - init_state))
    if action_policy is not None:
        for _ac_po, _state in zip(
            np.round(_action_policy, 3), np.round(_full_state, 3)
        ):
            print(_ac_po, _state)

    plot_traj(_images, _gripper_coordinates_2d)


def plot_traj_on_single_image(image, gripper_coordinates_2d):
    assert gripper_coordinates_2d != None

    fig, ax = plt.subplots()
    width, height, _ = image.shape
    ax.imshow(image, extent=[0, width, height, 0])
    norm = plt.Normalize(0, len(gripper_coordinates_2d))
    cmap = plt.get_cmap("plasma")
    for i in range(len(gripper_coordinates_2d)):
        ax.plot(
            gripper_coordinates_2d[i][0],
            gripper_coordinates_2d[i][1],
            "-",
            color=cmap(norm(i)),
            marker="o",
        )
    plt.show()


def load_traj(
    path,
    camera_angle="images0",
    gripper_coordinates_2d=None,
    segments=None,
    plot=False,
    load_images=False,
):
    if "images" not in path:
        sample = Path(path) / camera_angle
    else:
        sample = Path(path)
    try:
        instruction = open(sample.parent / "lang.txt").readlines()[0]
    except Exception:
        instruction = "Empty"
    sorted_files = sorted(sample.iterdir(), key=extract_image_number)
    full_state = read_pickle(sample.parent / "obs_dict.pkl")["full_state"]

    if load_images == False:
        return instruction, None, full_state

    images = [
        np.array(Image.open(image_file).resize((256, 256), Image.Resampling.LANCZOS))
        for image_file in sorted_files
    ]

    if type(segments) == np.ndarray:
        segments = segments.tolist()

    if plot == True:
        print("Instruction: ", instruction)
        plot_traj_segments(images, full_state, segments, gripper_coordinates_2d)
    if type(plot) == int:
        print("Instruction: ", instruction)
        assert segments == None
        image = images[plot]
        plot_traj_on_single_image(image, gripper_coordinates_2d)

    # sample_policy = read_pickle(sample.parent / 'policy_out.pkl')

    return instruction, images, full_state


def describe_move(move_vec):
    """
    {False: "move backward", True: "move forward"},
    {False: "move right", True: "move left"},
    {False: "move downward", True: "move upward"},
    {False: "roll downward", True: "roll upward"},
    {False: "pitch downward", True: "pitch upward"},
    {False: "yaw clockwise", True: "yaw counterclockwise"},
    {False: "close gripper", True: "open gripper"},
    """
    assert len(move_vec) == 7
    names = [
        {False: "move backward", True: "move forward"},
        {False: "move right", True: "move left"},
        {False: "move downward", True: "move upward"},
        {False: "roll downward", True: "roll upward"},
        {False: "pitch downward", True: "pitch upward"},
        {False: "yaw clockwise", True: "yaw counterclockwise"},
        {False: "close gripper", True: "open gripper"},
    ]

    description = ""
    for i, mv in enumerate(move_vec):
        if i < 3:  # xyz
            description += names[i][mv > 0] + f" {abs(round(mv*1000))} steps; "
        elif 3 <= i < 6:  # oxyz
            description += names[i][mv > 0] + f" {abs(round(mv*180/math.pi))} steps; "
        elif i == 6:  # gripper
            description += names[i][mv > 0.5] + ";"

    return description


read_pickle = lambda x: pickle.load(open(x, "rb"))
