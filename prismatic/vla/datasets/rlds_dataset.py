from prismatic.vla.datasets.rlds.dataset import load_bridgev2_dataset
from prismatic.vla.datasets.rlds.oxe import (
    OXE_NAMED_MIXTURES,
    get_oxe_dataset_kwargs_and_weights,
)
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType


def bridge_v2_dataset(split):
    # Configure RLDS Dataset(s)
    data_mix = "bridge_orig"
    if data_mix in OXE_NAMED_MIXTURES:
        mixture_spec = OXE_NAMED_MIXTURES[data_mix]
    else:
        # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
        mixture_spec = [(data_mix, 1.0)]

    # fmt: off
    per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
        "./",
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
            resize_size=tuple([224, 224]),
            num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
        ),
        dataset_kwargs_list=per_dataset_kwargs,
        shuffle_buffer_size=256_000,
        sample_weights=weights,
        balance_weights=True,
        traj_transform_threads=len(mixture_spec),
        traj_read_threads=len(mixture_spec),
        train=(split=="train"),
    )

    # If applicable, enable image augmentations
    if False:
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
    dataset, dataset_len, all_dataset_statistics = load_bridgev2_dataset(**rlds_config)
    return dataset, dataset_len, all_dataset_statistics


if __name__ == "__main__":
    dataset, dataset_len, all_dataset_statistics = bridge_v2_dataset("train")
    # dataset, dataset_len, all_dataset_statistics = bridge_v2_dataset("val")
