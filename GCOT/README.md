# Setup and download datasets
1. download the rlds of bridgeV2 dataset following [openvla](https://github.com/openvla/openvla)
```bash
# Change directory to your base datasets folder
cd <PATH TO BASE DATASETS DIR>

# Download the full dataset (124 GB)
wget -r -nH --cut-dirs=4 --reject="index.html*" https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/

# Rename the dataset to `bridge_orig` (NOTE: Omitting this step may lead to runtime errors later)
mv bridge_dataset bridge_orig
```

2. download the ecot dataset (we need this to get the gripper positions):

download [link](https://huggingface.co/datasets/Embodied-CoT/embodied_features_bridge)

## Run dataset creation

1. get gripper position in 2d image trajectories by `python gripper_position.py`
2. get high-level plan by `python generate_plans.py`
3. run dataset by `python create_dataset.py`


