import json

import numpy as np
import tensorflow as tf
from PIL import Image
from transformers import AutoTokenizer

from prismatic.models.backbones.llm.prompting.base_prompter import PurePromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.rlds_dataset import bridge_v2_dataset

class ValDataset:
    def __init__(
        self,
        prompt_builder_fn = PurePromptBuilder,
        file_name: str = "text-how-to-go/aug_multi_policy/first_version_val.json",
        # file_name: str = "text-how-to-go/single_policy/first_version_val.json", # 修改 测试有memery 版本
        data_mix: str = "pred_all",
        mask_inst: bool = False,
    ) -> None:
        self.prompt_builder_fn = prompt_builder_fn
        self.mask_inst = mask_inst
        self.prompt_display_count = 2
        self.data_mix = data_mix
        self.split = file_name.split("_")[-1].replace(".json","")              
        # self.split = "train"
        dataset, dataset_len, statistics = bridge_v2_dataset(split=self.split)
        self.build_dataset(dataset)
        self.llm_max_length = 1024
        tokenizer = AutoTokenizer.from_pretrained(
            'meta-llama/Llama-2-7b-hf', model_max_length=self.llm_max_length, padding_side="right"
        )
        self.action_tokenizer = ActionTokenizer(tokenizer)


        with open(f"{file_name}", "r") as file:
            self.data = json.load(file)
        self.dataset_statistics = {
            "fast_dataset": {
                # TODO: Update this
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def build_dataset(self, dataset):
        print("Finished loading dataset in TF")
        self.info = dict()
        for i, traj in enumerate(dataset.as_numpy_iterator()):
            key = traj["file_path"][0].decode("utf-8") + "|" + str(traj["episode_id"][0])
            self.info[key] = traj
        print("Finish building dataset")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        key = "/".join(item["current_image_path"].split("/")[:-1])
        current_image_index = int(item["current_image_path"].split("/")[-1].split("_")[-1].split(".")[0])
        info = self.info[key]
        images = [
            Image.fromarray(
                tf.io.decode_image(
                    info["observation"]["image_primary"][i][0], expand_animations=False, dtype=tf.uint8
                ).numpy()
            )
            for i in range(len(info["observation"]["image_primary"]))
        ]
        image = images[current_image_index]

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

        conversation = [
            {"from": "human", "value": prompt_str},
            {"from": "gpt", "value": ""},
        ]

        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        return dict(image=image, images=images, current_image_index=current_image_index, input_ids=prompt_builder.get_prompt(), labels=gpt_output)


    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


if __name__ == "__main__":
    val_ds = ValDataset()
    for o in val_ds:
        print(o.keys())
        print(o["image"])
        print(o["input_ids"])
        print(o["labels"])
        break
