"""
generate.py

Simple CLI script to interactively test generating from a pretrained VLM; provides a minimal REPL for specify image
URLs, prompts, and language generation parameters.

Run with: python scripts/generate.py --model_path <PATH TO LOCAL MODEL OR HF HUB>
"""

from cProfile import label
from copy import deepcopy
import os
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import tqdm

from dataclasses import dataclass
from pathlib import Path
from string import printable
from typing import Union

import draccus
import requests
import torch
from PIL import Image

from prismatic import load,load_vla
from prismatic.overwatch import initialize_overwatch
from prismatic.vla.datasets.val_dataset import ValDataset
import copy
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from prismatic.vla.action_tokenizer import ActionTokenizer
import re






# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# Default Image URL (Beignets)
DEFAULT_IMAGE_URL = (
    # "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
    "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj0/images0/im_12.jpg"
)


@dataclass
class GenerateConfig:
    # fmt: off
    model_path: Union[str, Path] = (                                    # Path to Pretrained VLM (on disk or HF Hub)
      
        "/home/sunqi/openvla/Logs/prism-dinosiglip-224px+mx-aug-multi-policy-wposition-4gpu+n0+b8+x7/checkpoints/step-012234-epoch-01-loss=0.1403.pt"  
    )

     # HF Hub Credentials (required for Gated Models like LLaMa-2)
    hf_token: Union[str, Path] = Path(".hf_token")                      # Environment variable or Path to HF Token

    # Default Generation Parameters =>> subscribes to HuggingFace's GenerateMixIn API
    do_sample: bool = False
    temperature: float = 1.0
    max_new_tokens: int = 128
    min_length: int = 1

    # fmt: on

def sample_val(val_ds):
    sampled=[]
    # random.shuffle(sampled)
    for i in tqdm(range(0,len(val_ds))):
        item=val_ds[i]
        if "pick" in item["input_ids"] or "place" in item["input_ids"] or  "put" in item["input_ids"]:
            sampled.append(item)
        if len(sampled)==1000:
            break
    return sampled

@draccus.wrap()
def generate(cfg: GenerateConfig) -> None:
    tag="test"
    overwatch.info(f"Initializing Generation Playground with Prismatic Model `{cfg.model_path}`")

    # initial the test dataset
    val_ds = ValDataset() 
    print(len(val_ds))
    dataset=sample_val(val_ds)
    # dataset=val_ds
    print(len(dataset))
    # print(dataset[0]["input_ids"])
    # print(dataset[0].keys())


    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load the pretrained VLM --> uses default `load()` function
    vlm = load_vla(model_id_or_path=cfg.model_path, hf_token=hf_token)
    vlm.to(device, dtype=torch.bfloat16)

    #  previous Initial Setup
    # image = Image.open(requests.get(DEFAULT_IMAGE_URL, stream=True).raw).convert("RGB")
    # prompt_builder = vlm.get_prompt_builder()
    # system_prompt = prompt_builder.system_prompt

    # Build Prompt
    # prompt_builder.add_turn(role="human", message="What action should the robot take to achieve the instruction\nInstruction: \nclose drawer of boxAccomplished Actions: \nCurrent position of the gripper in the image: [39, 66]\nCurrent Observation Image:")
    # prompt_text = prompt_builder.get_prompt()

    # Generate from the VLM
    import time; start = time.time()
    resutls=[]
    for j in tqdm(range(len(dataset))):
        item=dataset[j]
        result_item={}
        generated_text = vlm.generate(
        image=item["image"],
        prompt_text=item["input_ids"].replace("</s>",""),
        do_sample=cfg.do_sample,
        temperature=cfg.temperature,
        max_new_tokens=cfg.max_new_tokens,
        min_length=cfg.min_length,
        )
        
        result_item["prompt_text"]=item["input_ids"].replace("</s>","")
        result_item["ground_truth"]=item["labels"]
        result_item["VLM_Response"]=generated_text
        resutls.append(result_item)
        if j%50==0:
            # print(f"\t|=>> VLM Response >>> {generated_text}\n")
            print("VLM response of frame: ", j)
            print(generated_text)
            
       
        # print(resutls)
        # break
    elapsed_time = time.time() - start
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    formatted_time = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
    print("Time used,", formatted_time)

    print("len of resutls: ",len(resutls))
    resutls=json.dumps(resutls)
    if tag=="train":
        path="./VLM_aug_multiple_results_train_pick.json"
    else: 
        path="./VLM_aug_multiple_results_val_pick.json"
    with open(path,'w') as f1:
        f1.write(resutls)
    
    
    metric_movement(tag=tag)
    metric(tag=tag)

    # prompt_builder.add_turn(role="gpt", message=generated_text)

def metric(tag):
    if tag=="train":
        results=json.load(open("./VLM_aug_multiple_results_train_pick.json"))
    else:    
        results=json.load(open("./VLM_aug_multiple_results_val_pick.json"))
    print("len of predict: ",len(results))
    # print("<<VLM_Response>>")
    # print(results[-1]["VLM_Response"])
    # print("<<ground_truth>>")
    # print(results[-1]["ground_truth"])
    correct_num=0
    predict_num=0
    total_num=0
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", model_max_length=2048, padding_side="right")
    # breakpoint()
    for i in tqdm(range(len(results))):
        predict=results[i]["VLM_Response"]
        labels=results[i]["ground_truth"]
        predict=predict.split("Next action policies:")[-1]
        predict=predict.replace("\n","")
        predict=predict.split(";")
        
        labels=labels.split("Next action policies:")[-1]
        labels=labels.replace("\n","")
        labels=labels.split(";")
        print("^^^^^^^^^^^^^^^^^^^^^^^")
        print("<<VLM_Response>>")
        print(predict)
        predict_policy_token = tokenizer(predict, add_special_tokens=False).input_ids
        # print(predict_policy_token)

        labels_policy_token = tokenizer(labels, add_special_tokens=False).input_ids
        # print(labels_policy_token)
        # break
        
        print("<<ground_truth>>")
        print(labels)
        print("_______________________")
        
        min_len=min(len(predict),len(labels))
        # print(min_len)
        # print(len(labels[-1]))
        
        for j in range(0,min_len):

            # if len(predict[j])!=7:
            #     print("predict length error!: ",predict[j]) 
            # print("length of predict is : ",len(predict_policy_token[j]))
            # print("length of labels is : ", len(labels_policy_token[j]))
            min_len_single=min(len(predict_policy_token[j]),len(labels_policy_token[j]))
            predict_num+=len(predict_policy_token[j])-1
            total_num+=len(predict_policy_token[j][1:])
            for k in range(1,min_len_single):    # the first token is the begin token
                if predict_policy_token[j][k]==labels_policy_token[j][k]:
                    correct_num+=1
        
        # break
    print("<<metric>>")
    print("correct_num: ",correct_num)
    print("predict_num: ",predict_num)
    print("total_num: ",total_num)
    p=correct_num/predict_num
    r=correct_num/total_num
    f1=2*((p*r)/(p+r))
    print(f"precision: {p*100:.2f}")
    print(f"recall: {r*100:.2f}")
    print(f"F1: {f1*100:.2f}")

def find_substring_between(text, start, end):
    # 正则表达式查找 start 和 end 之间的内容
    pattern = f'{re.escape(start)}(.*?){re.escape(end)}'
    match = re.search(pattern, string=text)
    if match:
        return match.group(1)  # 返回匹配的内容
    else:
        return None
    
def extract_movement(text):
    try:
        text=text.split("\n\n")[-2]      
        text=text.replace("Movement for next action:","")
        text=text.replace("\n","") 
        text=[o.strip() for o in text.split(";")]
        moveList=text[:-1]
    except Exception as e:
        print("Can not find movement!")
        return []
    return moveList

def metric_movement(tag):
    if tag=="train":
        results=json.load(open("./VLM_aug_multiple_results_train_pick.json"))
    else:    
        results=json.load(open("./VLM_aug_multiple_results_val_pick.json"))
    print("len of predict: ",len(results))
    # print("<<VLM_Response>>")
    # print(results[-1]["VLM_Response"])
    # print("<<ground_truth>>")
    # print(results[-1]["ground_truth"])
    correct_num=0
    predict_num=0
    total_num=0
   
    for i in tqdm(range(len(results))):
        predict=results[i]["VLM_Response"]
        labels=results[i]["ground_truth"]
        predict=extract_movement(predict)
        labels=extract_movement(labels)
      
        print("-------------------")
        print("<<VLM_Response>>")
        print(predict)
 
        print("<<ground_truth>>")
        print(labels)
        print("-------------------")
        
        predict_num+=len(predict)
        total_num+=len(labels)
        min_len=min(len(predict),len(labels))
       
        for j in range(min_len):
            
            if predict[j]==labels[j]:
                correct_num+=1
               
    print("<<metric>>")
    print("correct_num: ",correct_num)
    print("predict_num: ",predict_num)
    print("total_num: ",total_num)
    p=correct_num/predict_num
    r=correct_num/total_num
    f1=2*((p*r)/(p+r))
    print(f"precision: {p*100:.2f}")
    print(f"recall: {r*100:.2f}")
    print(f"F1: {f1*100:.2f}")
    
           
if __name__ == "__main__":
    generate()
    # metric(tag="test")
    # metric_movement(tag="test")