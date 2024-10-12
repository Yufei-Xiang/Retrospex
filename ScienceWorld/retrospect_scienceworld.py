from pathlib import Path

import os
import json
import argparse
import numpy as np
import torch
import time
from tqdm import trange
import random
import transformers
import pandas as pd
import sentencepiece as spm

from IQL.src.iql import ImplicitQLearning,  ImplicitQLearning_webshop
from IQL.src.util import return_range, set_seed, Log, sample_batch, torchify, evaluate_policy, DEFAULT_DEVICE, sample_batch_all

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

transformers.logging.set_verbosity_error()

def clean(strIn):
    charsToFilter = ['\t', '\n', '*', '-']
    for c in charsToFilter:
        strIn = strIn.replace(c, ' ')
    return strIn.strip()

def get_dataset_swift():
    dataset = {"observations":[], 'taskDes':[], "freelook":[], "inv":[], "actions":[], "next_observations":[], "next_look":[], "next_inv":[], "rewards":[], "terminals":[]}
    filedir = args.dataset_path
    file_lst = os.listdir(filedir)
    json_file = [file for file in file_lst if os.path.splitext(file)[1]=='.json']
    print(json_file)
    for file_name in json_file:
        with open(os.path.join(filedir, file_name), 'r') as f:
            data = json.load(f)
            total_traj = len(data.keys())
            for i in range(total_traj):
                key = list(data.keys())[i]
                temp = data[key]
                task_des = temp["history"]["taskDescription"]
                # print(temp)
                history = temp['history']['history']
                # print(history)
                for j in range(len(history)):
                    dataset["taskDes"].append(clean(task_des))
                    current_step = history[j]
                    reward = 0
                    if j == 0:
                        dataset["observations"].append("[OBSERVATION]: Start now.")
                        reward = float(current_step["score"])
                    else:
                        pre_step = history[j-1]
                        reward = float(current_step["score"])-float(pre_step["score"])
                        dataset["observations"].append(clean('[OBSERVATION]: '+pre_step["observation"]))
                    
                    terminal = 0
                    if current_step["isCompleted"]=="true":
                        terminal = 1
                    dataset["actions"].append(current_step["action"])
                    dataset["freelook"].append(clean(current_step["freelook"]))
                    dataset["inv"].append(clean(current_step["inventory"]))
                    dataset["next_observations"].append(clean('[OBSERVATION]: '+current_step["observation"]))
                    dataset["rewards"].append(reward)
                    dataset["terminals"].append(terminal)
                    if j != len(history)-1:
                        next_step = history[j+1]
                        dataset["next_look"].append(clean(next_step["freelook"]))
                        dataset["next_inv"].append(clean(next_step["inventory"]))
                    else:
                        dataset["next_look"].append("The trajectory is finished. No need to get freelook.")
                        dataset["next_inv"].append("The trajectory is finished. No need to get inventory.")
                    

    data_pd = pd.DataFrame(dataset)
    data_pd = data_pd.sample(frac=1)
    dataset['taskDes'] = list(data_pd['taskDes'].values)
    dataset["observations"] = list(data_pd['observations'].values)
    dataset["actions"] = list(data_pd['actions'].values)
    dataset["freelook"] = list(data_pd['freelook'].values)
    dataset["inv"] = list(data_pd['inv'].values)
    dataset["next_observations"] = list(data_pd["next_observations"].values)
    dataset["next_look"] = list(data_pd["next_look"].values)
    dataset["next_inv"] = list(data_pd["next_inv"].values)
    dataset["rewards"] = list(data_pd['rewards'].values)
    dataset["terminals"] = list(data_pd['terminals'].values)

    return dataset

def get_dataset_swift_balanced(args):
    dataset = {"observations":[], 'taskDes':[], "freelook":[], "inv":[], "actions":[], "next_observations":[], "next_look":[], "next_inv":[], "rewards":[], "terminals":[]}
    filedir = args.dataset_path
    file_lst = os.listdir(filedir)
    json_file = [file for file in file_lst if os.path.splitext(file)[1]=='.json']
    scale_dict = {'0':20, '1':20, '3':5, '9':20, '12':5, '13':5, '14':3, '16':3, '21':5, '22':20, '23':2, '25':5, '26':20}
    print(json_file)
    for file_name in json_file:
        with open(os.path.join(filedir, file_name), 'r') as f:
            data = json.load(f)
            total_traj = len(data.keys())
            rounds = 1
            if file_name.split('-')[0][4:] in scale_dict:
                rounds = scale_dict[file_name.split('-')[0][4:]]
                print(file_name.split('-')[0][4:])
                print(rounds)
            for round in range(rounds):
                for i in range(total_traj):
                    key = list(data.keys())[i]
                    temp = data[key]
                    task_des = temp["history"]["taskDescription"]
                # print(temp)
                    history = temp['history']['history']
                # print(history)
                    for j in range(len(history)):
                        dataset["taskDes"].append(clean(task_des))
                        current_step = history[j]
                        reward = 0
                        if j == 0:
                            dataset["observations"].append("[OBSERVATION]: Start now.")
                            reward = float(current_step["score"])
                        else:
                            pre_step = history[j-1]
                            reward = float(current_step["score"])-float(pre_step["score"])
                            dataset["observations"].append(clean('[OBSERVATION]: '+pre_step["observation"]))
                    
                        terminal = 0
                        if current_step["isCompleted"]=="true":
                            terminal = 1
                        dataset["actions"].append(current_step["action"])
                        dataset["freelook"].append(clean(current_step["freelook"]))
                        dataset["inv"].append(clean(current_step["inventory"]))
                        dataset["next_observations"].append(clean('[OBSERVATION]: '+current_step["observation"]))
                        dataset["rewards"].append(reward)
                        dataset["terminals"].append(terminal)
                        if j != len(history)-1:
                            next_step = history[j+1]
                            dataset["next_look"].append(clean(next_step["freelook"]))
                            dataset["next_inv"].append(clean(next_step["inventory"]))
                        else:
                            dataset["next_look"].append("The trajectory is finished. No need to get freelook.")
                            dataset["next_inv"].append("The trajectory is finished. No need to get inventory.")

    data_pd = pd.DataFrame(dataset)
    data_pd = data_pd.sample(frac=1)
    dataset['taskDes'] = list(data_pd['taskDes'].values)
    dataset["observations"] = list(data_pd['observations'].values)
    dataset["actions"] = list(data_pd['actions'].values)
    dataset["freelook"] = list(data_pd['freelook'].values)
    dataset["inv"] = list(data_pd['inv'].values)
    dataset["next_observations"] = list(data_pd["next_observations"].values)
    dataset["next_look"] = list(data_pd["next_look"].values)
    dataset["next_inv"] = list(data_pd["next_inv"].values)
    dataset["rewards"] = list(data_pd['rewards'].values)
    dataset["terminals"] = list(data_pd['terminals'].values)

    return dataset

def reform_datapd(data_pd):
    lst = list(data_pd['terminal'].values)
    end_lst = [-1]
    for i in range(9361):
        if lst[i] != 0:
            end_lst.append(i)
    for end in range(1,2001):
        reward = data_pd.at[end_lst[end],'reward']
        length = end_lst[end]-end_lst[end-1]
        avg_reward = reward/length
        for j in range(end_lst[end-1]+1,end_lst[end]+1):
            data_pd.at[j,'reward'] = avg_reward
    return

def check_dataset(dataset):
    for i in dataset.keys():
        value = dataset[i]
        for j in range(len(value)):
            if len(value[j]) <= 0:
                print(i,j)

def train_iql_swift(args):
    torch.set_num_threads(1)
    set_seed(args.seed, env=None)
    dataset = get_dataset_swift_balanced(args)
    iql = ImplicitQLearning(
        args,
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=args.learning_rate)
    )
    start = 0
    k = list(dataset.keys())[0]
    n = len(dataset[k])
    round = 0
    for step in trange(args.n_steps):
        if round >= 20:
            break
        end = start + args.batch_size
        if end >= n:
            iql.update(**sample_batch_all(dataset, start, n))
            start = 0
            round += 1
        else:
            iql.update(**sample_batch_all(dataset, start, end))
            start = end
        print("Start = %d"%start)
    
    model_path = args.save_path
    torch.save(iql.state_dict(), model_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='logs')
    parser.add_argument('--spm_path', default='IQL/spm_models/unigram_8k.model')
    parser.add_argument('--rom_path', default='zork1.z5')
    parser.add_argument("--jar_path", type=str,
                        help="Path to the ScienceWorld jar file. Default: use builtin.")

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--embedding_dim', default=64, type=int)
    parser.add_argument('--n-hidden', type=int, default=2)
    parser.add_argument('--n-steps', type=int, default=200000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--deterministic-policy', action='store_true')
    parser.add_argument('--eval-period', type=int, default=10000)
    parser.add_argument('--n-eval-episodes', type=int, default=10)
    parser.add_argument('--max-episode-steps', type=int, default=1000)
    parser.add_argument('--dataset_path', default='memory_trajectories/train_trajs_new', type=str)
    parser.add_argument('--save_path', default='final_iql_swift.pt', type=str)


    return parser.parse_args()

if __name__ == '__main__':
    # main(parse_args())
    args = parse_args()
    train_iql_swift(args)