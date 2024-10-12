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

from IQL.src.iql import ImplicitQLearning_webshop
from IQL.src.util import return_range, set_seed, Log, sample_batch, torchify, evaluate_policy, DEFAULT_DEVICE, sample_batch_all

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

transformers.logging.set_verbosity_error()

def clean(strIn):
    charsToFilter = ['\t', '\n', '*', '-']
    for c in charsToFilter:
        strIn = strIn.replace(c, ' ')
    return strIn.strip()

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

def get_dataset_webshop():
    dataset = {"observations":[], 'taskDes':[], "actions":[], "next_observations":[], "rewards":[], "terminals":[]}
    data_pd = pd.read_csv("memory_trajectories/trajsOnWebshop.csv",header=None)
    data_pd.columns = ['taskDes','observation','next_observation','action','reward','terminal']
    reform_datapd(data_pd)
    data_pd = data_pd.sample(frac=1)
    dataset['taskDes'] = list(data_pd['taskDes'].values)
    dataset["observations"] = list(data_pd['observation'].values)
    dataset["actions"] = list(data_pd['action'].values)
    dataset["next_observations"] = list(data_pd["next_observation"].values)
    dataset["rewards"] = list(data_pd['reward'].values)
    dataset["terminals"] = list(data_pd['terminal'].values)
    return dataset


def train_iql_web(args):
    torch.set_num_threads(1)
    set_seed(args.seed, env=None)
    dataset = get_dataset_webshop()
    iql = ImplicitQLearning_webshop(
        args,
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=args.learning_rate)
    )
    start = 0
    k = list(dataset.keys())[0]
    n = len(dataset[k])
    print(n)
    round = 0
    for step in trange(args.n_steps):
        if round >= 20:
            break
        end = start + args.batch_size
        if end >= n:
            iql.update(**sample_batch_all(dataset, start, n))
            start = 0
            round += 1
            print('End round%d\n'%round)
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


    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--embedding_dim', default=64, type=int)
    parser.add_argument('--n-hidden', type=int, default=2)
    parser.add_argument('--n-steps', type=int, default=20000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--deterministic-policy', action='store_true')
    parser.add_argument('--eval-period', type=int, default=10000)
    parser.add_argument('--n-eval-episodes', type=int, default=10)
    parser.add_argument('--max-episode-steps', type=int, default=1000)
    parser.add_argument('--save_path', default='final_iql_webshop.pt', type=str)


    return parser.parse_args()

if __name__ == '__main__':
    # main(parse_args())
    args = parse_args()
    train_iql_web(args)