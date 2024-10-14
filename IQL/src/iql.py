import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

from .util import DEFAULT_DEVICE, compute_batched, update_exponential_moving_average
from .value_functions import DRRN_Q, DRRN_V, DRRN_V_web, DRRN_TwinQ

EXP_ADV_MAX = 100.


def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class ImplicitQLearning(nn.Module):
    def __init__(self, args, optimizer_factory):
        super().__init__()
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(args.spm_path)
        self.qf = DRRN_Q(len(self.sp), args.embedding_dim, args.hidden_dim).to(DEFAULT_DEVICE)
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(DEFAULT_DEVICE)
        self.vf = DRRN_V(len(self.sp), args.embedding_dim, args.hidden_dim).to(DEFAULT_DEVICE)
        self.v_optimizer = optimizer_factory(self.vf.parameters())
        self.q_optimizer = optimizer_factory(self.qf.parameters())
        self.tau = args.tau
        self.beta = args.beta
        self.discount = args.discount
        self.alpha = args.alpha

    def update(self,observations, taskDes, freelook, inv,  actions, next_observations, next_look, next_inv, rewards, terminals):
        obs_ids = [self.sp.EncodeAsIds(o) for o in observations]
        task_ids = [self.sp.EncodeAsIds(t) for t in taskDes]
        free_ids = [self.sp.EncodeAsIds(f) for f in freelook]
        inv_ids = [self.sp.EncodeAsIds(i) for i in inv]
        # TextWorld
        action_ids = [self.sp.EncodeAsIds(action) for action in actions]
        nextobs_ids = [self.sp.EncodeAsIds(next_ob) for next_ob in next_observations]
        nextfree_ids = [self.sp.EncodeAsIds(f) for f in next_look]
        nextinv_ids = [self.sp.EncodeAsIds(i) for i in next_inv]
        rewards = torch.tensor(rewards).to(DEFAULT_DEVICE)
        terminals = torch.tensor(terminals).to(DEFAULT_DEVICE)

        with torch.no_grad():
            target_q = self.q_target(task_ids, free_ids, inv_ids, obs_ids, action_ids)
            next_v = self.vf(task_ids, nextfree_ids, nextinv_ids, nextobs_ids)

        # v, next_v = compute_batched(self.vf, [observations, next_observations])

        # Update value function
        v = self.vf(task_ids, free_ids, inv_ids, obs_ids)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.tau)
        with open('loss_file.txt', 'a') as f:
            f.write("v_loss is"+str(v_loss))
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()
        
        # print(rewards.shape)
        # print(terminals.shape)
        # print(next_v.shape)
        # Update Q function
        targets = rewards + (1. - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf(task_ids, free_ids, inv_ids, obs_ids, action_ids)
        # print(qs.shape)
        # print(targets.shape)
        q_loss = F.mse_loss(qs.float(), targets.float())
        # with open('loss_file.txt', 'a') as f:
        #     f.write("q_loss is"+str(q_loss))
        #     f.write('\n')
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        update_exponential_moving_average(self.q_target, self.qf, self.alpha)

    def act(self, taskDes, freelook, inv, observation, actions):
        task_ids = [self.sp.EncodeAsIds(taskDes) for i in range(len(actions))]
        free_ids = [self.sp.EncodeAsIds(freelook) for i in range(len(actions))]
        inv_ids = [self.sp.EncodeAsIds(inv) for i in range(len(actions))]
        obs_ids = [self.sp.EncodeAsIds(observation) for i in range(len(actions))]
        action_ids = [self.sp.EncodeAsIds(action) for action in actions]
        q_value = list(self.q_target(task_ids, free_ids, inv_ids, obs_ids, action_ids))
        lst = [i for i in zip(q_value, actions)]
        lst.sort(reverse=True)
        return lst[0][1]
    
    def get_q(self, taskDes, freelook, inv, observation, actions):
        task_ids = [self.sp.EncodeAsIds(taskDes) for i in range(len(actions))]
        free_ids = [self.sp.EncodeAsIds(freelook) for i in range(len(actions))]
        inv_ids = [self.sp.EncodeAsIds(inv) for i in range(len(actions))]
        obs_ids = [self.sp.EncodeAsIds(observation) for i in range(len(actions))]
        action_ids = [self.sp.EncodeAsIds(action) for action in actions]
        q_value = self.q_target(task_ids, free_ids, inv_ids, obs_ids, action_ids)
        v_value = self.vf(task_ids, free_ids, inv_ids, obs_ids)
        return q_value-v_value
    
    
class ImplicitQLearning_webshop(nn.Module):
    def __init__(self, args, optimizer_factory):
        super().__init__()
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(args.spm_path)
        self.qf = DRRN_TwinQ(len(self.sp), args.embedding_dim, args.hidden_dim).to(DEFAULT_DEVICE)
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(DEFAULT_DEVICE)
        self.vf = DRRN_V_web(len(self.sp), args.embedding_dim, args.hidden_dim).to(DEFAULT_DEVICE)
        self.v_optimizer = optimizer_factory(self.vf.parameters())
        self.q_optimizer = optimizer_factory(self.qf.parameters())
        self.tau = args.tau
        self.beta = args.beta
        self.discount = args.discount
        self.alpha = args.alpha

    def update(self,observations, taskDes,  actions, next_observations, rewards, terminals):
        obs_ids = [self.sp.EncodeAsIds(o) for o in observations]
        task_ids = [self.sp.EncodeAsIds(t) for t in taskDes]
        # TextWorld
        action_ids = [self.sp.EncodeAsIds(action) for action in actions]
        nextobs_ids = [self.sp.EncodeAsIds(next_ob) for next_ob in next_observations]
        rewards = torch.tensor(rewards).to(DEFAULT_DEVICE)
        terminals = torch.tensor(terminals).to(DEFAULT_DEVICE)

        with torch.no_grad():
            target_q = self.q_target(task_ids, obs_ids, action_ids)
            next_v = self.vf(task_ids, nextobs_ids)

        # v, next_v = compute_batched(self.vf, [observations, next_observations])

        # Update value function
        v = self.vf(task_ids, obs_ids)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.tau)
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()
        
        # print(rewards.shape)
        # print(terminals.shape)
        # print(next_v.shape)
        # Update Q function
        targets = rewards + (1. - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf(task_ids, obs_ids, action_ids)
        # print(qs.shape)
        # print(targets.shape)
        q_loss = F.mse_loss(qs, targets)
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        update_exponential_moving_average(self.q_target, self.qf, self.alpha)

    def act(self, taskDes, observation, actions):
        task_ids = [self.sp.EncodeAsIds(taskDes) for i in range(len(actions))]
        obs_ids = [self.sp.EncodeAsIds(observation) for i in range(len(actions))]
        action_ids = [self.sp.EncodeAsIds(action) for action in actions]
        q_value = list(self.q_target(task_ids, obs_ids, action_ids))
        lst = [i for i in zip(q_value, actions)]
        lst.sort(reverse=True)
        return lst[0][1]
    
    def get_q(self, taskDes, observation, actions):
        task_ids = [self.sp.EncodeAsIds(taskDes) for i in range(len(actions))]
        obs_ids = [self.sp.EncodeAsIds(observation) for i in range(len(actions))]
        action_ids = [self.sp.EncodeAsIds(action) for action in actions]
        with torch.no_grad():
            q_value = self.q_target(task_ids, obs_ids, action_ids)
            v_value = self.vf(task_ids, obs_ids)
        return q_value-v_value
