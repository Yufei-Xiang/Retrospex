import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import mlp, DEFAULT_DEVICE
from .util_drrn import pad_sequences
from collections import namedtuple
import itertools
from collections import namedtuple


class DRRN_Q(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(DRRN_Q, self).__init__()
        self.embedding    = nn.Embedding(vocab_size, embedding_dim)
        self.taskdes_encoder = nn.GRU(embedding_dim, hidden_dim)
        self.obs_encoder  = nn.GRU(embedding_dim, hidden_dim)
        self.look_encoder = nn.GRU(embedding_dim, hidden_dim)
        self.inv_encoder = nn.GRU(embedding_dim, hidden_dim)
        self.act_encoder  = nn.GRU(embedding_dim, hidden_dim)
        self.hidden       = nn.Linear(5*hidden_dim, hidden_dim)
        self.act_scorer   = nn.Linear(hidden_dim, 1)


    def packed_rnn(self, x, rnn):
        lengths = torch.tensor([len(n) for n in x], dtype=torch.long, device=DEFAULT_DEVICE)
        # Sort this batch in descending order by seq length
        lengths, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        idx_sort = torch.autograd.Variable(idx_sort)
        idx_unsort = torch.autograd.Variable(idx_unsort)
        padded_x = pad_sequences(x)
        x_tt = torch.from_numpy(padded_x).type(torch.long).to(DEFAULT_DEVICE)
        x_tt = x_tt.index_select(0, idx_sort)
        # Run the embedding layer
        embed = self.embedding(x_tt).permute(1,0,2) # Time x Batch x EncDim
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embed, lengths.cpu())
        # Run the RNN
        out, _ = rnn(packed)
        # Unpack
        out, _ = nn.utils.rnn.pad_packed_sequence(out)
        # Get the last step of each sequence
        idx = (lengths-1).view(-1,1).expand(len(lengths), out.size(2)).unsqueeze(0)
        out = out.gather(0, idx).squeeze(0)
        # Unsort
        out = out.index_select(0, idx_unsort)
        return out


    def forward(self, taskdes_batch, look_batch, inv_batch, obs_batch, act_batch):
        taskdes_out = self.packed_rnn(taskdes_batch, self.taskdes_encoder)
        look_out = self.packed_rnn(look_batch, self.look_encoder)
        inv_out = self.packed_rnn(inv_batch, self.inv_encoder)
        act_out = self.packed_rnn(act_batch, self.act_encoder)
        # Encode the various aspects of the state
        obs_out = self.packed_rnn(obs_batch, self.obs_encoder)
        # Expand the state to match the batches of actions
        z = torch.cat([taskdes_out,look_out, inv_out, obs_out, act_out], dim=1) # Concat along hidden_dim
        z = F.relu(self.hidden(z))
        act_values = self.act_scorer(z).squeeze(-1)
        return act_values



class DRRN_V(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(DRRN_V, self).__init__()
        self.embedding    = nn.Embedding(vocab_size, embedding_dim)
        self.taskdes_encoder = nn.GRU(embedding_dim, hidden_dim)
        self.obs_encoder  = nn.GRU(embedding_dim, hidden_dim)
        self.look_encoder = nn.GRU(embedding_dim, hidden_dim)
        self.inv_encoder = nn.GRU(embedding_dim, hidden_dim)
        self.hidden       = nn.Linear(4*hidden_dim, hidden_dim)
        self.scorer   = nn.Linear(hidden_dim, 1)


    def packed_rnn(self, x, rnn):
        lengths = torch.tensor([len(n) for n in x], dtype=torch.long, device=DEFAULT_DEVICE)
        # Sort this batch in descending order by seq length
        lengths, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        idx_sort = torch.autograd.Variable(idx_sort)
        idx_unsort = torch.autograd.Variable(idx_unsort)
        padded_x = pad_sequences(x)
        x_tt = torch.from_numpy(padded_x).type(torch.long).to(DEFAULT_DEVICE)
        x_tt = x_tt.index_select(0, idx_sort)
        # Run the embedding layer
        embed = self.embedding(x_tt).permute(1,0,2) # Time x Batch x EncDim
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embed, lengths.cpu())
        # Run the RNN
        out, _ = rnn(packed)
        # Unpack
        out, _ = nn.utils.rnn.pad_packed_sequence(out)
        # Get the last step of each sequence
        idx = (lengths-1).view(-1,1).expand(len(lengths), out.size(2)).unsqueeze(0)
        out = out.gather(0, idx).squeeze(0)
        # Unsort
        out = out.index_select(0, idx_unsort)
        return out


    def forward(self, taskdes_batch, look_batch, inv_batch, obs_batch):
        # Encode the various aspects of the state
        obs_out = self.packed_rnn(obs_batch, self.obs_encoder)
        taskdes_out = self.packed_rnn(taskdes_batch, self.taskdes_encoder)
        look_out = self.packed_rnn(look_batch, self.look_encoder)
        inv_out = self.packed_rnn(inv_batch, self.inv_encoder)
        z = torch.cat([taskdes_out,look_out, inv_out, obs_out], dim=1)
        z = F.relu(self.hidden(z))
        act_values = self.scorer(z).squeeze(-1)
        return act_values
