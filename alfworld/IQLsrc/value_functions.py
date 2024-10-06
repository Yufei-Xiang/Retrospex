import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import mlp, DEFAULT_DEVICE
from .util_drrn import pad_sequences
from collections import namedtuple
import itertools
# from util_drrn import pad_sequences
from collections import namedtuple
from sentence_transformers import SentenceTransformer
from transformers import BertModel, T5EncoderModel, T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Bertmodel = SentenceTransformer("bert-base-nli-mean-tokens")


def use_T5(input_text = "translate English to German: How old are you?"):
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", device_map="auto", load_in_8bit=True)
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

    outputs = model.generate(input_ids)
    print(tokenizer.decode(outputs[0]))

# tokenizer = AutoTokenizer.from_pretrained("bakedpotat/T5EncoderModel")
# model = AutoModel.from_pretrained("bakedpotat/T5EncoderModel")
T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
auto_model = T5EncoderModel.from_pretrained("t5-base")


class TwinQ(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = mlp(dims, squeeze_output=True).to(DEFAULT_DEVICE)
        self.q2 = mlp(dims, squeeze_output=True).to(DEFAULT_DEVICE)

    def both(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state, action):
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = mlp(dims, squeeze_output=True).to(DEFAULT_DEVICE)

    def forward(self, state):
        return self.v(state)
    

# class BertQ(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim=256, n_hidden=2):
#         super().__init__()
#         self.l1 = BertModel.from_pretrained('bert-base-uncased')
#         self.l2 = torch.nn.Dropout(0.3)
#         dims = [768, *([hidden_dim] * n_hidden), 1]
#         self.l3 = mlp(dims, squeeze_output=True).to(DEFAULT_DEVICE)

#     def forward(self, ids, mask, token_type_ids):
#         _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
#         output_2 = self.l2(output_1)
#         output = self.l3(output_2)
#         return output


# class BertV(nn.Module):
#     def __init__(self, state_dim, hidden_dim=256, n_hidden=2):
#         super().__init__()
#         self.l1 = BertModel.from_pretrained('bert-base-uncased')
#         self.l2 = torch.nn.Dropout(0.3)
#         dims = [768, *([hidden_dim] * n_hidden), 1]
#         self.l3 = mlp(dims, squeeze_output=True).to(DEFAULT_DEVICE)

#     def forward(self, ids, mask, token_type_ids):
#         _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
#         output_2 = self.l2(output_1)
#         output = self.l3(output_2)
#         return output


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
        lengths = torch.tensor([len(n) for n in x], dtype=torch.long, device=device)
        # Sort this batch in descending order by seq length
        lengths, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        idx_sort = torch.autograd.Variable(idx_sort)
        idx_unsort = torch.autograd.Variable(idx_unsort)
        padded_x = pad_sequences(x)
        x_tt = torch.from_numpy(padded_x).type(torch.long).to(device)
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
        lengths = torch.tensor([len(n) for n in x], dtype=torch.long, device=device)
        # Sort this batch in descending order by seq length
        lengths, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        idx_sort = torch.autograd.Variable(idx_sort)
        idx_unsort = torch.autograd.Variable(idx_unsort)
        padded_x = pad_sequences(x)
        x_tt = torch.from_numpy(padded_x).type(torch.long).to(device)
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

class DRRN_Q_web(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(DRRN_Q_web, self).__init__()
        self.embedding    = nn.Embedding(vocab_size, embedding_dim)
        self.taskdes_encoder = nn.GRU(embedding_dim, hidden_dim)
        self.obs_encoder  = nn.GRU(embedding_dim, hidden_dim)
        self.act_encoder  = nn.GRU(embedding_dim, hidden_dim)
        self.hidden       = nn.Linear(3*hidden_dim, hidden_dim)
        self.act_scorer   = nn.Linear(hidden_dim, 1)


    def packed_rnn(self, x, rnn):
        lengths = torch.tensor([len(n) for n in x], dtype=torch.long, device=device)
        # Sort this batch in descending order by seq length
        lengths, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        idx_sort = torch.autograd.Variable(idx_sort)
        idx_unsort = torch.autograd.Variable(idx_unsort)
        padded_x = pad_sequences(x)
        x_tt = torch.from_numpy(padded_x).type(torch.long).to(device)
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


    def forward(self, taskdes_batch, obs_batch, act_batch):
        taskdes_out = self.packed_rnn(taskdes_batch, self.taskdes_encoder)
        act_out = self.packed_rnn(act_batch, self.act_encoder)
        # Encode the various aspects of the state
        obs_out = self.packed_rnn(obs_batch, self.obs_encoder)
        # Expand the state to match the batches of actions
        z = torch.cat([taskdes_out, obs_out, act_out], dim=1) # Concat along hidden_dim
        z = F.relu(self.hidden(z))
        act_values = self.act_scorer(z).squeeze(-1)
        return act_values

class DRRN_TwinQ(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.q1 = DRRN_Q_web(vocab_size, embedding_dim, hidden_dim)
        self.q2 = DRRN_Q_web(vocab_size, embedding_dim, hidden_dim)

    def both(self, taskdes_batch, obs_batch, act_batch):
        return self.q1(taskdes_batch, obs_batch, act_batch), self.q2(taskdes_batch, obs_batch, act_batch)

    def forward(self, taskdes_batch, obs_batch, act_batch):
        return torch.min(*self.both(taskdes_batch, obs_batch, act_batch))

class DRRN_V_web(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(DRRN_V_web, self).__init__()
        self.embedding    = nn.Embedding(vocab_size, embedding_dim)
        self.taskdes_encoder = nn.GRU(embedding_dim, hidden_dim)
        self.obs_encoder  = nn.GRU(embedding_dim, hidden_dim)
        self.hidden       = nn.Linear(2*hidden_dim, hidden_dim)
        self.scorer   = nn.Linear(hidden_dim, 1)


    def packed_rnn(self, x, rnn):
        lengths = torch.tensor([len(n) for n in x], dtype=torch.long, device=device)
        # Sort this batch in descending order by seq length
        lengths, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        idx_sort = torch.autograd.Variable(idx_sort)
        idx_unsort = torch.autograd.Variable(idx_unsort)
        padded_x = pad_sequences(x)
        x_tt = torch.from_numpy(padded_x).type(torch.long).to(device)
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


    def forward(self, taskdes_batch, obs_batch):
        # Encode the various aspects of the state
        obs_out = self.packed_rnn(obs_batch, self.obs_encoder)
        taskdes_out = self.packed_rnn(taskdes_batch, self.taskdes_encoder)
        z = torch.cat([taskdes_out, obs_out], dim=1)
        z = F.relu(self.hidden(z))
        act_values = self.scorer(z).squeeze(-1)
        return act_values


class GOLDEN_Q(torch.nn.Module):
    """
        Deep Reinforcement Relevance Network - He et al. '16

    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(DRRN_Q, self).__init__()
        self.embedding    = nn.Embedding(vocab_size, embedding_dim)
        self.obs_encoder  = nn.GRU(embedding_dim, hidden_dim)
        self.act_encoder  = nn.GRU(embedding_dim, hidden_dim)
        self.hidden       = nn.Linear(2*hidden_dim, hidden_dim)
        self.act_scorer   = nn.Linear(hidden_dim, 1)


    def packed_rnn(self, x, rnn):
        """ Runs the provided rnn on the input x. Takes care of packing/unpacking.

            x: list of unpadded input sequences
            Returns a tensor of size: len(x) x hidden_dim
        """
        lengths = torch.tensor([len(n) for n in x], dtype=torch.long, device=device)
        # Sort this batch in descending order by seq length
        lengths, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        idx_sort = torch.autograd.Variable(idx_sort)
        idx_unsort = torch.autograd.Variable(idx_unsort)
        padded_x = pad_sequences(x)
        x_tt = torch.from_numpy(padded_x).type(torch.long).to(device)
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


    def forward(self, obs_batch, act_batch):
        """
            Batched forward pass.
            obs_id_batch: iterable of unpadded sequence ids
            act_batch: iterable of lists of unpadded admissible command ids

            Returns a tuple of tensors containing q-values for each item in the batch
        """
        act_out = self.packed_rnn(act_batch, self.act_encoder)
        # Encode the various aspects of the state
        obs_out = self.packed_rnn(obs_batch, self.obs_encoder)
        # Expand the state to match the batches of actions
        z = torch.cat([obs_out, act_out], dim=1) # Concat along hidden_dim
        z = F.relu(self.hidden(z))
        act_values = self.act_scorer(z).squeeze(-1)
        return act_values



class GOLDEN_V(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(DRRN_V, self).__init__()
        self.embedding    = nn.Embedding(vocab_size, embedding_dim)
        self.obs_encoder  = nn.GRU(embedding_dim, hidden_dim)
        self.hidden       = nn.Linear(hidden_dim, hidden_dim)
        self.scorer   = nn.Linear(hidden_dim, 1)


    def packed_rnn(self, x, rnn):
        """ Runs the provided rnn on the input x. Takes care of packing/unpacking.

            x: list of unpadded input sequences
            Returns a tensor of size: len(x) x hidden_dim
        """
        lengths = torch.tensor([len(n) for n in x], dtype=torch.long, device=device)
        # Sort this batch in descending order by seq length
        lengths, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        idx_sort = torch.autograd.Variable(idx_sort)
        idx_unsort = torch.autograd.Variable(idx_unsort)
        padded_x = pad_sequences(x)
        x_tt = torch.from_numpy(padded_x).type(torch.long).to(device)
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


    def forward(self, obs_batch):
        """
            Batched forward pass.
            obs_id_batch: iterable of unpadded sequence ids
            act_batch: iterable of lists of unpadded admissible command ids

            Returns a tuple of tensors containing q-values for each item in the batch
        """
        # Encode the various aspects of the state
        obs_out = self.packed_rnn(obs_batch, self.obs_encoder)
        z = F.relu(self.hidden(obs_out))
        act_values = self.scorer(z).squeeze(-1)
        return act_values


class T5_Q(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(T5_Q, self).__init__()
        self.embedding    = nn.Embedding(vocab_size, embedding_dim)
        self.obs_encoder  = nn.GRU(embedding_dim, hidden_dim)
        self.act_encoder  = nn.GRU(embedding_dim, hidden_dim)
        self.hidden       = nn.Linear(2*hidden_dim, hidden_dim)
        self.act_scorer   = nn.Linear(hidden_dim, 1)


class T5_V(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(T5_V, self).__init__()
        self.embedding    = nn.Embedding(vocab_size, embedding_dim)
        self.obs_encoder  = nn.GRU(embedding_dim, hidden_dim)
        self.hidden       = nn.Linear(hidden_dim, hidden_dim)
        self.scorer   = nn.Linear(hidden_dim, 1)

class DRRNS_Q(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(DRRNS_Q, self).__init__()
        self.embedding    = nn.Embedding(vocab_size, embedding_dim)
        self.obs_encoder  = nn.GRU(embedding_dim, hidden_dim)
        self.look_encoder = nn.GRU(embedding_dim, hidden_dim)
        self.inv_encoder = nn.GRU(embedding_dim, hidden_dim)
        self.act_encoder  = nn.GRU(embedding_dim, hidden_dim)
        self.hidden       = nn.Linear(4*hidden_dim, hidden_dim)
        self.act_scorer   = nn.Linear(hidden_dim, 1)


    def packed_rnn(self, x, rnn):
        lengths = torch.tensor([len(n) for n in x], dtype=torch.long, device=device)
        # Sort this batch in descending order by seq length
        lengths, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        idx_sort = torch.autograd.Variable(idx_sort)
        idx_unsort = torch.autograd.Variable(idx_unsort)
        padded_x = pad_sequences(x)
        x_tt = torch.from_numpy(padded_x).type(torch.long).to(device)
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


    def forward(self, look_batch, inv_batch, obs_batch, act_batch):
        look_out = self.packed_rnn(look_batch, self.look_encoder)
        inv_out = self.packed_rnn(inv_batch, self.inv_encoder)
        act_out = self.packed_rnn(act_batch, self.act_encoder)
        # Encode the various aspects of the state
        obs_out = self.packed_rnn(obs_batch, self.obs_encoder)
        # Expand the state to match the batches of actions
        z = torch.cat([look_out, inv_out, obs_out, act_out], dim=1) # Concat along hidden_dim
        z = F.relu(self.hidden(z))
        act_values = self.act_scorer(z).squeeze(-1)
        return act_values



class DRRNS_V(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(DRRNS_V, self).__init__()
        self.embedding    = nn.Embedding(vocab_size, embedding_dim)
        self.obs_encoder  = nn.GRU(embedding_dim, hidden_dim)
        self.look_encoder = nn.GRU(embedding_dim, hidden_dim)
        self.inv_encoder = nn.GRU(embedding_dim, hidden_dim)
        self.hidden       = nn.Linear(3*hidden_dim, hidden_dim)
        self.scorer   = nn.Linear(hidden_dim, 1)


    def packed_rnn(self, x, rnn):
        lengths = torch.tensor([len(n) for n in x], dtype=torch.long, device=device)
        # Sort this batch in descending order by seq length
        lengths, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        idx_sort = torch.autograd.Variable(idx_sort)
        idx_unsort = torch.autograd.Variable(idx_unsort)
        padded_x = pad_sequences(x)
        x_tt = torch.from_numpy(padded_x).type(torch.long).to(device)
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


    def forward(self, look_batch, inv_batch, obs_batch):
        # Encode the various aspects of the state
        obs_out = self.packed_rnn(obs_batch, self.obs_encoder)
        look_out = self.packed_rnn(look_batch, self.look_encoder)
        inv_out = self.packed_rnn(inv_batch, self.inv_encoder)
        z = torch.cat([look_out, inv_out, obs_out], dim=1)
        z = F.relu(self.hidden(z))
        act_values = self.scorer(z).squeeze(-1)
        return act_values
