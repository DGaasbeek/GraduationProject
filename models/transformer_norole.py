import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from sklearn.metrics import mean_absolute_error, precision_recall_fscore_support, accuracy_score
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split

import heapq

device = "cuda" if torch.cuda.is_available() else "cpu"

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_length=5):
        super().__init__()
        self.max_length = max_length
        self.d_model = d_model

    def forward(self, x):
        even_i = torch.arange(0, self.d_model, 2)
        denominator = torch.pow(10000, even_i / self.d_model)
        position = torch.arange(self.max_length).reshape(self.max_length, 1)
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        PE = PE[:, :self.d_model]
        # PE = PE.to(device)

        return x + PE

class TransformerBlock(nn.Module):
    def __init__(self, d_hidden, d_model, full_dims, n_heads, dropout):
        super().__init__()

        self.d_hidden = d_hidden
        self.d_model = d_model
        self.full_dims = full_dims
        self.n_heads = n_heads
        self.dropout = dropout

        self.multihead = nn.MultiheadAttention(self.full_dims, self.n_heads, dropout=dropout, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(self.full_dims, self.d_hidden),
            nn.ReLU(),
            nn.Linear(self.d_hidden, self.full_dims)
        )

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, input):
        #Input: (B, T, n*d_model)
        B,T,D = input.shape

        att_output, att_weights = self.multihead(input, input, input, average_attn_weights=False)
        att_output = self.dropout_layer(att_output)

        #Skip connection
        B, T, D = att_output.shape
        if torch.cuda.is_available():
            layernorm_a = nn.LayerNorm([D]).cuda()
        else:
            layernorm_a = nn.LayerNorm([D])

        out_a = layernorm_a(input + att_output)  # Output dim: (B, T, n*d_model)

        c_t = self.feedforward(out_a)  # Output dim: (B, T, n*d_model)
        c_t = self.dropout_layer(c_t)

        if torch.cuda.is_available():
            layernorm_b = nn.LayerNorm([D]).cuda()
        else:
            layernorm_b = nn.LayerNorm([D])

        c_t = layernorm_b(out_a + c_t) #Output: (B, T, n*d_model)

        return c_t, att_weights

class Transformer_NR(nn.Module):
    def __init__(self, ac_weights, next_activity, dropout, d_model, d_hidden, n_size):
        super().__init__()

        self.ac_weights = ac_weights
        self.next_activity = next_activity
        self.dropout = dropout
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.n_size = n_size


        self.full_dims = 2 * self.d_model
        self.n_heads = 2

        self.ac_vocab = ac_weights.shape[0]
        self.ac_embedding = nn.Embedding(self.ac_vocab, self.d_model)
        self.t_transform = nn.Linear(1, self.d_model)

        self.pos_encoding = PositionalEncoding(self.full_dims,
                                               max_length=self.n_size)  # This positional encoding differs from the one used in Bukhsh

        self.transformer_block = TransformerBlock(self.d_hidden, self.d_model, self.full_dims,
                                                  self.n_heads, self.dropout)

        self.dropout_layer = nn.Dropout(dropout)

        self.ff_act = nn.Sequential(
            nn.Linear(self.full_dims, self.d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_hidden, next_activity)
        )

        self.ff_time = nn.Sequential(
            nn.Linear(3 * self.d_model, self.d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.time_output = nn.Linear(self.d_hidden, 1)


    def forward(self, x_ac, x_t):
        x_ac = x_ac.long()
        ac_embs = self.ac_embedding(x_ac)  # (B, T) --> (B, T, D_model)

        x_t = x_t.unsqueeze(-1)  # (B, T) --> (B, T, 1)
        t_embs = self.t_transform(x_t.to(self.t_transform.weight.dtype))  # (B, T, 1) --> (B, T, D_model)

        full_embs = torch.cat([ac_embs, t_embs], dim=-1)  # (B, T, D_model * 2)
        full_embs = self.pos_encoding(full_embs.float())

        c_t, att_weights = self.transformer_block(full_embs) # (B, T, D_model * n) and #(B,H,S,S)

        context = torch.sum(c_t, dim = 1) # (B, T, D_model * n)
        context = self.dropout_layer(context)

        #Feedforward + Output linear layer
        act_output = self.ff_act(context)

        fft_in = torch.cat([c_t, t_embs], dim=-1)  # Concatenating two tensors to form (B, T, _)

        fft_out = self.ff_time(fft_in) #In (B, T, n+m * d_model), out (B, T, d_hidden)
        fft_out = torch.sum(fft_out, dim=1)
        time_output = self.time_output(fft_out)


        return act_output, time_output, att_weights

    # def extract_combination_weights(self):



def model_training(model, optimizer, train_data, val_data, test_data, MILESTONE_DIR, epochs=10, patience=15,
                   scheduler=None, N_SIZE=5):
    print(f'The device is {device}')

    criterion = nn.CrossEntropyLoss()

    train_losses = []
    train_accs = []
    train_maes = []
    val_losses = []
    val_accs = []
    val_maes = []

    reg_share = 0.5
    counter1 = 0
    counter2 = 0
    best_val_loss = float('inf')
    best_val_mae = float('inf')

    start = time.time()
    for epoch in range(epochs):
        train_loss, train_acc, train_mae = train_loop_nr(model, train_data, criterion, reg_share, optimizer, scheduler)
        val_loss, val_acc, val_mae, _ = test_loop_nr(model, val_data, criterion)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_maes.append(train_mae)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_maes.append(val_mae)

        scheduler.step()

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            counter1 = 0
        else:
            counter1 += 1
            if counter1 >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

        print(f'Epoch: {epoch + 1}, train loss = {train_loss:.4f}, train accuracy = {train_acc:.4f}, train MAE = {train_mae:.4f}, '
              f'val loss = {val_loss:.4f}, val accuracy = {val_acc:.4f}, val MAE = {val_mae:.4f}, LR = {scheduler.get_last_lr()}')
    end = time.time()
    training_time = (end-start) / 60
    s_epoch = training_time/(epoch+1) * 60

    print(f'Training done, total training time = {training_time:.4f} minutes, {s_epoch:.4f} seconds per epoch')
    path = os.path.join(os.path.join(MILESTONE_DIR, 'models'),
                        f'transformer_norole_{N_SIZE}n_{reg_share}reg_{epoch + 1}epochs.pt')
    torch.save(model.state_dict(), path)
    print(f'Model saved at {path}')

    return model

def model_evaluation(model, vec_test, args, y_scaler2):

    start = time.time()
    X_test, y_test_act, y_test_time = generate_inputs_nr(vec_test, args)
    out_act, out_t, out_att = model(X_test[0].long(), X_test[1])

    out_act = np.array(out_act.argmax(1).detach()).reshape(-1,1)
    out_t = np.array(out_t.squeeze(-1).detach()).reshape(-1,1)
    _out_t = y_scaler2.inverse_transform(out_t)

    act_targets = np.array(y_test_act.argmax(1).detach()).reshape(-1,1)
    t_targets = np.array(y_test_time.squeeze(-1).detach()).reshape(-1,1)
    _t_targets = y_scaler2.inverse_transform(t_targets)

    MAE = mean_absolute_error(_out_t, _t_targets)
    accuracy = accuracy_score(act_targets, out_act)
    precision, recall, f1, _ = precision_recall_fscore_support(act_targets, out_act, average='weighted')
    end = time.time()

    inference_time = (end-start) / 60

    print(f"Accuracy: {accuracy:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}")
    print(f"MAE: {MAE:.4f}")
    print(f"Inference time: {inference_time:.4f}")



def train_loop_nr(model, train_data, criterion, reg_share, optimizer, scheduler=None):
    train_loss = 0.0
    train_acc = 0.0
    train_mae = 0.0
    num_correct = 0
    num_examples = 0

    crit_reg = nn.L1Loss()

    model.train()

    for batch in tqdm(train_data):
        # Clear the gradients
        optimizer.zero_grad()

        x_ac, x_t, y_act, y_time = batch
        # x_ac, x_t, y_act, y_time = x_ac.to(device), x_t.to(device), y_act.cuda().to(device), y_time.to(device)

        # Forward pass
        output_act, output_time, _ = model(x_ac=x_ac.long(), x_t=x_t)

        # Find the loss
        loss_act = criterion(output_act, y_act.argmax(1))
        loss_time = crit_reg(output_time, y_time.unsqueeze(-1))

        loss = (1 - reg_share) * loss_act + reg_share * loss_time
        # Backpropagate
        loss.backward()
        optimizer.step()

        _, preds = torch.max(output_act, 1)

        num_correct += (preds == y_act.argmax(1)).sum().item()
        num_examples += y_act.size(0)
        train_loss += loss_act.item()
        train_mae += loss_time.item()

    train_loss /= len(train_data)
    train_acc = num_correct / num_examples
    train_mae /= len(train_data)

    return train_loss, train_acc, train_mae


def test_loop_nr(model, test_data, criterion):
    test_loss = 0.0
    test_acc = 0.0
    test_mae = 0.0
    num_correct = 0
    num_examples = 0

    crit_reg = nn.L1Loss()

    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(test_data):

            x_ac, x_t, y_act, y_time = batch

            # x_ac, x_t, y_act, y_time = x_ac.to(device), x_t.to(device), y_act.to(device), y_time.to(device)

            output_act, output_time, _ = model(x_ac=x_ac.long(), x_t=x_t)

            loss_act = criterion(output_act, y_act.argmax(1))
            loss_time = crit_reg(output_time, y_time.unsqueeze(-1))
            loss = loss_act + loss_time

            _, preds = torch.max(output_act, 1)

            num_correct += (preds == y_act.argmax(1)).sum().item()
            num_examples += y_act.size(0)
            test_loss += loss_act.item()
            test_mae += loss_time.item()

            if i == 1:
                sample = output_time

    test_loss /= len(test_data)
    test_acc = num_correct / num_examples
    test_mae /= len(test_data)

    return test_loss, test_acc, test_mae, sample

def generate_inputs_nr(vec, args):
    #Removed indexes from input
    # index_ac = indexes['index_ac']
    # index_ne = indexes['index_ne']

    prefix_len = args['prefix_length']

    if prefix_len == 'fixed':
        MAX_LEN = args['n_size']
    else:
        MAX_LEN = vec['prefixes']['x_ac_inp'].shape[1]

    x = [vec['prefixes']['x_ac_inp'][:, :MAX_LEN]]
    x.append(vec['prefixes']['xt_inp'][:, :MAX_LEN])
    x = torch.tensor(x)

    y_act = torch.tensor(vec['next_activity'])
    y_time = torch.tensor(vec['next_time'])

    return x, y_act, y_time

def get_dataloaders(vec_train, vec_test, args, batch_size=64):
    X_train, y_train_act, y_train_time = generate_inputs_nr(vec_train, args)
    X_test, y_test_act, y_test_time = generate_inputs_nr(vec_test, args)

    train_dataset = TensorDataset(X_train[0], X_train[1], y_train_act, y_train_time)

    val_perc = 0.2
    val_size = int(np.round(val_perc*len(train_dataset)))
    train_size = len(train_dataset) - val_size

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    test_dataset = TensorDataset(X_test[0], X_test[1], y_test_act, y_test_time)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f'Train cases: {len(train_dataset)}, validation cases: {len(val_dataset)}, test cases: {len(test_dataset)}')

    return train_dataloader, val_dataloader, test_dataloader


def prepare_data_suff_nr(test_path):
    test_df = pd.read_csv(test_path)
    pd.set_option('display.max_columns', None)

    test_df['timelapsed'] = pd.to_datetime(test_df['end_timestamp']) - pd.to_datetime(test_df['start_time'])
    test_df['timelapsed'] = test_df['timelapsed'].dt.total_seconds() / (60 * 60 * 24)
    test_df['timelapsed'] = test_df['timelapsed'].round(0)
    test_df.loc[:, 'task'] = test_df['task'].astype(str)

    return test_df

def create_pref_suff_nr(df, ac_index):
    prefixes = list()
    cases = df.caseid.unique()
    print('Number of cases', len(cases))

    for case in cases:
        trace = df[df.caseid == case].to_dict('records')

        ac_pref = list()
        t_pref = list()

        for i in range(0, len(trace) - 1):
            ac_pref.append(ac_index[trace[i]['task']])  # Adds the task itself, not its index as in index_ac or ac_index
            t_pref.append(trace[i]['timelapsed'])
            prefixes.append(dict(ac_pref=ac_pref.copy(),
                                 ac_suff=[ac_index[x['task']] for x in trace[i + 1:]],
                                 t_pref=t_pref.copy(),
                                 rem_time=[x['timelapsed'] for x in trace[i + 1:]],
                                 pref_size=i + 1))

    for x in prefixes:
        x['ac_suff'].append(ac_index['EOS'])
        x['rem_time'].append(0)

    return prefixes

def chunks(lst, n):
    """yield successive n-sized chunks from lst"""
    for i in range(0, len(lst), n):
        yield lst[i, i + n]


def predict_suffixes_nr(model, prefixes, max_trc_length, time_dim, index_ac, scaler):
    count = 0
    for prefix in tqdm(prefixes):
        count += 1

        # Act input shape (1,n)
        x_ac_n = np.append(np.zeros(time_dim),
                           np.array(prefix['ac_pref']),
                           axis=0)[-time_dim:].reshape((1, time_dim))

        # Time1 input shape (1,n)
        x_t_n = np.append(np.zeros(time_dim),
                          np.array(prefix['t_pref']), axis=0)[-time_dim:].reshape((1, time_dim))

        acum_tbtw = 0
        ac_suf, t_suf = list(), list()

        warnings.filterwarnings("ignore", category=FutureWarning)
        # 100 = max_trace_size
        for _ in range(1, max_trc_length):

            a_inp = np.flip(x_ac_n, axis=1).copy()
            t_inp = np.flip(x_t_n, axis=1).copy()

            pred_act, pred_time, _ = model(x_ac=torch.tensor(a_inp), x_t=torch.tensor(t_inp))

            pos = torch.argmax(pred_act, dim=1)
            t_next = pred_time.detach().squeeze(0).squeeze(0)
            t_next = scaler.inverse_transform(t_next.reshape(-1,1)) #No clue if this scalar is correct like this
            # print(t_next)

            x_ac_n = np.append(x_ac_n, [[pos]], axis=1)
            x_ac_n = np.delete(x_ac_n, 0, 1)
            x_t_n = np.append(x_t_n, t_next, axis=1)
            x_t_n = np.delete(x_t_n, 0, 1)

            ac_suf.append(pos.item())
            t_suf.append(t_next.item())

            if index_ac[pos.item()] == 'EOS':
                break

        prefix['ac_suff_pred'] = ac_suf
        prefix['t_suff_pred'] = t_suf

        warnings.resetwarnings()

    return prefixes

def predict_beam_nr(model, prefixes, max_trc_length, time_dim, index_ac, beam_size=3):
    count = 0
    beam_size = beam_size if len(index_ac.keys()) >= beam_size else len(index_ac.keys()) - 1

    for prefix in tqdm(prefixes):
        count += 1

        # Act input shape (1,n)
        x_ac_n = np.append(np.zeros(time_dim),
                           np.array(prefix['ac_pref']),
                           axis=0)[-time_dim:].reshape((1, time_dim))

        # Time1 input shape (1,n)
        x_t_n = np.append(np.zeros(time_dim),
                          np.array(prefix['t_pref']), axis=0)[-time_dim:].reshape((1, time_dim))

        ac_suf, t_suf = list(), list()

        beam = [(1, x_ac_n, x_t_n, ac_suf, t_suf)]
        completed = []

        for _ in range(max_trc_length):
            new_beam = []

            for score, x_ac_n, x_t_n, ac_suf, t_suf in beam:
                a_inp = np.flip(x_ac_n, axis=1).copy()
                t_inp = np.flip(x_t_n, axis=1).copy()

                pred_act, pred_time, _ = model(x_ac=torch.tensor(a_inp), x_t=torch.tensor(t_inp))

                t_next = pred_time.detach().squeeze(0).squeeze(0).item()
                logits = torch.nn.Softmax(dim=1)(pred_act).detach().squeeze(0)
                top_n_val, top_n_id = torch.topk(logits, k=beam_size, dim=0)

                for i in range(beam_size):
                    pos = top_n_id[i].item()
                    prob = top_n_val[i].item()

                    new_x_ac_n = np.append(x_ac_n, [[pos]], axis=1)
                    new_x_ac_n = np.delete(new_x_ac_n, 0, 1)
                    new_x_t_n = np.append(x_t_n, [[t_next]], axis=1)
                    new_x_t_n = np.delete(new_x_t_n, 0, 1)
                    new_ac_suf = ac_suf + [pos]
                    new_t_suf = t_suf + [t_next]

                    if index_ac[pos] == 'EOS':
                        completed.append((score * prob, new_ac_suf, new_t_suf))
                    else:
                        new_beam.append((score * prob, new_x_ac_n, new_x_t_n, new_ac_suf, new_t_suf))

            if len(completed) >= beam_size:
                break

            if not completed:
                best = max(beam, key=lambda x: x[0])
                prefix['ac_suff_pred'] = best[3]
                prefix['t_suff_pred'] = best[4]

            beam = heapq.nlargest(beam_size, new_beam, key=lambda x: x[0])

            # Sort and select the top completed sequences
        if len(completed) >= beam_size:
            completed = sorted(completed, key=lambda x: x[0], reverse=True)
            best = completed[0]
            prefix['ac_suff_pred'] = best[1]
            prefix['t_suff_pred'] = best[2]

    return prefixes