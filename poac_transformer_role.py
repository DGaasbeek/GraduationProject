import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from sklearn.metrics import mean_absolute_error, precision_recall_fscore_support, accuracy_score, recall_score, precision_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split

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

class Transformer_Poac(nn.Module):
    def __init__(self, ac_weights, rl_weights, next_activity, dropout, d_model, d_hidden, n_size):
        super().__init__()

        self.ac_weights = ac_weights
        self.rl_weights = rl_weights
        self.next_activity = next_activity
        self.dropout = dropout
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.n_size = n_size

        self.full_dims = 3 * self.d_model
        self.n_heads = 3

        self.ac_vocab = ac_weights.shape[0]
        self.ac_embedding = nn.Embedding(self.ac_vocab, self.d_model)
        self.rl_vocab = rl_weights.shape[0]
        self.rl_embedding = nn.Embedding(self.rl_vocab, self.d_model)
        self.t_transform = nn.Linear(1, self.d_model)

        self.pos_encoding = PositionalEncoding(self.full_dims,
                                               max_length=self.n_size)  # This positional encoding differs from the one used in Bukhsh

        self.transformer_block = TransformerBlock(self.d_hidden, self.d_model, self.full_dims,
                                                  self.n_heads, self.dropout)

        self.ff_poac = nn.Sequential(
            nn.Linear(self.full_dims, self.d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.dropout_layer = nn.Dropout(dropout)

        self.poac_output = nn.Sequential(
            nn.Linear(self.d_hidden, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x_ac, x_rl, x_t):
        x_ac = x_ac.long()
        ac_embs = self.ac_embedding(x_ac)  # (B, T) --> (B, T, D_model)

        x_rl = x_rl.long()
        rl_embs = self.rl_embedding(x_rl)

        x_t = x_t.unsqueeze(-1)  # (B, T) --> (B, T, 1)
        t_embs = self.t_transform(x_t.to(self.t_transform.weight.dtype))  # (B, T, 1) --> (B, T, D_model)

        full_embs = torch.cat([ac_embs, rl_embs, t_embs], dim=-1)  # (B, T, D_model * 2)
        full_embs = self.pos_encoding(full_embs.float())

        c_t, att_weights = self.transformer_block(full_embs) # (B, T, D_model * n) and #(B,H,S,S)

        context = torch.sum(c_t, dim = 1) # (B, T, D_model * n)
        context = self.dropout_layer(context)

        poac_out = self.ff_poac(context)
        poac_output = self.poac_output(poac_out)

        return poac_output, att_weights

def poac_model_training_role(model, optimizer, train_data, val_data, test_data, MILESTONE_DIR, epochs=10, patience=15,
                             scheduler=None, N_SIZE=5):
    print(f'The device is {device}')

    criterion = nn.CrossEntropyLoss()

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    counter1 = 0
    counter2 = 0
    best_val_loss = float('inf')

    start = time.time()
    for epoch in range(epochs):
        train_loss, train_acc = poac_train_loop_role(model, train_data, criterion, optimizer, scheduler)
        val_loss, val_acc = poac_test_loop_role(model, val_data, criterion)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter1 = 0
        else:
            counter1 += 1
            if counter1 >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

        print(f'Epoch: {epoch + 1}, train loss = {train_loss:.4f}, train accuracy = {train_acc:.4f}'
              f'val loss = {val_loss:.4f}, val accuracy = {val_acc:.4f}, LR = {scheduler.get_last_lr()}')
    end = time.time()
    training_time = (end-start) / 60
    s_epoch = training_time/(epoch+1) * 60

    print(f'Training done, total training time = {training_time:.4f} minutes, {s_epoch:.4f} seconds per epoch')
    path = os.path.join(os.path.join(MILESTONE_DIR, 'models'),
                        f'poac_transformer_role_{N_SIZE}n__{epoch + 1}epochs.pt')
    torch.save(model.state_dict(), path)
    print(f'Model saved at {path}')

    return model

def poac_model_evaluation_role(model, vec_test, args): #todo:

    start = time.time()
    X_test, y_test_poac = generate_inputs_poac(vec_test, args)
    out_poac, out_att = model(X_test[0].long(), X_test[1].long(), X_test[2])

    out_poac = np.array(out_poac.argmax(1).detach()).reshape(-1,1)
    poac_targets = np.array(y_test_poac.argmax(1).detach()).reshape(-1,1)

    accuracy = accuracy_score(poac_targets, out_poac)
    precision = precision_score(poac_targets, out_poac)
    recall = recall_score(poac_targets, out_poac)
    f1 = f1_score(poac_targets, out_poac)
    # precision, _, f1, _ = precision_recall_fscore_support(poac_targets, out_poac, average='weighted')
    end = time.time()

    inference_time = (end-start) / 60

    print(f"Accuracy: {accuracy:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}")
    print(f"Inference time: {inference_time:.4f}")


def poac_train_loop_role(model, train_data, criterion, optimizer, scheduler=None):
    train_loss = 0.0
    train_acc = 0.0
    num_correct = 0
    num_examples = 0

    model.train()

    for batch in tqdm(train_data):
        # Clear the gradients
        optimizer.zero_grad()

        x_ac, x_rl, x_t, y_poac = batch
        # x_ac, x_rl, x_t, y_act, y_time = x_ac.to(device), x_rl.to(device), x_t.to(device), y_act.cuda().to(device), y_time.to(device)

        # Forward pass
        output_poac, _ = model(x_ac=x_ac.long(), x_rl=x_rl.long(), x_t=x_t)

        # Find the loss
        loss_poac = criterion(output_poac, y_poac.argmax(1))

        # Backpropagate
        loss_poac.backward()
        optimizer.step()

        _, preds = torch.max(output_poac, 1)

        num_correct += (preds == y_poac.argmax(1)).sum().item()
        num_examples += y_poac.size(0)
        train_loss += loss_poac.item()

    train_loss /= len(train_data)
    train_acc = num_correct / num_examples

    return train_loss, train_acc


def poac_test_loop_role(model, test_data, criterion):
    test_loss = 0.0
    test_acc = 0.0
    num_correct = 0
    num_examples = 0

    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(test_data):

            x_ac, x_rl, x_t, y_poac = batch

            # x_ac, x_t, x_rl, y_act, y_time = x_ac.to(device), x_t.to(device), x_rl.to(device), y_act.to(device), y_time.to(device)

            output_poac, _ = model(x_ac=x_ac.long(), x_rl=x_rl.long(), x_t=x_t)

            loss_poac = criterion(output_poac, y_poac.argmax(1))

            _, preds = torch.max(output_poac, 1)

            num_correct += (preds == y_poac.argmax(1)).sum().item()
            num_examples += y_poac.size(0)
            test_loss += loss_poac.item()

    test_loss /= len(test_data)
    test_acc = num_correct / num_examples

    return test_loss, test_acc

def generate_inputs_poac(vec, args):

    prefix_len = args['prefix_length']

    if prefix_len == 'fixed':
        MAX_LEN = args['n_size']
    else:
        MAX_LEN = vec['prefixes']['x_ac_inp'].shape[1]

    x = [vec['prefixes']['x_ac_inp'][:, :MAX_LEN]]
    x.append(vec['prefixes']['x_rl_inp'][:, :MAX_LEN])
    x.append(vec['prefixes']['xt_inp'][:, :MAX_LEN])
    x = torch.tensor(x)

    y_poac = torch.tensor(vec['poac'])

    return x, y_poac

def get_dataloaders_poac(vec_train, vec_test, args, batch_size=64):
    X_train, y_train_poac = generate_inputs_poac(vec_train, args)
    X_test, y_test_poac = generate_inputs_poac(vec_test, args)

    train_dataset = TensorDataset(X_train[0], X_train[1], X_train[2], y_train_poac)

    val_perc = 0.2
    val_size = int(np.round(val_perc*len(train_dataset)))
    train_size = len(train_dataset) - val_size

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    test_dataset = TensorDataset(X_test[0], X_test[1], X_test[2], y_test_poac)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f'Train cases: {len(train_dataset)}, validation cases: {len(val_dataset)}, test cases: {len(test_dataset)}')

    return train_dataloader, val_dataloader, test_dataloader