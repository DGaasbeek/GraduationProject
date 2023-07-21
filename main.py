import sys
import os
import numpy as np
import pandas as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pm4py

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import TensorDataset, DataLoader, random_split

from tqdm import tqdm
import time

from processor import *
from args import *
from models.transformer_norole import *
from models.transformer_role import *
from models.poac_transformer_role import *
from utils import *

#Some global variables

MILESTONE = 'All' #'A_PREACCEPTED' # 'W_Completeren aanvraag', 'All'
EXPERIMENT =  'OHE'#'Standard'#'OHE', 'No_loops'
N_SIZE = 5
MAX_SIZE = 1000 #  3, 5, 10, 15, 20, 30, 50, 95
MIN_SIZE = 0  # 0, 3, 5, 10, 15, 20, 30, 50
BATCH_SIZE = 64

TASK = 'next_act_time' #next_act_time' 'binary_poac' 'suffix' 'suffix_poac'

DATASET = 'synthetic_post'
MY_WORKSPACE_DIR = os.path.join(os.path.join(os.getcwd(), 'datasets'), DATASET)
MILESTONE_DIR = os.path.join(MY_WORKSPACE_DIR, MILESTONE)
args = get_parameters(DATASET.lower(), MILESTONE_DIR, MY_WORKSPACE_DIR, MILESTONE, N_SIZE, MAX_SIZE, MIN_SIZE)

model_path = 'transformer_role_5n_0.5reg_76epochs.pt'

INCL_RES = False #if args['log_name'] == 'helpdesk' or args['log_name'] == 'synthetic' else True
print(INCL_RES)
#args['test_traces'] = os.path.join(MILESTONE_DIR, f'{MILESTONE}_test.csv') if MILESTONE != 'All' else args['test_traces']

#Preprocess the data
processor = Processor(MY_WORKSPACE_DIR, MILESTONE_DIR, args)

vec_train, vec_test, weights, indexes, pre_index = processor.process(cont_features=['timelapsed', 'next_time'])

print(vec_test['poac_time'][:10])
print(vec_test['prefixes']['x_ac_inp'].shape[0])

index_ac = indexes['index_ac']
index_rl = indexes['index_rl']
index_ne = indexes['index_ne']

ac_index = pre_index['ac_index']
rl_index = pre_index['rl_index']
ne_index = pre_index['ne_index']

y_scaler2 = weights['y_scaler2']

if TASK == 'next_act_time':
    if INCL_RES:
        model = next_act_time_role(vec_train, vec_test, args, BATCH_SIZE, weights, y_scaler2, N_SIZE, MILESTONE_DIR=MILESTONE_DIR)
    else:
        model = next_act_time_norole(vec_train, vec_test, args, BATCH_SIZE, weights, y_scaler2, N_SIZE, MILESTONE_DIR=MILESTONE_DIR)
elif TASK == 'binary_poac':
    model = binary_poac(vec_train, vec_test, args, BATCH_SIZE, weights, N_SIZE, MILESTONE_DIR=MILESTONE_DIR)
elif TASK == 'binary_poac_time':
    model = binary_poac_time(vec_train, vec_test, args, BATCH_SIZE, weights, N_SIZE, MILESTONE_DIR=MILESTONE_DIR)
elif TASK == 'suffix':
    sizes = [1,3,5,7,10,15]
    if INCL_RES:
        _, results = suffix_pred_role(args, ac_index, index_ac, rl_index, weights, MILESTONE_DIR, DATASET, scaler=y_scaler2, beam_sizes=sizes, model_path=model_path)
        print(f'Saving the results...')
        results.to_csv(args['suff_pred_res'])
    else:
        _, results = suffix_pred_nr(args, ac_index, index_ac, weights, MILESTONE_DIR, DATASET, beam_sizes=sizes, model_path=model_path)
        print(f'Saving the results...')
        results.to_csv(args['suff_pred_res'])
elif TASK == 'suffix_poac':
    # model_path = 'transformer_role_5n_0.5reg_47epochs.pt'
    suffixes = suffix_poac(model_path, args, weights, MILESTONE_DIR, DATASET, ac_index, index_ac, rl_index, MILESTONE=MILESTONE)
elif TASK == 'suffix_poac_time':
    # model_path = 'transformer_role_5n_0.5reg_47epochs.pt'
    suffixes = suffix_poac_time(model_path, args, weights, MILESTONE_DIR, DATASET, ac_index, index_ac, rl_index, scaler=y_scaler2, MILESTONE=MILESTONE)
    suffixes_df = pd.DataFrame(data=suffixes)
    output_path = os.path.join(MILESTONE_DIR, "poac_time_preds.csv")
    suffixes_df.to_csv(output_path)
    print(suffixes[0])
else:
    raise Exception(f"The task {TASK} is invalid. Choose another task.")


