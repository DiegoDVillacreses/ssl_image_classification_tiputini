import os
os.environ["CUDA_DEVICE_ORDER"]   = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,3,6"

import sys
current_dir = "/home/dvillacreses/code"
src_dir = os.path.join(current_dir, "src")
sys.path.append(src_dir)
os.chdir(current_dir)


from src.config import dir_data_labeled
from src.utils import seed_everything
from src.data_processing import (load_labeled_data, df_undersampling_strat, load_unlabeled_metadata, 
                                 train_transform_labeled, val_transform_labeled,train_transform_labeled_vit, 
                                 val_transform_labeled_vit, contrast_transforms, val_transform_labeled_simclr, 
                                 ContrastiveTransformations, latent_dino_torch_dataset,torchdataset_to_dataframe,
                                 latent_simclr_vit_torch_dataset)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold, train_test_split

from src.datasets import UnlabelDataModule, LabeledDataModule, LabeledImageDataset
from src.models import (CNNLightningModule, ViTLightningModule, EmbeddingExtractor_VIT, 
                        prepare_data_features, train_logreg, get_smaller_dataset,
                        MLPEvaluation, LinearEvaluation)
from scipy.stats import ttest_ind
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import random

import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32   = False
torch.set_float32_matmul_precision('highest')

from src.training import (train_supervised_model, train_supervised_model_v2, 
                          train_simclr,classifier_ssl_trainer,
                          hpo_mlp)

from itertools import product
import time
import pickle
from tqdm import tqdm
import os

import pandas as pd
import numpy as np


import multiprocessing as mp
from multiprocessing import Pool

# Global variables
TRAIN_GRID_SUPERVISED = False
TRAIN_ENCODER_SIMCLR = False
TRAIN_ENCODER_SIMCLR_VIT = False
TRAIN_HPO_SSL_CLASSIFIER = True
TRAIN_GRID_SIMCLR_CLASSIFIER = False
TRAIN_GRID_SIMCLR_VIT_CLASSIFIER = False
COMPUTE_STATISTICAL_COMPARISON = False


## Data Loading
# Load and prepare data
df = load_labeled_data(dir_data_labeled)
df['u']=1
df_balanced = pd.concat([
    df_undersampling_strat(df[df['subset']=='train'],subset_col='u',label_col='label'),
    df_undersampling_strat(df[df['subset']=='valid'],subset_col='u',label_col='label'),
    df_undersampling_strat(df[df['subset']=='test'],subset_col='u',label_col='label')
])


# For subsamples
iter_list = list(product([.5], range(15)))
for fraction, seed in iter_list:
    sampled = (
        df[df['subset']!='test']
        .groupby('label', group_keys=False)
        .sample(frac=fraction, random_state=seed)
    )
    train_idx, val_idx = train_test_split(
        sampled.index,
        train_size=0.8,
        stratify=sampled['label'],
        random_state=seed
    )
    col = f'split_prop{fraction}_v{seed}'
    df[col] = ""
    df.loc[train_idx, col] = 'train'
    df.loc[val_idx,   col] = 'valid'
    df[f'sample_prop{fraction}_v{seed}'] = df.index.isin(sampled.index).astype(int)

#--------------------------------------------------------------#
# ### HPO for all MLP Classifiers

all_ssl_models = [i for i in os.listdir('outputs') if ('.pkl' in i) & (('simclr' in i) | ('dino' in i))]
all_ssl_models = sorted(all_ssl_models)


iter_list = list(product([.5], range(5) ,all_ssl_models))
iter_list = [list + (torch.device(f"cuda:{i%3}"),) for i, list in enumerate(iter_list)]


iter_list_all = []
for iter in iter_list:
    fraction,seed,ssl_name,dev = iter

    subset_var=f'sample_prop{fraction}_v{seed}'
    subset_train_test = subset_var.replace('sample_prop','split_prop')
    list_in = [
        "./outputs/optuna_logs",
        pd.concat([
            df_undersampling_strat(df[(df[subset_train_test]=='train') & (df[subset_var]==1)],subset_col='u',label_col='label'),
            df_undersampling_strat(df[(df[subset_train_test]=='valid') & (df[subset_var]==1)],subset_col='u',label_col='label'),
        ]),
        ssl_name,
        subset_var,
        dev
        ]
    iter_list_all.append(list_in)

if __name__ == "__main__":
    print("Starting HPO...")
    mp.set_start_method("spawn", force=True)
    max_procs = 4
    with Pool(processes=max_procs) as pool:
        results = pool.map(hpo_mlp, iter_list_all)