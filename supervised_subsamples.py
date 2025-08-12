import os
os.environ["CUDA_DEVICE_ORDER"]   = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
current_dir = "/home/dvillacreses/code"
src_dir = os.path.join(current_dir, "src")
sys.path.append(src_dir)
os.chdir(current_dir)


from src.config import (dir_data_labeled, MAX_EPOCHS_SUPERVISED,
                        PATH_OUTPUTS, NUM_WORKERS_SUPERVISED)
from src.utils import seed_everything
from src.data_processing import (load_labeled_data, df_undersampling_strat, load_unlabeled_metadata,
                                 train_transform_labeled, val_transform_labeled,train_transform_labeled_vit,
                                 val_transform_labeled_vit, contrast_transforms, val_transform_labeled_simclr,
                                 ContrastiveTransformations, latent_dino_torch_dataset,torchdataset_to_dataframe,
                                 latent_simclr_vit_torch_dataset)
from sklearn.model_selection import RepeatedKFold, train_test_split

from src.datasets import UnlabelDataModule, LabeledDataModule, LabeledImageDataset
from src.models import (CNNLightningModule, ViTLightningModule, EmbeddingExtractor_VIT,
                        prepare_data_features, train_logreg, get_smaller_dataset,
                        MLPEvaluation, LinearEvaluation)
import random

import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32   = False
torch.set_float32_matmul_precision('highest')

from src.training import (train_supervised_model, train_supervised_model_v2,
                          train_simclr,classifier_ssl_trainer,
                          hpo_mlp,hpo_logistic)

from itertools import product
import time
import pickle
from tqdm import tqdm
import os

import pandas as pd
import numpy as np





## Data Loading
# df = load_labeled_data(dir_data_labeled)
df = pd.read_pickle('df.pkl')

# Subsamples
iter_list = list(product([0.1, 0.25, .5, .6, .7, .8, .9], range(25)))
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

    

## Supervised Grid Search
df_grid = pd.DataFrame(product(
    [0.0001],[32],[0.15],
    ['vit_base'],
    [0.1,0.25,.5, .6, .7, .8, .9], 
    range(25)
    ),
    columns = ['lr', 'batch_size','frozen_prop','model_name', 'prop_sample', "i_sample"]
    )



all_results_list = []
## Loop for grid_search
for i in range(df_grid.shape[0]):
    lr, batch_size, frozen_prop, model_name, prop_sample, i_sample = df_grid.iloc[i,:]
    subset_var=f'sample_prop{prop_sample}_v{i_sample}'
    subset_train_test = subset_var.replace('sample_prop','split_prop')
    df_balanced =  pd.concat([
            df_undersampling_strat(df[(df[subset_train_test]=='train') & (df[subset_var]==1)],subset_col='u',label_col='label'),
            df_undersampling_strat(df[(df[subset_train_test]=='valid') & (df[subset_var]==1)],subset_col='u',label_col='label')
            ]).copy()

    batch_size = int(batch_size)
    print("-"*50)
    print(f'Iter: {i+1}/{df_grid.shape[0]} - lr: {lr}, batch_size: {batch_size}, frozen_prop: {frozen_prop}, model_name: {model_name}, proportion of sample: {prop_sample}, number of sample: {i_sample}')
    print("-"*50)
    # Seed for reproducibility
    seed_everything(0)
    # Start datasets
    if 'vit' not in model_name :
        train_dataset_labeled = LabeledImageDataset(
            image_list = df_balanced.loc[df_balanced[subset_train_test]=='train','cropped_image'].to_list(),
            labels=df_balanced.loc[df_balanced[subset_train_test]=='train','label'].to_list(),
            transform=train_transform_labeled
            )
        valid_dataset_labeled = LabeledImageDataset(
            image_list = df_balanced.loc[df_balanced[subset_train_test]=='valid','cropped_image'].to_list(),
            labels=df_balanced.loc[df_balanced[subset_train_test]=='valid','label'].to_list(),
            transform=val_transform_labeled
            )
        test_dataset_labeled = LabeledImageDataset(
            image_list = df_balanced.loc[df_balanced[subset_train_test]=='test','cropped_image'].to_list(),
            labels=df_balanced.loc[df_balanced[subset_train_test]=='test','label'].to_list(),
            transform=val_transform_labeled
            )
    if 'vit' in model_name:
        train_dataset_labeled = LabeledImageDataset(
            image_list = df_balanced.loc[df_balanced[subset_train_test]=='train','cropped_image'].to_list(),
            labels=df_balanced.loc[df_balanced[subset_train_test]=='train','label'].to_list(),
            transform=train_transform_labeled_vit
            )
        valid_dataset_labeled = LabeledImageDataset(
            image_list = df_balanced.loc[df_balanced[subset_train_test]=='valid','cropped_image'].to_list(),
            labels=df_balanced.loc[df_balanced[subset_train_test]=='valid','label'].to_list(),
            transform=val_transform_labeled_vit
            )
        test_dataset_labeled = LabeledImageDataset(
            image_list = df_balanced.loc[df_balanced[subset_train_test]=='test','cropped_image'].to_list(),
            labels=df_balanced.loc[df_balanced[subset_train_test]=='test','label'].to_list(),
            transform=val_transform_labeled_vit
            )

    # Start lightning data module
    labeled_datamodule = LabeledDataModule(train_dataset_labeled,
                                        valid_dataset_labeled,
                                        test_dataset_labeled,
                                        batch_size=batch_size,
                                        num_workers=NUM_WORKERS_SUPERVISED)
    # Start model instance
    if 'vit' not in model_name:
        lightning_model = CNNLightningModule(learning_rate=lr,
                                            frozen_prop=frozen_prop,
                                            model_name=model_name)
    if 'vit' in model_name:
        lightning_model = ViTLightningModule(learning_rate=lr,
                                            frozen_prop=frozen_prop,
                                            model_name=model_name)
    # Train model
    results_list = train_supervised_model(lightning_model,
                                        labeled_datamodule,
                                        total_epochs=MAX_EPOCHS_SUPERVISED,
                                        model_name=model_name)
    # Save results in a list
    all_results_list.append(results_list)

    # Locally save after each iteration
    with open(os.path.join(PATH_OUTPUTS,'all_results_grid_supervised_subsamples_25.pkl'), 'wb') as file:
        pickle.dump(all_results_list, file)

