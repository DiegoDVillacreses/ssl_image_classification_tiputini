import os
os.environ["CUDA_DEVICE_ORDER"]   = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,3,6"

import sys
current_dir = "/home/dvillacreses/code"
src_dir = os.path.join(current_dir, "src")
sys.path.append(src_dir)
os.chdir(current_dir)


from src.config import (dir_data_labeled,PATH_RESULTS)
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

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec

import optuna
import multiprocessing as mp
from multiprocessing import Pool

# ## Data Loading
df = load_labeled_data(dir_data_labeled)
    # df.shape=(7337, 6)

# Full sample
df['u']=1
df_balanced = pd.concat([
    df_undersampling_strat(df[df['subset']=='train'],subset_col='u',label_col='label'),
    df_undersampling_strat(df[df['subset']=='valid'],subset_col='u',label_col='label'),
    df_undersampling_strat(df[df['subset']=='test'],subset_col='u',label_col='label')
])

## Save summary of results
def save_csv_results():
    folder_path = "outputs/optuna_logs"
    all_files = []

    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            all_files.append(full_path)
    print(len(all_files))

    l = []
    l2 = []
    for i in all_files:
        model_name = i.split("/")[2]
        sample_name = i.split("/")[3]
        if sample_name == 'full_sample_or':
            
            try:
                storage = f"sqlite:////home/dvillacreses/code/outputs/optuna_logs/{model_name}/{sample_name}/my_study.db"
                study_names = optuna.get_all_study_names(storage=storage)
                s = optuna.load_study(study_name=study_names[0], storage=storage)
                l.append(s.best_trial.value )
                l2.append([model_name,sample_name]+ list(s.best_params.values()) + [s.trials_dataframe().shape[0]])
            except:
                l.append(np.nan)

    df = pd.DataFrame(l2, columns=['model','prop','mlp_batch_size', 'mlp_lr', 'mlp_weight_decay', 'mlp_dropout','mlp_total_iterations_bhpo'])
    df['val_acc'] = l
    df = df.sort_values('val_acc',ascending=False).reset_index(drop=True)
    df.to_csv(os.path.join(PATH_RESULTS,'ssl_hpo_mlp_full_sample.csv'), index=False)

#--------------------------------------------------------------#
# ### HPO for all MLP Classifiers

all_ssl_models = [i for i in os.listdir('outputs') if ('.pkl' in i) & (('simclr' in i) | ('dino' in i))]
all_ssl_models = sorted(all_ssl_models)
all_ssl_models = [[torch.device(f"cuda:{i%3}"), list] for i, list in enumerate(all_ssl_models)]

iter_list_all = []
subset_var='full_sample_or'

for dev, ssl_name in all_ssl_models:
    list_in = [
        "./outputs/optuna_logs",
        df_balanced,
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

    save_csv_results()