import os
# script_dir = os.path.dirname(os.path.abspath(__file__))
# os.chdir(script_dir)
import gc

import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import torch
import optuna
import pandas as pd
from config import PATIENCE, PATH_SUPERVISED_MODELS, DEVICE
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, RichProgressBar
from pytorch_lightning.loggers import CSVLogger
from time import time
from models import SimCLR, SimCLR_ViT, MLPEvaluation
from src.datasets import LabeledImageDataset, LabeledDataModule
from data_processing import latent_dino_torch_dataset, latent_simclr_vit_torch_dataset,val_transform_labeled_simclr
from utils import seed_everything

def train_supervised_model(lightning_model, 
                           labeled_datamodule,
                           total_epochs = 200, 
                           model_name = "model_name"):
    t0 = time()
    # Define Callbacks
    callback_check = ModelCheckpoint(
        dirpath=PATH_SUPERVISED_MODELS, 
        save_top_k=1,
        mode="min",
        monitor="val_loss",
        filename="best-checkpoint",
        save_weights_only=True,
        verbose=True
    )
    callback_tqdm = RichProgressBar(leave=True)
    callback_early_stop = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        mode="min",
        verbose=True,
    )
    # Initialize loggers
    csv_logger = CSVLogger(save_dir=PATH_SUPERVISED_MODELS, name=model_name)

    # Define Trainger
    trainer = Trainer(
        max_epochs=total_epochs,
        callbacks=[callback_check, callback_tqdm, callback_early_stop],
        # accelerator="auto",
        # devices="auto",
        accelerator="gpu",  
        devices=[DEVICE.index],        # Specify GPU 1 (CUDA device 1)
        logger=[csv_logger]
    )

    # Start Training
    trainer.fit(model=lightning_model, datamodule=labeled_datamodule)

    # Get best model
    best_model_path = callback_check.best_model_path

    # Save Results
    res = trainer.validate(ckpt_path=best_model_path, datamodule=labeled_datamodule)  
    df_metrics = pd.read_csv(f"{csv_logger.log_dir}/metrics.csv")

    # Measure computing time
    t1 = time()
    computing_time_minutes = (t1-t0)/60
    # Return results
    return res, df_metrics, best_model_path, computing_time_minutes

def train_simclr(batch_size, train_loader, val_loader, max_epochs=500, patience=20, 
                 accelerator = 'cuda',devices = [1], num_nodes = 1,
                 log_path = 'simclr_logs', model_name = 'resnet', **kwargs):
    """
    Train a SimCLR model with early stopping.
    
    Args:
        batch_size (int): Batch size for training.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        max_epochs (int): Maximum number of epochs for training.
        patience (int): Number of epochs to wait for improvement before stopping.
        **kwargs: Additional arguments for the SimCLR model.
        
    Returns:
        SimCLR model: The trained model loaded from the best checkpoint.
    """
    # Initialize loggers
    csv_logger = CSVLogger(os.path.join(PATH_SUPERVISED_MODELS, log_path), 
                           name="train_simclr_resnet")
    
    # Define callbacks
    callbacks = [
        ModelCheckpoint(save_weights_only=True, mode='max', monitor='val_acc_top1'),
        LearningRateMonitor('epoch'),
        EarlyStopping(monitor='val_acc_top1', patience=patience, mode='max')
    ]
    gc.collect()
    # Initialize trainer
    trainer = pl.Trainer(
        default_root_dir=os.path.join(PATH_SUPERVISED_MODELS, 'SimCLR'),
        accelerator=accelerator,
        devices=devices,
        #devices=[1],
        #strategy="fsdp", # https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/
        num_nodes=num_nodes,
        max_epochs=max_epochs,
        enable_progress_bar=True,
        logger=[csv_logger],
        enable_model_summary=True,  # Reduce memory usage by disabling summary
        callbacks=callbacks,
        #log_every_n_steps=1
    )
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Set seed for reproducibility
    pl.seed_everything(0)
    
    # Initialize model
    if model_name=='resnet':
        model = SimCLR(max_epochs=max_epochs, **kwargs)
    if model_name=='vit_base':
        model = SimCLR_ViT(max_epochs=max_epochs, **kwargs)
    if model_name=='vit_large_p16':
        model = SimCLR_ViT(model_name=model_name, max_epochs=max_epochs, **kwargs)
    if model_name=='vit_large_p14':
        model = SimCLR_ViT(model_name=model_name, max_epochs=max_epochs, **kwargs)
    gc.collect()
    # Train model
    trainer.fit(model, train_loader, val_loader)
    gc.collect()
    # Load the best checkpoint
    if model_name=='resnet':
        model = SimCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    if 'vit' in model_name:
        model = SimCLR_ViT.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    
    return model








def train_supervised_model_v2(lightning_model, 
                           labeled_datamodule,
                           total_epochs = 200, 
                           model_name = "model_name"):
    t0 = time()
    # Define Callbacks
    callback_check = ModelCheckpoint(
        dirpath=PATH_SUPERVISED_MODELS, 
        save_top_k=1,
        mode="min",
        monitor="val_loss",
        filename="best-checkpoint",
        save_weights_only=True,
        verbose=True
    )
    callback_tqdm = RichProgressBar(leave=True)
    callback_early_stop = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        mode="min",
        verbose=True,
    )
    # Initialize loggers
    csv_logger = CSVLogger(save_dir=PATH_SUPERVISED_MODELS, name=model_name)

    # Define Trainger
    trainer = Trainer(
        max_epochs=total_epochs,
        callbacks=[callback_check, callback_tqdm, callback_early_stop],
        # accelerator="auto",
        # devices="auto",
        accelerator="gpu",  
        devices=[1],        # Specify GPU 1 (CUDA device 1)
        logger=[csv_logger]
    )

    # Start Training
    trainer.fit(model=lightning_model, datamodule=labeled_datamodule)

    # Get best model
    best_model_path = callback_check.best_model_path

    # Save Results
    res = trainer.validate(ckpt_path=best_model_path, datamodule=labeled_datamodule)  
    df_metrics = pd.read_csv(f"{csv_logger.log_dir}/metrics.csv")

    # Measure computing time
    t1 = time()
    computing_time_minutes = (t1-t0)/60
    # Return results
    return res, trainer


def classifier_ssl_trainer(
        model, 
        train_loader, 
        val_loader, 
        patience=50, 
        max_epochs = 500,
        dir_outputs = "outputs/ssl_classifier/",
        DEVICE = DEVICE
        ):
    early_stop_cb = EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=patience,
        verbose=True
    )
    csv_logger = CSVLogger(
        save_dir=dir_outputs,
        name="",    # subfolder
        version=""
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename=os.path.join(dir_outputs, 'best-val-acc'),
        enable_version_counter=False
    )
    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stop_cb],
        logger=csv_logger,
        accelerator='gpu' if DEVICE.type == 'cuda' else 'cpu', 
        devices=[DEVICE.index] if DEVICE.type == 'cuda' else 1
    )


    trainer.fit(model, train_loader, val_loader)

    log_dir = trainer.logger.log_dir
    try:
        metrics_df = pd.read_csv(
            os.path.join(log_dir, "metrics.csv"),
            engine="python",
            on_bad_lines="skip",       
            skip_blank_lines=True
        )

        train = (
            metrics_df[metrics_df["train_acc"].notna()]
            .groupby("epoch", as_index=True)
            .agg(train_acc = ("train_acc", "last"),
                train_loss= ("train_loss", "last"))
        )

        val = (
            metrics_df[metrics_df["val_acc"].notna()]
            .groupby("epoch", as_index=True)
            .agg(val_acc  = ("val_acc", "last"),
                val_loss = ("val_loss", "last"))
        )
        df_epoch = train.join(val)
        df_epoch.reset_index(inplace=True)
    except:
        df_epoch = None

    best_val_acc = checkpoint_callback.best_model_score 
    return best_val_acc, df_epoch
import io

def hpo_mlp(args):
    optuna_dir = args[0]
    df_balanced = args[1]
    ssl_model_name = args[2]
    subset_var = args[3]
    DEVICE = args[4]
    subset_train_test = subset_var.replace('sample_prop','split_prop')

    #torch.cuda.set_device(DEVICE.index)

    optuna_dir_full = os.path.join(optuna_dir,ssl_model_name,subset_var)

    try:
        os.makedirs(optuna_dir_full, exist_ok=True)
    except:
        pass

    class CPU_Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                # any nested byte‑storage load goes through torch.load on CPU
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
            return super().find_class(module, name)

    # in your hpo_mlp:
    with open(f"outputs/{ssl_model_name}", "rb") as f:
        ssl_model = CPU_Unpickler(f).load()  
    ssl_model = ssl_model.to(DEVICE)
    
    if 'simclr' in ssl_model_name:
        ssl_model = ssl_model.vit

    ssl_model.eval()

    encoder_dim = 1024
    if 'base' in ssl_model_name:
        encoder_dim = 768
    try: 
        os.remove(os.path.join(optuna_dir_full,'my_study.db'))
    except:
        pass
    
    storage_path = f"sqlite:///{optuna_dir_full}/my_study.db"


    def make_no_improvement_callback(k: int):
        best_value = None
        best_trial_no = 0

        def callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
            nonlocal best_value, best_trial_no

            # we assume higher is better; if your objective is minimize, invert the comparison
            if best_value is None or (trial.value is not None and trial.value > best_value):
                best_value = trial.value
                best_trial_no = trial.number
            elif trial.number - best_trial_no >= k:
                # stop the study once we've had k consecutive non-improving trials
                study.stop()

        return callback

    def objective(trial: optuna.trial.Trial):
        batch_size    = trial.suggest_categorical("batch_size", [16, 32])
        lr    =         trial.suggest_categorical("lr", [5e-6,1e-6, 5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3, 5e-2, 1e-2, 1e-1])
        weight_decay =  trial.suggest_categorical("weight_decay", [0, 5e-6, 1e-6, 5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3, 5e-2, 1e-2, 5e-1, 1e-1])
        dropout =       trial.suggest_categorical("dropout", [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
        print("-"*50)
        print(f"Learning Rate: {lr}")
        print(f"weight_decay: {weight_decay}")
        print(f"dropout: {dropout}")
        print(f"batch_size: {batch_size}")
        seed_everything(0)

        df_tmp = df_balanced.loc[df_balanced[subset_train_test]=='train'].copy()
        train_dataset_labeled = LabeledImageDataset(
            image_list = df_tmp['cropped_image'].to_list(),
            labels=df_tmp['label'].to_list(),
            transform=val_transform_labeled_simclr
            )
        df_tmp = df_balanced.loc[df_balanced[subset_train_test]=='valid'].copy()
        valid_dataset_labeled = LabeledImageDataset(
            image_list = df_tmp['cropped_image'].to_list(),
            labels=df_tmp['label'].to_list(),
            transform=val_transform_labeled_simclr
            )
        df_tmp = df_balanced.loc[df_balanced[subset_train_test]=='valid'].copy()
        test_dataset_labeled = LabeledImageDataset(
            image_list = df_tmp['cropped_image'].to_list(),
            labels=df_tmp['label'].to_list(),
            transform=val_transform_labeled_simclr
            )

        ###
        labeled_datamodule = LabeledDataModule(train_dataset_labeled, 
                                            valid_dataset_labeled, 
                                            test_dataset_labeled, 
                                            batch_size=batch_size,
                                            num_workers=0)
        labeled_datamodule_train = labeled_datamodule.train_dataloader()
        labeled_datamodule_val = labeled_datamodule.val_dataloader()
        
        if 'dino' in ssl_model_name: 
            train_dataset_labeled_latent = latent_dino_torch_dataset(
                model = ssl_model, 
                datamodule = labeled_datamodule_train, 
                device = DEVICE
                )
            valid_dataset_labeled_latent = latent_dino_torch_dataset(
                model = ssl_model,
                datamodule = labeled_datamodule_val,
                device = DEVICE
                )
        if 'simclr' in ssl_model_name: 
            train_dataset_labeled_latent = latent_simclr_vit_torch_dataset(
                simclr_vit_model = ssl_model, 
                datamodule = labeled_datamodule_train, 
                device = DEVICE
                )
            valid_dataset_labeled_latent = latent_simclr_vit_torch_dataset(
                simclr_vit_model = ssl_model,
                datamodule = labeled_datamodule_val,
                device = DEVICE
                )
        labeled_datamodule_latent = LabeledDataModule(train_dataset_labeled_latent, 
                                            valid_dataset_labeled_latent, 
                                            valid_dataset_labeled_latent, 
                                            batch_size=batch_size,
                                            num_workers=0)

        train_loader = labeled_datamodule_latent.train_dataloader()
        val_loader = labeled_datamodule_latent.val_dataloader()
                
        model = MLPEvaluation(
            feature_dim=encoder_dim, 
            num_classes=2, 
            hidden_dim = encoder_dim,
            lr=lr,
            weight_decay=weight_decay,
            dropout =dropout
        )
        model = model.to(DEVICE)
        seed_everything(0)
        res0 = classifier_ssl_trainer(
            model, 
            train_loader, 
            val_loader,
            max_epochs=500,
            patience=50,
            dir_outputs= os.path.join("outputs/ssl_classifier/",ssl_model_name,subset_var))
        acc = res0[0].item()        
        print(acc)
        torch.cuda.empty_cache()
        return acc
    
    seed_everything(0)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        load_if_exists=False,
        storage=storage_path
    )
    patience_hpo = 20
    stop_callback = make_no_improvement_callback(patience_hpo)
    study.optimize(
        objective, 
        n_trials=200, 
        timeout=3600*3,
        callbacks=[stop_callback]
        )

    return study.best_trial



def hpo_logistic(args):
    df_balanced = args[0]
    ssl_model_name = args[1]
    batch_size = 32

    with open(f"outputs/{ssl_model_name}", "rb") as f:
        ssl_model = pickle.load(f)
    if 'simclr' in ssl_model_name:
        ssl_model = ssl_model.vit
    ssl_model.eval()

    encoder_dim = 1024
    if 'base' in ssl_model_name:
        encoder_dim = 768

    seed_everything(0)

    df_tmp = df_balanced.loc[df_balanced['subset']=='train'].copy()
    train_dataset_labeled = LabeledImageDataset(
        image_list = df_tmp.loc[df_tmp['subset']=='train','cropped_image'].to_list(),
        labels=df_tmp.loc[df_tmp['subset']=='train','label'].to_list(),
        transform=val_transform_labeled_simclr
        )
    df_tmp = df_balanced.loc[df_balanced['subset']=='valid'].copy()
    valid_dataset_labeled = LabeledImageDataset(
        image_list = df_tmp.loc[df_tmp['subset']=='valid','cropped_image'].to_list(),
        labels=df_tmp.loc[df_tmp['subset']=='valid','label'].to_list(),
        transform=val_transform_labeled_simclr
        )
    df_tmp = df_balanced.loc[df_balanced['subset']=='valid'].copy()
    test_dataset_labeled = LabeledImageDataset(
        image_list = df_tmp.loc[df_tmp['subset']=='valid','cropped_image'].to_list(),
        labels=df_tmp.loc[df_tmp['subset']=='valid','label'].to_list(),
        transform=val_transform_labeled_simclr
        )

    ###
    labeled_datamodule = LabeledDataModule(train_dataset_labeled, 
                                        valid_dataset_labeled, 
                                        test_dataset_labeled, 
                                        batch_size=batch_size,
                                        num_workers=0)
    labeled_datamodule_train = labeled_datamodule.train_dataloader()
    labeled_datamodule_val = labeled_datamodule.val_dataloader()
    
    if 'dino' in ssl_model_name: 
        train_dataset_labeled_latent = latent_dino_torch_dataset(
            model = ssl_model, 
            datamodule = labeled_datamodule_train, 
            device = DEVICE
            )
        valid_dataset_labeled_latent = latent_dino_torch_dataset(
            model = ssl_model,
            datamodule = labeled_datamodule_val,
            device = DEVICE
            )
    if 'simclr' in ssl_model_name: 
        train_dataset_labeled_latent = latent_simclr_vit_torch_dataset(
            simclr_vit_model = ssl_model, 
            datamodule = labeled_datamodule_train, 
            device = DEVICE
            )
        valid_dataset_labeled_latent = latent_simclr_vit_torch_dataset(
            simclr_vit_model = ssl_model,
            datamodule = labeled_datamodule_val,
            device = DEVICE
            )
    labeled_datamodule_latent = LabeledDataModule(train_dataset_labeled_latent, 
                                        valid_dataset_labeled_latent, 
                                        valid_dataset_labeled_latent, 
                                        batch_size=batch_size,
                                        num_workers=0)

    train_loader = labeled_datamodule_latent.train_dataloader()
    val_loader = labeled_datamodule_latent.val_dataloader()

    clf = LogisticRegression(
        max_iter=100_000, 
        random_state=0, 
        tol = 1e-5,
        )
    clf.fit(train_dataset_labeled_latent.features, train_dataset_labeled_latent.labels)

    y_pred = clf.predict(valid_dataset_labeled_latent.features)
    return accuracy_score(valid_dataset_labeled_latent.labels, y_pred)

            
