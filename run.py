#Relevant packages
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchtext
import torchtext.vocab as vocab
from pathlib import Path
import os
from datetime import date
today = date.today()

from misc import get_split_indices
from misc import export_results
from data_preprocessing import data_loader, data_original
from build_vocabulary import vocab_geno
from build_vocabulary import vocab_pheno
from create_dataset import NCBIDataset
from bert_builder import BERT
from trainer import BertTrainer_ft
from trainer import BertTrainer_pt
from misc import get_paths
from misc import model_loader

####################################################
#Data directories
base_dir = Path(os.path.abspath(''))
os.chdir(base_dir)
data_dir, ab_dir, save_directory = get_paths()

#Run settings
limit_data = False #Reduces number of used samples in taining
wandb_mode = True #Uses wandb for logging
mode_ft = True  #True for fine tuning, False for pretraining
load_model = False #True to load a model from a file
export_model = True

cls_mode=False

#Hyperparameters
threshold_year = 1970
max_length = [51,44]
mask_prob = 0.625
drop_prob = 0.2
reduced_samples = 1000 

dim_emb = 512
dim_hidden = 512
attention_heads = 4 

num_encoders = 5

epochs = 100
batch_size = 32
lr = 0.0000001
stop_patience = 7

# WandB settingsS
wandb_project = "CLS_FirstRun"
wandb_run_name = "5EncEmb512l"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
####################################################

#set mode for run, True for fine tuning, False for pretraining 
if mode_ft:
    print(f"Fine tuning mode")
    include_pheno = True
    # export_model = False
else:
    print(f"Pretraining mode")
    include_pheno = False
    # export_model = True

if device.type == "cuda":
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()
else:
    print("Using CPU")  
    
print(f"\n Retrieving data from: {data_dir}")
print("Loading data...")
NCBI,ab_df = data_loader(include_pheno,threshold_year,data_dir,ab_dir)
NCBI_geno_only = data_original(threshold_year,data_dir, ab_dir)
print(f"Data correctly loaded, {len(NCBI)} samples found")
print("Creating vocabulary...")
vocabulary_geno = vocab_geno(NCBI_geno_only)
vocabulary_pheno = vocab_pheno(ab_df)

print(f"Vocabulary created with number of elements:",len(vocabulary_geno))
if include_pheno:
    print(f"Number of antibiotics:",len(vocabulary_pheno))

if limit_data:
    print(f"Reducing samples to {reduced_samples}")
    NCBI = NCBI.head(reduced_samples)

train_indices, val_indices = get_split_indices(len(NCBI), 0.2)
train_set = NCBIDataset(NCBI.iloc[train_indices], vocabulary_geno, vocabulary_pheno, max_length, mask_prob,include_pheno)
val_set = NCBIDataset(NCBI.iloc[val_indices], vocabulary_geno, vocabulary_pheno, max_length, mask_prob,include_pheno)
print(f"Datasets has been created with {len(train_set)} samples in the training set and {len(val_set)} samples in the validation set")

print(f"Creating model...")
if load_model:
    savepath = ""
    model = model_loader(savepath, vocabulary_geno, vocabulary_pheno, dim_emb, dim_hidden, num_encoders, drop_prob, cls_mode, device).to(device)
    if mode_ft:
        model.finetune_unfreezeing()
    else:
        model.pretrain_freezing()
else:
    if mode_ft:
        model = BERT(vocab_size=len(vocabulary_geno), dim_embedding = dim_emb, dim_hidden=dim_hidden, attention_heads=8, num_encoders=num_encoders, dropout_prob=drop_prob, num_ab=len(vocabulary_pheno), cls_mode=cls_mode, device=device).to(device)
        model.finetune_unfreezeing()
    else:
        model = BERT(vocab_size=len(vocabulary_geno), dim_embedding = dim_emb, dim_hidden=dim_hidden, attention_heads=8, num_encoders=num_encoders, dropout_prob=drop_prob, num_ab=len(vocabulary_pheno),cls_mode=cls_mode, device=device).to(device)
        model.pretrain_freezing()
print(f"Model successfully loaded")
print(f"---------------------------------------------------------")
print(f"Starting training...")
if mode_ft:
    trainer = BertTrainer_ft(model, max_length, train_set, val_set, epochs, batch_size, lr, device, stop_patience, wandb_mode, wandb_project, wandb_run_name)
else: 
    trainer = BertTrainer_pt(model, max_length, train_set, val_set, epochs, batch_size, lr, device, stop_patience, wandb_mode, wandb_project, wandb_run_name)

results = trainer()
print(f"---------------------------------------------------------")
if export_model:
    print(f"Exporting model...")
    export_model_label = str(today)+"model"+"Enc"+str(num_encoders)+"Emb"+str(dim_emb)+"Mask"+str(mask_prob)+"Mode"+str(mode_ft)+".pt"
    trainer._save_model(save_directory+"/"+export_model_label)
print("Exporting results...")
export_results_label = str(today)+"run"+"Mode"+str(mode_ft)+".pkl"
export_results(results, save_directory+"/"+export_results_label)
print(f"---------------------------------------------------------")

print(f"F1 printing:")
print("Sensitivity:")
print(results['sensitivity'])
print("Specificity:")
print(results['specificity'])




