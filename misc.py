import numpy as np
import pickle
from pathlib import Path
import torch
import os
from bert_builder import BERT
def get_split_indices(size_to_split, val_share, random_state: int = 42):
    indices = np.arange(size_to_split)
    np.random.seed(random_state)
    np.random.shuffle(indices)
    
    train_share = 1 - val_share
    
    train_size = int(train_share * size_to_split)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    return train_indices, val_indices

def export_results(results, savepath):
    with open(savepath, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {savepath}")

def get_paths():
    cwd = os.getcwd()
    print(cwd)
    if  cwd == "c:\\Users\\erikw\\Desktop\\ExjobbKod\\base":
        data_dir = 'c:\\Users\\erikw\\Desktop\\ExjobbKod\\data'
        ab_dir = 'c:\\Users\\erikw\\Desktop\\ExjobbKod\\base'
        save_directory = 'c:\\Users\\erikw\\Desktop\\ExjobbKod\\results'
    elif cwd == "c:\\Users\\erika\\Desktop\\Exjobb\\repo\\base":
        data_dir = 'c:\\Users\\erika\\Desktop\\Exjobb\\data'
        ab_dir = 'c:\\Users\\erika\\Desktop\\Exjobb\\repo\\base'
        save_directory = 'c:\\Users\\erika\\Desktop\\Exjobb\\savefiles'
    elif cwd == '/cephyr/users/aeerik/Alvis/base':
        data_dir = '/cephyr/users/aeerik/Alvis/data/raw'
        ab_dir = '/cephyr/users/aeerik/Alvis/base'
        save_directory = '/cephyr/users/aeerik/Alvis/runs'     
    return data_dir, ab_dir, save_directory

def model_loader(savepath: Path,vocabulary_geno, vocabulary_pheno, dim_emb, dim_hidden, num_encoders, drop_prob, cls_mode, device):
    print(f"Loading model from {savepath}")
    model = BERT(vocab_size=len(vocabulary_geno), dim_embedding = dim_emb, dim_hidden=dim_hidden, attention_heads=8, num_encoders=num_encoders, dropout_prob=drop_prob, num_ab=len(vocabulary_pheno),cls_mode=cls_mode, device=device)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(savepath))
    else:
        model.load_state_dict(torch.load(savepath, map_location=torch.device('cpu')))
    
    return model