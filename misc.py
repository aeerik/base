import numpy as np
import pickle
import os
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
    return data_dir, ab_dir, save_directory