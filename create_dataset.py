import pandas as pd
import numpy as np
from pathlib import Path
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from torchtext.vocab import vocab


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MASK_PERCENTAGE = 0.15


# data, vocabulary, max sequence length, mask probability, include sequences, some random state
class NCBIDataset(Dataset):

    MASKED_INDICES_COLUMN = 'masked_indices'
    TARGET_COLUMN = 'indices'
    TOKEN_MASK_COLUMN = 'token_mask'
    AB_INDEX = 'ab_index'
    SR_CLASS = 'sr_class'

    def __init__(self,
                 data: pd.DataFrame,
                 vocab_geno: vocab,
                 vocab_pheno: vocab,
                 max_seq_len: list,
                 mask_prob: float,
                 include_pheno:bool,
                 random_state: int = 23,
                 ):
        
        self.random_state = random_state
        np.random.seed(self.random_state)

        CLS = '[CLS]'
        PAD = '[PAD]'
        MASK = '[MASK]'
        UNK = '[UNK]'

        self.include_pheno = include_pheno
        self.data = data.reset_index(drop=True) 
        self.num_samples = self.data.shape[0]
        self.vocab_geno = vocab_geno
        self.vocab_pheno = vocab_pheno
        self.vocab_size_geno = len(self.vocab_geno)
        self.CLS = CLS 
        self.PAD = PAD
        self.MASK = MASK
        self.UNK = UNK
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
        if self.include_pheno:
            self.columns = [self.MASKED_INDICES_COLUMN, self.TARGET_COLUMN, self.AB_INDEX, self.SR_CLASS]
        else:
            self.columns = [self.MASKED_INDICES_COLUMN, self.TARGET_COLUMN]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        input = torch.Tensor(item[self.MASKED_INDICES_COLUMN],device=device).long()
        token_mask  = torch.tensor(item[self.TARGET_COLUMN], device=device).long()
        attention_mask = (input == self.vocab_geno[self.PAD]).unsqueeze(0)
        
        if self.include_pheno:
            ab_idx  = torch.tensor(item[self.AB_INDEX], device=device).long()
            sr_class = torch.tensor(item[self.SR_CLASS], device=device).long()
            return input, token_mask , attention_mask, ab_idx, sr_class
        else:
            return input, token_mask , attention_mask

    def _construct_masking(self):
        sequences = deepcopy(self.data['genes'].tolist())
        masked_sequences = []
        target_indices_list = []
        seq_starts = [[self.CLS, self.data['year'].iloc[i], self.data['location'].iloc[i]] for i in range(self.data.shape[0])]

        for i, geno_seq in enumerate(sequences):
            seq_len = len(geno_seq)
            masking_index = np.random.rand(seq_len) < self.mask_prob   
            target_indices = np.array([-1]*seq_len)
            indices = masking_index.nonzero()[0]
            target_indices[indices] = self.vocab_geno.lookup_indices([geno_seq[i] for i in indices])
            for i in indices:
                r = np.random.rand()
                if r < 0.8:
                    geno_seq[i] = self.MASK
                elif r > 0.9:
                    geno_seq[i] = self.vocab_geno.lookup_token(np.random.randint(self.vocab_size_geno))
            geno_seq = seq_starts[i] + geno_seq
            target_indices = [-1]*3 + target_indices.tolist() 
            masked_sequences.append(geno_seq)
            target_indices_list.append(target_indices)
        masked_sequences = [seq + [self.PAD]*(self.max_seq_len[0] - len(seq)) for seq in masked_sequences]
        for i in range(len(target_indices_list)):
            indices = target_indices_list[i]
            padding = [-1] * (self.max_seq_len[0] - len(indices))
            target_indices_list[i] = indices + padding
        return masked_sequences, target_indices_list 
    
    def _Ab_SR_indexing(self):
        sequences = deepcopy(self.data['AST_phenotypes'].tolist())
        list_idx = []
        list_SR = []
        for i in range(len(sequences)):
            current_seq = sequences[i]
            current_idxs = []
            current_SRs = []
            for j in range(len(current_seq)):
                item = current_seq[j].split('=')
                abs = item[0]   
                sr = item[1]
                current_idxs.append(self.vocab_pheno.lookup_indices([abs]))
                for k in range(len(sr)):
                    if sr == 'R':
                        current_SRs.append(1)
                    else:
                        current_SRs.append(0)
            current_idxs = [int(item[0]) for item in current_idxs]
            for i in range(0,max_length[1] - len(current_idxs)):
                current_idxs.append(-1)
            for i in range(0,max_length[1] - len(current_SRs)):
                current_SRs.append(-1)
            list_idx.append(current_idxs)
            list_SR.append(current_SRs)
        return list_idx, list_SR
    
    def prepare_dataset(self):
        masked_sequences, target_indices = self._construct_masking()
        indices_masked = [self.vocab_geno.lookup_indices(masked_seq) for masked_seq in masked_sequences]
        if self.include_pheno:
            list_idx, list_SR = self._Ab_SR_indexing()

        if self.include_pheno:
            rows = zip(indices_masked, target_indices, list_idx, list_SR)
            self.df = pd.DataFrame(rows, columns=self.columns)
        else:
            rows = zip(indices_masked, target_indices)
            self.df = pd.DataFrame(rows, columns=self.columns)