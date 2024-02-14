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
    NSP_TARGET_COLUMN = 'is_next'
    TOKEN_MASK_COLUMN = 'token_mask'

    def __init__(self,
                 data: pd.DataFrame,
                 vocab: vocab,
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

        self.data = data.reset_index(drop=True) 
        self.num_samples = self.data.shape[0]
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.CLS = CLS 
        self.PAD = PAD
        self.MASK = MASK
        self.UNK = UNK
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
        self.columns = [self.MASKED_INDICES_COLUMN, self.TARGET_COLUMN]
        self.include_pheno = include_pheno

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        input = torch.Tensor(item[self.MASKED_INDICES_COLUMN],device=device).long()
        token_mask  = torch.tensor(item[self.TARGET_COLUMN], device=device).long()
        attention_mask = (input == self.vocab[self.PAD]).unsqueeze(0)

        return input, token_mask , attention_mask

    def _construct_masking_geno(self):
        sequences = deepcopy(self.data['genes'].tolist())
        masked_sequences = list()
        target_indices_list = list()
        seq_starts = [[self.CLS, self.data['year'].iloc[i], self.data['location'].iloc[i]] for i in range(self.data.shape[0])]

        for i, geno_seq in enumerate(sequences):
            seq_len = len(geno_seq)
            masking_index = np.random.rand(seq_len) < self.mask_prob   
            target_indices = np.array([-1]*seq_len)
            indices = masking_index.nonzero()[0]
            target_indices[indices] = self.vocab.lookup_indices([geno_seq[i] for i in indices])
            for i in indices:
                r = np.random.rand()
                if r < 0.8:
                    geno_seq[i] = self.MASK
                elif r > 0.9:
                    geno_seq[i] = self.vocab.lookup_token(np.random.randint(self.vocab_size))
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
    
    
    def _construct_masking_pheno(self):
        sequences = [(gene, pheno) for gene, pheno in zip(self.data['genes'].tolist(), self.data['AST_phenotypes'].tolist())]

        # Deepcopy the list of tuples
        sequences_deepcopy = deepcopy(sequences)
        masked_sequences = list()
        target_indices_list = list()
        seq_starts = [[self.CLS, self.data['year'].iloc[i], self.data['location'].iloc[i]] for i in range(self.data.shape[0])]

        for i, info_seq in enumerate(sequences_deepcopy):
            geno_len = len(info_seq[0])
            pheno_len = len(info_seq[1])

            masking_index_geno = np.random.rand(geno_len) < self.mask_prob
            masking_index_pheno = np.random.rand(pheno_len) < self.mask_prob

            target_indices_geno = np.array([-1]*geno_len)
            target_indices_pheno = np.array([-1]*pheno_len)

            indices_geno = masking_index_geno.nonzero()[0]
            indices_pheno = masking_index_pheno.nonzero()[0]

            target_indices_geno[indices_geno] = self.vocab.lookup_indices([info_seq[0][i] for i in indices_geno])
            target_indices_pheno[indices_pheno] = self.vocab.lookup_indices([info_seq[1][i] for i in indices_pheno])

            for i in indices_geno:
                r = np.random.rand()
                if r < 0.8:
                    info_seq[0][i] = self.MASK
                elif r > 0.9:
                    info_seq[0][i] = self.vocab.lookup_token(np.random.randint(self.vocab_size))
            
            for i in indices_pheno:
                r = np.random.rand()
                if r < 0.8:
                    info_seq[1][i] = self.MASK
                elif r > 0.9:
                    info_seq[1][i] = self.vocab.lookup_token(np.random.randint(self.vocab_size))
            geno_seq_pad = info_seq[0]+ [self.PAD]*(self.max_seq_len[1] - geno_len)
            pheno_seq_pad = info_seq[1]+ [self.PAD]*(self.max_seq_len[2] - pheno_len)
            seq = seq_starts[i] + geno_seq_pad +pheno_seq_pad

            target_indices_geno =  target_indices_geno.tolist()
            padding = [-1] * (self.max_seq_len[1] - geno_len)
            target_indices_geno = target_indices_geno + padding

            target_indices_pheno =  target_indices_pheno.tolist()
            padding = [-1] * (self.max_seq_len[2] - pheno_len)
            target_indices_pheno = target_indices_pheno + padding

            target_indices = [-1]*3 + target_indices_geno + target_indices_pheno
            
            masked_sequences.append(seq)
            target_indices_list.append(target_indices)
        
        return masked_sequences, target_indices_list 
        
    def prepare_dataset(self):

        masked_sequences, target_indices = self._construct_masking_geno()
        indices_masked = [self.vocab.lookup_indices(masked_seq) for masked_seq in masked_sequences]

        rows = zip(indices_masked, target_indices)
        self.df = pd.DataFrame(rows, columns=self.columns)