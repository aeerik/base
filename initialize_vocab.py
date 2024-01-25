import random
import typing
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from itertools import chain
import torch

from tqdm import tqdm
from torch.utils.data import Dataset
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer 

def make_vocabulary(dataset: pd.DataFrame):
    CLS = '[CLS]'
    PAD = '[PAD]'
    MASK = '[MASK]'
    UNK = '[UNK]'

    token_list = Counter()
    data = dataset.copy()

    location_tokens = list(dict.fromkeys(list(chain(list(data['location'])))))
    year_tokens = list(dict.fromkeys(list(chain(list(data['year'])))))
    genes_tokens = list(dict.fromkeys(list(chain(*data['genes']))))
   
    print(location_tokens)
    token_list.update(map(str, year_tokens))
    token_list.update(map(str, location_tokens))
    token_list.update(map(str, genes_tokens))
    
    print(token_list)
    vocabulary = vocab(token_list,specials = [CLS, PAD, MASK, UNK])

    return vocabulary