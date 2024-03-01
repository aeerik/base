
from collections import Counter
import numpy as np
import pandas as pd
from itertools import chain
from torchtext.vocab import vocab



def vocab_geno(dataset: pd.DataFrame):
    CLS = '[CLS]'
    PAD = '[PAD]'
    MASK = '[MASK]'
    UNK = '[UNK]'

    token_list = Counter()
    data = dataset.copy()

    location_tokens = list(dict.fromkeys(list(chain(list(data['location'])))))
    year_tokens = list(dict.fromkeys(list(chain(list(data['year'])))))
    genes_tokens = list(dict.fromkeys(list(chain(*data['genes']))))

   
    token_list.update(map(str, year_tokens))
    token_list.update(map(str, location_tokens))
    token_list.update(map(str, genes_tokens))

    vocabulary = vocab(token_list,specials = [CLS, PAD, MASK, UNK])
    return vocabulary

def vocab_pheno(dataset: pd.DataFrame):

    token_list = Counter()
    data = dataset.copy()

    pheno_tokens = list(dict.fromkeys(list(chain(list(data['antibiotic'])))))

    token_list.update(map(str, pheno_tokens))
    
    vocabulary = vocab(token_list)
    return vocabulary