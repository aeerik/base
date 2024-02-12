
from collections import Counter
import numpy as np
import pandas as pd
from itertools import chain
from torchtext.vocab import vocab



def make_vocabulary(dataset: pd.DataFrame, include_pheno):
    CLS = '[CLS]'
    PAD = '[PAD]'
    MASK = '[MASK]'
    UNK = '[UNK]'

    token_list = Counter()
    data = dataset.copy()

    location_tokens = list(dict.fromkeys(list(chain(list(data['location'])))))
    year_tokens = list(dict.fromkeys(list(chain(list(data['year'])))))
    genes_tokens = list(dict.fromkeys(list(chain(*data['genes']))))
    if include_pheno:
        pheno_tokens = list(dict.fromkeys(list(chain(*data['AST_phenotypes']))))

   
    token_list.update(map(str, year_tokens))
    token_list.update(map(str, location_tokens))
    token_list.update(map(str, genes_tokens))
    
    if include_pheno:
        token_list.update(map(str, pheno_tokens))
    vocabulary = vocab(token_list,specials = [CLS, PAD, MASK, UNK])
    return vocabulary