import pandas as pd
import os
import numpy as np
from pathlib import Path
import math
import warnings
warnings.filterwarnings("ignore")

def data_loader(include_pheno, threshold_year,data_path,ab_path):
    data_dir =Path(os.path.abspath(data_path))
    os.chdir(data_dir)
    NCBI_raw = pd.read_csv('NCBI.tsv',sep='\t',header=0,low_memory=False)

    selected_data = ['collection_date', 'geo_loc_name', 'AMR_genotypes_core']
    selected_data += ['AST_phenotypes'] if include_pheno else []

    NCBI_raw = NCBI_raw[selected_data]

    NCBI_raw.rename(columns={'geo_loc_name': 'location'}, inplace=True)
    NCBI_raw.rename(columns={'collection_date': 'year'}, inplace=True)
    NCBI_raw.rename(columns={'AMR_genotypes_core': 'genes'}, inplace=True)

    NCBI = NCBI_raw[NCBI_raw['genes'].notnull()]

    unknown = ['unknown','missing','not determined', 'not collected', 'not provided', 'Not Provided', 'OUTPATIENT','missing: control sample', 'Not collected', 'Not Collected', 'not available', '-']
    
    #genomic filtering
    NCBI.loc[:,'genes'] = NCBI['genes'].replace(unknown, np.nan)

    labels = ['=PARTIAL', '=MISTRANSLATION', '=HMM', '=PARTIAL_END_OF_CONTIG']
    NCBI['genes'] = NCBI['genes'].str.split(',')
    NCBI['genes'] = NCBI['genes'].apply(lambda x: list(set([g.strip() for g in x])))
    NCBI['genes'] = NCBI['genes'].apply(lambda x: [g for g in x if not g.endswith(tuple(labels))]) 
    NCBI = NCBI[NCBI['genes'].apply(lambda x: len(x) > 0)] 

    #collection date
    NCBI.loc[:,'year'] = NCBI['year'].replace(unknown, np.nan)
    NCBI.loc[:,'year'] = NCBI['year'].str.split('-').str[0]
    NCBI.loc[:,'year'] = NCBI['year'].str.split('/').str[0]
    NCBI.loc[:,'year'] = NCBI['year'].str.split(':').str[0]

    year_idx = NCBI[NCBI['year'].astype(float) < threshold_year].index
    NCBI.drop(year_idx, inplace=True)   

    #location
    NCBI.loc[:,'location'] = NCBI['location'].replace(unknown, np.nan)
    NCBI.loc[:,'location'] = NCBI['location'].str.split(',').str[0]
    NCBI.loc[:,'location'] = NCBI['location'].str.split(':').str[0] 
    NCBI.loc[:,'location'] = NCBI['location'].replace(
        {'United Kingdom': 'UK', 'United Arab Emirates': 'UAE', 'Democratic Republic of the Congo': 'DRC',
         'Republic of the Congo': 'DRC', 'Czechia': 'Czech Republic', 'France and Algeria': 'France'})
    
    #phenotype 
    if include_pheno:
        labels = ['=ND', '=I', '=NS', "=DD"]
        NCBI = NCBI[NCBI['AST_phenotypes'].notnull()]
        NCBI['AST_phenotypes'] = NCBI['AST_phenotypes'].str.split(',')
        NCBI['AST_phenotypes'] = NCBI['AST_phenotypes'].apply(lambda x: list(set([g.strip() for g in x])) if isinstance(x, list) else [])
        NCBI['AST_phenotypes'] = NCBI['AST_phenotypes'].apply(lambda x: [g for g in x if not g.endswith(tuple(labels))] if isinstance(x, list) else [])
        NCBI = NCBI[NCBI['AST_phenotypes'].apply(lambda x: len(x) > 0)]

    NCBI.fillna("[PAD]", inplace=True)

    ab_dir =Path(os.path.abspath(ab_path))
    os.chdir(ab_dir)
    ab_list = open("antibiotic_list.txt","r")
    ab_list = ab_list.read().splitlines()
    ab_df = pd.DataFrame(ab_list, columns = ['antibiotic'])
    return NCBI, ab_df