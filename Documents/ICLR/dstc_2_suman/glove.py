#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 21:25:27 2017

@author: suman
"""

import pandas as pd
import csv
import json
import numpy as np

def get_vocab(vocab_fname):
    return json.load(open(vocab_fname,'r'))


def vec(w,words):
    if w in words.index:
        return words.loc[w].as_matrix()
    else:
        return 0

def get_vocab_init_matrix(vocab_fname,glove_fname):
    np.random.seed(1234)
    rand=np.random.rand(504,100)
    glove_data_file=open(glove_fname,'r')
    words = pd.read_table(glove_data_file, sep=" ",index_col=0,header=None,quoting=csv.QUOTE_NONE)

    v=get_vocab(vocab_fname)
    embedding_dict=words.T.to_dict('list')
    d={}
    
    for key,value in v.items():
        if key in embedding_dict.keys():
            d[key]=np.asarray(embedding_dict[key])
        
    no_emb=[]
    for i in v:
        if not hasattr(vec(i,words),"__len__"):
            no_emb.append(i)

    for ind,item in enumerate(no_emb):
        d[item]=rand[ind]
        
    vocab={k: v for v, k in enumerate(d.keys())}
    init_matrix=np.asarray(list(d.values()))
    return vocab,init_matrix

vocab_file='data/p-dialog-babi-task6-dstc2-vocab.json'
glove_file='data/glove.6B.100d.txt'
vocab,init_matrix=get_vocab_init_matrix(vocab_file,glove_file)


