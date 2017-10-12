#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 17:51:37 2017

@author: suman
"""

import tensorflow as tf
import numpy as np
import json
<<<<<<< HEAD
import pickle
from model import VariableEmbedder, PositionEncoder, Model


flags = tf.app.flags
flags.DEFINE_string("data_dir", "data/dialog-babi", "Data directory [data/babi]")
flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
flags.DEFINE_integer("hidden_size", 100, "hidden dimensions of the word embeddings.")
FLAGS = flags.FLAGS


=======
#from model import *
>>>>>>> b9142bdc5bb8c7718091544202fa6c1f1d987ffd

def pad(data,stats):
    fill=list(np.zeros(stats[3],dtype=int))
    fillkb=list(np.zeros(3,dtype=int))

    for ind,d in enumerate(data):
            for context in d:
<<<<<<< HEAD
                for i in range(stats[ind]-len(context)):
                    if ind==1:
                        context.append(fillkb)
                    elif ind==3 or ind==4 or ind==5:
                        context.append(0)
                    elif ind==0 or ind==2:
                        context.append(fill)
=======
                if ind==4:
                    ind=3
                for i in range(stats[ind]-len(context)):
                    if ind==1:
                        context.append(fillkb)
                    elif ind==3 or ind==4:
                        context.append(0)
                    elif ind==0:
                        context.append(fill)
        
  
>>>>>>> b9142bdc5bb8c7718091544202fa6c1f1d987ffd
    for i in data[0]:
        for j in i:
            if len(j)!=stats[3]:
                for i in range(stats[3]-len(j)):
                    j.append(0)
    
    for i in data[2]:
        for j in i:
            if len(j)!=stats[3]:
                for i in range(stats[3]-len(j)):
                    j.append(0)

<<<<<<< HEAD
def lower(word):

    if len(word)>1:
        if word[0].isupper() and '_' not in word and '.' not in word:
            return word[0].lower()+word[1:]
        else:
            return word
    else:
        return word


def replace_token_no(data,vocab):
    
    for ind,d in enumerate(data):
        if ind!=3 and ind!=4 and ind!=5:
            for i1,context in enumerate(d):
                for i2,i in enumerate(context):
                    d[i1][i2]=[w if w==0 else vocab[lower(w)] for w in i]
        else:
            for i1,context in enumerate(d):
                data[ind][i1]=[w if w==0 else vocab[lower(w)] for w in context]
            


dir_=FLAGS.data_dir

=======
flags = tf.app.flags
flags.DEFINE_string("data_dir", "data/dialog-babi", "Data directory [data/babi]")
FLAGS = flags.FLAGS


dir_=FLAGS.data_dir
>>>>>>> b9142bdc5bb8c7718091544202fa6c1f1d987ffd
with open(dir_+'/p-dialog-babi-task6-dstc2-train.json','r') as fp:
    train_data=json.load(fp)
with open(dir_+'/p-dialog-babi-task6-dstc2-test.json','r') as fp:
    test_data=json.load(fp)
with open(dir_+'/p-dialog-babi-task6-dstc2-dev.json','r') as fp:
    dev_data=json.load(fp)
with open(dir_+'/p-dialog-babi-task6-dstc2-stats.json','r') as fp:
    stats=json.load(fp)
<<<<<<< HEAD
with open(dir_+'/p-dialog-babi-task6-dstc2-vocab.json','r') as fp:
    vocab=json.load(fp)
with open(dir_+'/p-dialog-babi-task6-dstc2-init.pkl','rb') as f:
    matrix = pickle.load(f)

params_dict=FLAGS.__flags

params_dict['max_pre_size']=stats[0]
params_dict['max_kb_size']=stats[1]
params_dict['max_post_size']=stats[2]
params_dict['max_sent_size']=stats[3]
params_dict['vocab_size']=len(vocab)
params_dict['emb_matrix']=matrix


pad(train_data,stats)
replace_token_no(train_data,vocab)
pre=np.asarray(train_data[0])
kb=np.asarray(train_data[1])
post=np.asarray(train_data[2])
query=np.asarray(train_data[3])
response=np.asarray(train_data[4])


#batch_dict={'pre':pre[0:32],'kb':kb[0:32],'post':post[0:32],'query':query[0:32],'response':response[0:32]}
#
#with tf.Session() as sess:
#    model = Model(sess,params_dict)
#    aa=model.batch_fit(batch_dict)
#    
=======

pad(train_data,stats)
>>>>>>> b9142bdc5bb8c7718091544202fa6c1f1d987ffd
