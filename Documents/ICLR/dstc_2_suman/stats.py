#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 00:08:20 2017

@author: suman
"""

import re
import os

home = os.path.expanduser("~")
source_dir = os.path.join(home, "data", "dialog_dstc_2")
source_fname = source_dir+ '/dialog-babi-task6-dstc2-'

train_input = open(source_fname+ 'trn.txt','r')
test_input = open(source_fname+ 'tst.txt','r')
dev_input = open(source_fname+ 'dev.txt','r')


s = ''
for i in train_input.readlines():
    s = s + i

all_dialogues=s.split('\n\n')
del all_dialogues[-1]

utterances=set([])
for single_dialogue in all_dialogues:
    content = single_dialogue.split('\n')
    for timestep in content:
        utterance_response=timestep.split('\t')
        if len(utterance_response)==1:
            continue
        utterance_response[0]=utterance_response[0].split(' ',1)[1]
        utterance_response[0]=utterance_response[0].strip()
        utterance_response[1]=utterance_response[1].strip()
        utterances.add(utterance_response[0])
        utterances.add(utterance_response[1])

utt=sorted(utterances)