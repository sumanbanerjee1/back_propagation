import pickle
import re
import os
import argparse
import json
<<<<<<< HEAD
import pandas as pd
=======
#import pandas as pd
>>>>>>> b9142bdc5bb8c7718091544202fa6c1f1d987ffd
import csv
import numpy as np


def get_args():
<<<<<<< HEAD
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    source_dir = os.path.join(home, "data", "dialog_dstc_2") #when pushing to git we need to add a download script here
    parser.add_argument("--source_dir", default=source_dir)
    parser.add_argument("--target_dir", default=source_dir)
    args = parser.parse_args()
    return args
=======
	parser = argparse.ArgumentParser()
	home = os.path.expanduser("~")
	source_dir = os.path.join(home, "data", "dialog_dstc_2") #when pushing to git we need to add a download script here
	parser.add_argument("--source_dir", default=source_dir)
	parser.add_argument("--target_dir", default=source_dir)
	args = parser.parse_args()
	return args
>>>>>>> b9142bdc5bb8c7718091544202fa6c1f1d987ffd

eos = '<eos>'
beg = ['<beg>']
eok = '<eok>'
def count_kb(content):
	total_api_calls = len(re.findall('\tapi_call',content))
	failed_api_calls = len(re.findall('api_call no result',content))
	return total_api_calls - failed_api_calls

def locate_kb(content):
<<<<<<< HEAD
    flag=0
    kb_start_found = False
    start_index = -1
    end_index = -1
    for turn_no, current_turn in enumerate(content):
        if "R_post_code" in current_turn and not kb_start_found:
            kb_start_found = True
            start_index = turn_no
        if kb_start_found:
            if "<SILENCE>" in current_turn:
                if turn_no+2<len(content) and turn_no+3<len(content):
                    if 'R_post_code' in content[turn_no+2] or 'R_post_code' in content[turn_no+3]:
                        flag=1
                        continue
                end_index = turn_no

                break
    return start_index,end_index,flag
=======
	
	kb_start_found = False
	start_index = []
	end_index = []
	#kb_counter = count_kb
	
	for turn_no, current_turn in enumerate(content):
		if "R_post_code" in current_turn and not kb_start_found:
			kb_start_found = True
			start_index.append(turn_no)
		if kb_start_found:
			if "<SILENCE>" in current_turn:
				end_index.append(turn_no) #used for programming ease
				kb_start_found = False

	start_index.append(len(content)) #full dialog ease of programming
	return start_index,end_index
>>>>>>> b9142bdc5bb8c7718091544202fa6c1f1d987ffd

def _tokenize(sentence):
    return sentence.split() + [eos] 

def _tokenize_kb(sentence): 
    return sentence.split() + [eok]

def process_kb(given_kb):
<<<<<<< HEAD
    processed_kb = []
    for i in given_kb:
        processed_kb = processed_kb + [re.sub('\d+','',i,1).strip().split(' ')]
    if processed_kb:
        return processed_kb # only restaurant name is being returned if we are using processed_kb[0]
    return None

def get_all_dialogs(filename):
    fname=open(filename,'r')
    s = ''
    for i in fname.readlines():
        s = s + i
    all_=s.split('\n\n')
    fname.close()
    return all_[0:-1]

def get_vocab(train_fname,test_fname,dev_fname):
    train=get_all_dialogs(train_fname)
    test=get_all_dialogs(test_fname)
    dev=get_all_dialogs(dev_fname)
    all_dialogs=train+test+dev
    
    words=set([])
    for i in all_dialogs:
        dialog=i.split('\n')
        for utterance_response in dialog:
            utterances=re.sub('\d+','',utterance_response,1).strip().split('\t')
            words.update(utterances[0].split(' '))
            if len(utterances)>1:
                words.update(utterances[1].split(' '))
    w=sorted(words)
    for ind,i in enumerate(w):
        if len(i)>1 and i[len(i)-1]==',':
            w.remove(i)
        if len(i)>1 and '_' not in i and i[0].isupper() and '.' not in i:
            w[ind]=i[0].lower()+i[1:]
    return sorted(set(w))

def get_data(fname): #pass a file object only
    all_dialogues = get_all_dialogs(fname)

    pre_kb = []
    post_kb = []
    kb = []
    utterance = []
    response = []
    count=0
    for dialog_num , single_dialogue in enumerate(all_dialogues):
        history = [beg]
        content = single_dialogue.split('\n')
        len_of_dialogue = len(content)
        kb_start_index, kb_end_index,flag = locate_kb(content)
        if flag==1:
            count+=1
        if kb_start_index == -1: #no kb found
            kb_start_index = kb_end_index = len_of_dialogue
        for i in range(0,kb_start_index):
            utterance_response = content[i].split('\t')
            utterance_response[0]=re.sub('\d+','',utterance_response[0],1).strip()
            
            if len(utterance_response) < 2: #handles api call no result
                history = history + [re.sub('\d+','',content[i],1).strip().split(' ')]
                #print(dialog_num)
                continue
            pre_kb.append(history)
            current_utterance = utterance_response[0].split(' ')
            current_response = utterance_response[1].split(' ')
            kb.append([])
            post_kb.append([])
            utterance.append(current_utterance)
            response.append(current_response)
            history = history + [current_utterance] + [current_response]

        current_kb = process_kb(content[kb_start_index:kb_end_index])
        history = []

        for i in range(kb_end_index,len_of_dialogue):
            utterance_response = content[i].split('\t')
            utterance_response[0]=re.sub('\d+','',utterance_response[0],1).strip()
            pre_kb.append([])
            current_utterance = utterance_response[0].split(' ')
            if len(utterance_response)>1:
                current_response = utterance_response[1].split(' ')
            kb.append(current_kb)
            post_kb.append(history)
            utterance.append(current_utterance)
            response.append(current_response)
            
            history = history + [current_utterance] + [current_response]
    data = [pre_kb,kb,post_kb,utterance,response]
    print(count)
    return data

def data_stats(data): # can be coded up at a later stage
    
    for ind,d in enumerate(data):
        if ind==0: #pre
            pre_len=[]
            for context in d:
                pre_len.append(len(context))
        if ind==1: #KB
            kb_len=[]
            for context in d:
                kb_len.append(len(context))
        if ind==2: #ost
            post_len=[]
            for context in d:
                post_len.append(len(context))
        if ind==3: #utterance
            utt_len=[]
            for context in d:
                utt_len.append(len(context))
        if ind==4: #response
            resp_len=[]
            for context in d:
                resp_len.append(len(context))
    
    utterances_len=utt_len+resp_len
    
    return [max(pre_len),max(kb_len),max(post_len),max(utterances_len)]




def vec(w,words):
    if w in words.index:
        return words.loc[w].as_matrix()
    else:
        return 0

def get_vocab_init_matrix(glove_fname,v):
    np.random.seed(1234)
    
    glove_data_file=open(glove_fname,'r')
    words = pd.read_table(glove_data_file, sep=" ",index_col=0,header=None,quoting=csv.QUOTE_NONE)

    embedding_dict=words.T.to_dict('list')
    d={}
    
    for key,value in v.items():
        if key in embedding_dict.keys():
            d[key]=np.asarray(embedding_dict[key])
        
    no_emb=[]
    for i in v:
        if not hasattr(vec(i,words),"__len__"):
            no_emb.append(i)
    rand=np.random.rand(len(no_emb),100)
    for ind,item in enumerate(no_emb):
        d[item]=rand[ind]
        
    vocab={k: val for val, k in enumerate(d.keys())}
    init_matrix=np.asarray(list(d.values()))
    return vocab,init_matrix


def prepro(args):
    source_dir = args.source_dir 
    target_dir = args.target_dir
    source_fname = source_dir+ '/dialog-babi-task6-dstc2-'	
    target_fname = target_dir+ '/p-dialog-babi-task6-dstc2-'
    
    train_input = source_fname+ 'trn.txt'
    test_input = source_fname+ 'tst.txt'
    dev_input = source_fname+ 'dev.txt'
     
    train_output = get_data(train_input)
    test_output = get_data(test_input)
    dev_output = get_data(dev_input)
    
    train_stats=data_stats(train_output)
    test_stats=data_stats(test_output)
    dev_stats=data_stats(dev_output)
    
    total_stats=[max(test_stats[0],max(train_stats[0],dev_stats[0])),max(test_stats[1],max(train_stats[1],dev_stats[1])),
                 max(test_stats[2],max(train_stats[2],dev_stats[2])),max(test_stats[3],max(train_stats[3],dev_stats[3]))]
    
    
    vocab=get_vocab(train_input,test_input,dev_input)
    vocab.append('<beg>')
    vocab_dict={k: v for v, k in enumerate(vocab)}
    
    glove_file='data/glove.6B.100d.txt'
    vocab_final,init_matrix=get_vocab_init_matrix(glove_file,vocab_dict)

    
    with open(target_fname+'vocab.json','w+') as fp:
        json.dump(vocab_final,fp)
    
    pickle.dump(init_matrix,open(target_fname+'init.pkl','wb'))
    
    with open(target_fname+'train.json','w+') as fp1:
        json.dump(train_output,fp1)

    with open(target_fname+'test.json','w+') as fp2:
        json.dump(test_output,fp2)
    
    with open(target_fname+'dev.json','w+') as fp3:
        json.dump(dev_output,fp3)
    
    with open(target_fname+'stats.json','w+') as fp:
        json.dump(total_stats,fp)
    
=======
	processed_kb = []
	for i in given_kb:
		processed_kb = processed_kb + [re.sub('\d+','',i,1).strip().split(' ')]
	if processed_kb:
		return processed_kb # only restaurant name is being returned if we are using processed_kb[0]
	return None

def get_all_dialogs(filename):
	fname=open(filename,'r')
	s = ''
	for i in fname.readlines():
		s = s + i
	all_=s.split('\n\n')
	fname.close()
	return all_[0:-1]

def get_vocab(train_fname,test_fname,dev_fname):
	train=get_all_dialogs(train_fname)
	test=get_all_dialogs(test_fname)
	dev=get_all_dialogs(dev_fname)
	all_dialogs=train+test+dev
	
	words=set([])
	for i in all_dialogs:
		dialog=i.split('\n')
		for utterance_response in dialog:
			utterances=re.sub('\d+','',utterance_response,1).strip().split('\t')
			words.update(utterances[0].split(' '))
			if len(utterances)>1:
				words.update(utterances[1].split(' '))
	w=sorted(words)
	for ind,i in enumerate(w):
		if len(i)>1 and i[len(i)-1]==',':
			w.remove(i)
		if len(i)>1 and '_' not in i and i[0].isupper() and '.' not in i:
			w[ind]=i[0].lower()+i[1:]
	return sorted(set(w))

def get_data(fname): #pass a file object only
	all_dialogues = get_all_dialogs(fname)

	pre_kb = []
	post_kb = []
	kb = []
	utterance = []
	response = []
	count=0
	for dialog_num , single_dialogue in enumerate(all_dialogues):
		history = [beg]
		content = single_dialogue.split('\n')
		len_of_dialogue = len(content)
		kb_start_index, kb_end_index = locate_kb(content) #single arrays if no kb found
		kb_occurences = len(kb_start_index) - 1
		
		for i in range(0,kb_start_index[0]):
			utterance_response = content[i].split('\t')
			utterance_response[0]=re.sub('\d+','',utterance_response[0],1).strip()
			
			if len(utterance_response) < 2: #handles api call no result
				history = history + [re.sub('\d+','',content[i],1).strip().split(' ')]
				#print(dialog_num)
				continue
			pre_kb.append(history)
			current_utterance = utterance_response[0].split(' ')
			current_response = utterance_response[1].split(' ')
			kb.append([])
			post_kb.append([])
			utterance.append(current_utterance)
			response.append(current_response)
			history = history + [current_utterance] + [current_response]

		current_pre = history	#entire pre-kb conversation
		
		for m in range(0,kb_occurences):
		
		#kb processing
		
			current_kb = process_kb(content[kb_start_index[m]:kb_end_index[m]])
			
			
			if kb_occurences > 1: #adds the api call in the history for the second time.
				utterance_response = content[kb_start_index[m]-1].split('\t')
				utterance_response[0]=re.sub('\d+','',utterance_response[0],1).strip()
				current_utterance = utterance_response[0].split(' ')
				if len(utterance_response)>1:
					current_response = utterance_response[1].split(' ')
				current_pre = current_pre + [current_utterance] + [current_response]
			
			history = []

			for i in range(kb_end_index[m],kb_start_index[m+1]):
				utterance_response = content[i].split('\t')
				utterance_response[0]=re.sub('\d+','',utterance_response[0],1).strip()
				pre_kb.append(current_pre) #pre remains fixed over timesteps
				current_utterance = utterance_response[0].split(' ')
				if len(utterance_response)>1:
					current_response = utterance_response[1].split(' ')
				kb.append(current_kb)
				post_kb.append(history)
				utterance.append(current_utterance)
				response.append(current_response)
				
				history = history + [current_utterance] + [current_response]

	data = [pre_kb,kb,post_kb,utterance,response]
	print(count)
	return data

def data_stats(data): # can be coded up at a later stage
	
	for ind,d in enumerate(data):
		if ind==0: #pre
			pre_len=[]
			for context in d:
				pre_len.append(len(context))
		if ind==1: #KB
			kb_len=[]
			for context in d:
				kb_len.append(len(context))
		if ind==2: #ost
			post_len=[]
			for context in d:
				post_len.append(len(context))
		if ind==3: #utterance
			utt_len=[]
			for context in d:
				utt_len.append(len(context))
		if ind==4: #response
			resp_len=[]
			for context in d:
				resp_len.append(len(context))
	
	utterances_len=utt_len+resp_len
	
	return [max(pre_len),max(kb_len),max(post_len),max(utterances_len)]


'''

def vec(w,words):
	if w in words.index:
		return words.loc[w].as_matrix()
	else:
		return 0

def get_vocab_init_matrix(glove_fname,v):
	np.random.seed(1234)
	
	glove_data_file=open(glove_fname,'r')
	words = pd.read_table(glove_data_file, sep=" ",index_col=0,header=None,quoting=csv.QUOTE_NONE)

	embedding_dict=words.T.to_dict('list')
	d={}
	
	for key,value in v.items():
		if key in embedding_dict.keys():
			d[key]=np.asarray(embedding_dict[key])
		
	no_emb=[]
	for i in v:
		if not hasattr(vec(i,words),"__len__"):
			no_emb.append(i)
	rand=np.random.rand(len(no_emb),100)
	for ind,item in enumerate(no_emb):
		d[item]=rand[ind]
		
	vocab={k: val for val, k in enumerate(d.keys())}
	init_matrix=np.asarray(list(d.values()))
	return vocab,init_matrix
'''

def prepro(args):
	source_dir = args.source_dir 
	target_dir = args.target_dir
	source_fname = source_dir+ '/dialog-babi-task6-dstc2-'	
	target_fname = target_dir+ '/p-dialog-babi-task6-dstc2-'
	
	train_input = source_fname+ 'trn.txt'
	test_input = source_fname+ 'tst.txt'
	dev_input = source_fname+ 'dev.txt'
	 
	train_output = get_data(train_input)
	test_output = get_data(test_input)
	dev_output = get_data(dev_input)
	
	train_stats=data_stats(train_output)
	test_stats=data_stats(test_output)
	dev_stats=data_stats(dev_output)
	
	total_stats=[max(test_stats[0],max(train_stats[0],dev_stats[0])),max(test_stats[1],max(train_stats[1],dev_stats[1])),
				 max(test_stats[2],max(train_stats[2],dev_stats[2])),max(test_stats[3],max(train_stats[3],dev_stats[3]))]
	
	vocab=get_vocab(train_input,test_input,dev_input)
	vocab.append('<beg>')
	vocab_dict={k: v for v, k in enumerate(vocab)}
	
	glove_file='data/glove.6B.100d.txt'
	vocab_final,init_matrix=get_vocab_init_matrix(glove_file,vocab_dict)

	
	with open(target_fname+'vocab.json','w+') as fp:
		json.dump(vocab_final,fp)
	
	pickle.dump(init_matrix,open(target_fname+'init.pkl','wb'))
	
	with open(target_fname+'train.json','w+') as fp1:
		json.dump(train_output,fp1)

	with open(target_fname+'test.json','w+') as fp2:
		json.dump(test_output,fp2)
	
	with open(target_fname+'dev.json','w+') as fp3:
		json.dump(dev_output,fp3)
	
	with open(target_fname+'stats.json','w+') as fp:
		json.dump(total_stats,fp)
	
>>>>>>> b9142bdc5bb8c7718091544202fa6c1f1d987ffd
#    pickle.dump(train_output,open(target_fname+'trn.pkl','wb'))
#    pickle.dump(test_output,open(target_fname+'tst.pkl','wb'))
#    pickle.dump(dev_output,open(target_fname+'dev.pkl','wb'))

def main():
	args = get_args()
	prepro(args)

if __name__ == "__main__":
<<<<<<< HEAD
    main()
=======
	main()
>>>>>>> b9142bdc5bb8c7718091544202fa6c1f1d987ffd
