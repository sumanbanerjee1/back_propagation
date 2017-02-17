#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 23:08:11 2017

@author: suman
"""
import argparse
import cPickle, gzip,pickle
import numpy as np
#parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, help="Learning Rate")
parser.add_argument("--momentum", type=float, help="Momentum Factor")
parser.add_argument("--num_hidden", type=int, help="Number of hidden Layers")
parser.add_argument("--sizes",type=str, help="comma separated lsit for sizes of hidden Layers")
parser.add_argument("--activation", help="activation function in the hidden layers", choices=['tanh','sigmoid'])
parser.add_argument("--loss", help="Loss function",choices=['sq','ce'])
parser.add_argument("--opt", help="Optimization algorithm",choices=['gd','momentum','nag','adam'])
parser.add_argument("--batch_size", type=int, help="Batch size in mini batch training")
parser.add_argument("--anneal", help="Annealing of Learning rate",choices=['true','false'])
parser.add_argument("--save_dir", type=str, help="Directory in which model is to be saved")
parser.add_argument("--expt_dir", type=str, help="Directory in which log files are to be saved")
parser.add_argument("--mnist", type=str, help="Directory in which mnist data is stored")
args = parser.parse_args()
save_dir=args.save_dir
expt_dir=args.expt_dir
mnist=args.mnist
# Load the dataset
f = gzip.open(mnist, 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
f1=open('data/mnist_distorted.pkl','r')
new_train=pickle.load(f1)
f1.close()

temp1=np.concatenate((train_set[0],new_train[0]),axis=0)
temp2=np.concatenate((train_set[1],train_set[1]),axis=0)
data=[temp1]

data.append(temp2)
#initializations
input_units=data[0].shape[1]
output_units=10
N=data[0].shape[0]
np.random.seed(1234)
num_hidden=args.num_hidden
sizes=[int(item) for item in args.sizes.split(',')]
max_epochs=50
lr=args.lr
threshold=100.0
d=data[0].shape[1]
batch_size=args.batch_size
momentum=args.momentum
beta1=0.9
beta2=0.999
eps=1e-8
activation=args.activation
loss_func=args.loss
anneal=args.anneal
opt=args.opt
lambdaa=0.01
#random initialization of weights
W=(2*np.random.random((input_units,sizes[0]))-1)
weights=[W]
for i in range(num_hidden-1):
    W=(2*np.random.random((sizes[i],sizes[i+1]))-1)
    weights.append(W)
W=(2*np.random.random((sizes[num_hidden-1],output_units))-1)
weights.append(W)

#random initialization of biases
biases=[]
for i in range(num_hidden):
    b=(2*np.random.random((sizes[i]))-1)
    biases.append(b)
b=(2*np.random.random((output_units))-1)
biases.append(b)
#one hot encoding
def one_hot(data):
    t=np.zeros([data[0].shape[0],output_units])
    t[np.arange(data[0].shape[0]),data[1]]=1
    return t
target=one_hot(data)
#activation functions and their derivatives       
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
def tanh(x):
    return np.tanh(x)
def dsigmoid(x):
    return sigmoid(x)*(1.0-sigmoid(x))
def dtanh(x):
    return 1.0-tanh(x)**2
def softmax(x):
    ex=np.exp(x)
    return ex/np.sum(ex,axis=1,keepdims=True)
#initializations of coneections to zero
def initialize_connectionsw():
    temp1=np.zeros((input_units,sizes[0]))
    temp2=[temp1]
    for i in range(num_hidden-1):
        temp1=np.zeros((sizes[i],sizes[i+1]))
        temp2.append(temp1)
    temp1=np.zeros((sizes[num_hidden-1],output_units))
    temp2.append(temp1)
    return temp2
def initialize_connectionsb():
    bb=[]
    for i in range(num_hidden):
        b=np.zeros((sizes[i]))
        bb.append(b)
    b=np.zeros((output_units))
    bb.append(b)
    return bb

#predict the labels
def predict(data,weights,biases):
    a=np.add(np.dot(data[0],weights[0]),biases[0])
    preact=[a]
    if activation=='sigmoid':
        h=[sigmoid(a)]
    elif activation=='tanh':
        h=[tanh(a)]
    if num_hidden!=1:
        for i in range(num_hidden)[1:]:
            h_local_preactivation=np.add(np.dot(h[i-1],weights[i]),biases[i])
            preact.append(h_local_preactivation)
            if activation=='sigmoid':
                h_local_activation=sigmoid(h_local_preactivation)
            elif activation=='tanh':
                h_local_activation=tanh(h_local_preactivation)
            h.append(h_local_activation)
    else:
        i=0
    h_local_preactivation=np.add(np.dot(h[i],weights[num_hidden]),biases[num_hidden])
    h_local_activation=softmax(h_local_preactivation)
    h.append(h_local_activation)
    preact.append(h_local_preactivation)
    return (np.argmax(h[num_hidden],axis=1),h[num_hidden])
#save the model
def save():
    path=save_dir+'/model.pkl.gz'
    f=open(path,'w')
    cPickle.dump([weights,biases],f)
    f.close()
#load the model
def load():
    path=save_dir+'/model.pkl.gz'
    f=open(path,'r')
    [weights,biases]=cPickle.load(f)
    f.close()
#accuracy
def accuracy(data):
    yhat,yhatp=predict(data,weights,biases)
    count=0.0
    for i in range(data[0].shape[0]):
        if yhat[i]==data[1][i]:
            count+=1
            
    return (count/data[0].shape[0])*100
#cross entropy loss
def cross_entropy(data):
    yhat,yhatp=predict(data,weights,biases)
    truelabel=one_hot(data)
    yl=np.multiply(yhatp,truelabel)
    yl=yl[yl!=0]
    yl=-np.log(yl)
    yl=np.mean(yl)
    return yl
#squared loss
def squared_error(data):
    yhat,yhatp=predict(data,weights,biases)
    truelabel=one_hot(data)
    l=0.5*(yhatp-truelabel)**2
    loss=np.mean(l)
    return loss
#forward propagation
def forwardpass(weights,biases,point):
    a=np.add(np.dot(points,weights[0]),biases[0])
    preact=[a]
    
    if activation=='sigmoid':
        h=[sigmoid(a)]
    elif activation=='tanh':
        h=[tanh(a)]
    
    if num_hidden!=1:
        for i in range(num_hidden)[1:]:
            h_local_preactivation=np.add(np.dot(h[i-1],weights[i]),biases[i])
            preact.append(h_local_preactivation)
            if activation=='sigmoid':
                h_local_activation=sigmoid(h_local_preactivation)
            elif activation=='tanh':
                h_local_activation=tanh(h_local_preactivation)        
            h.append(h_local_activation)
    else:
        i=0
    h_local_preactivation=np.add(np.dot(h[i],weights[num_hidden]),biases[num_hidden])
    h_local_activation=softmax(h_local_preactivation)
    h.append(h_local_activation)
    preact.append(h_local_preactivation)
    return (h,preact)

#iniialize the delta changes
def initialize():
    m=[]
    for i in range(num_hidden+1):
        init=np.zeros(num_hidden)
        m.append(init)
    return m
#backward propagation
def backwardpass(h,preact):
    da=initialize()
    dh=initialize()
    dw=initialize()
    db=initialize()
    #derivatives of activations for the hidden layers
    derivs=[]
    for i in range(num_hidden+1):
        derivative=dsigmoid(preact[i])
        derivs.append(derivative)
    
    if loss_func=='sq':
        temp1=h[num_hidden]-target[index:index+batch_size,0:d]
        l=np.zeros([batch_size,output_units])
        for i in range(output_units):
            temp2=(target[index:index+batch_size,0:d][:,i]-h[num_hidden][:,i])*target[index:index+batch_size,0:d][:,i]
            l+=temp1*temp2[:,np.newaxis]
        
        da[num_hidden]=l
    
    elif loss_func=='ce':    
        da[num_hidden]=h[num_hidden]-target[index:index+batch_size,0:d]
    for k in range(num_hidden,0,-1):
        dw[k]=np.dot(np.transpose(h[k-1]),da[k])
        db[k]=da[k]    
        dh[k-1]=np.dot(da[k],np.transpose(weights[k]))
        da[k-1]=np.multiply(dh[k-1],derivs[k-1])
    k=k-1
    dw[k]=np.dot(np.transpose(points),da[k])
    db[k]=da[k]
    
    #average
    for i in range(num_hidden+1):
        db[i]=np.mean(db[i],axis=0)# averaging the biases
        dw[i]=dw[i]*(threshold/np.linalg.norm(dw[i]))#gradient clipping
    return (dw,db)
#some more initializations
loss_train,loss_valid,loss_test=np.zeros([max_epochs]),np.zeros([max_epochs]),np.zeros([max_epochs])
prev_v_w,prev_v_b,vw,vb=initialize_connectionsw(),initialize_connectionsb(),initialize_connectionsw(),initialize_connectionsb()
mw,mb=initialize_connectionsw(),initialize_connectionsb()
loss_valid_prev=1000
f1=open(expt_dir+"/log_loss_train.txt", 'w+')
f2=open(expt_dir+"/log_loss_valid.txt", 'w+')
f3=open(expt_dir+"/log_loss_test.txt", 'w+')
f4=open(expt_dir+"/log_err_train.txt", 'w+')
f5=open(expt_dir+"/log_err_valid.txt", 'w+')
f6=open(expt_dir+"/log_err_test.txt", 'w+')
f7=open(expt_dir+"/valid_predictions.txt", 'w+')
f8=open(expt_dir+"/test_predictions.txt", 'w+')

count=0
#training begins!!
for epochs in range(max_epochs):
    save()#save model in each epoch
    step=1
    
    for index in range(0,N,batch_size):
        points=data[0][index:index+batch_size,0:d]#get all training data points in a batch
        h,preact=forwardpass(weights,biases,points)#get the activations and pre-activations
        if step%100==0 and step!=0:
            if loss_func=='sq': 
                loss_train_stepwise=squared_error(train_set)
                loss_valid_stepwise=squared_error(valid_set)
                loss_test_stepwise=squared_error(valid_set)
            elif loss_func=='ce':
                loss_train_stepwise=cross_entropy(train_set)
                loss_valid_stepwise=cross_entropy(valid_set)
                loss_test_stepwise=cross_entropy(valid_set)
            error_rate_valid=(100-accuracy(valid_set))
            error_rate_train=(100-accuracy(train_set))
            error_rate_test=(100-accuracy(test_set))
            
            f1.write("Epoch: %d, Step %d, Loss: %f, lr: %f \n" % (epochs,step,loss_train_stepwise,lr))
            f2.write("Epoch: %d, Step %d, Loss: %f, lr: %f \n" % (epochs,step,loss_valid_stepwise,lr))
            f3.write("Epoch: %d, Step %d, Loss: %f, lr: %f \n" % (epochs,step,loss_test_stepwise,lr))
            f4.write("Epoch: %d, Step %d, Error: %2.2f, lr: %f \n" % (epochs,step,error_rate_train,lr))
            f5.write("Epoch: %d, Step %d, Error: %2.2f, lr: %f \n" % (epochs,step,error_rate_valid,lr))
            f6.write("Epoch: %d, Step %d, Error: %2.2f, lr: %f \n" % (epochs,step,error_rate_test,lr))
            print "Epoch",epochs,", Step",step,", Loss(validation):",loss_valid_stepwise,",error rate(validation):",error_rate_valid,"lr:",lr
        #vanilla Gradient Descent
        if opt=='gd':
            dw,db=backwardpass(h,preact)
        
            #update rule
            for i in range(num_hidden+1):
                dw[i]=dw[i]+lambdaa*weights[i]
                weights[i]=weights[i]-lr*dw[i]
                biases[i]=biases[i]-lr*db[i]
            
            step+=1
        #Nesterov's Accelerated Gradient
        elif opt=='nag':
            #NAG
            for i in range(num_hidden,-1,-1):
                weights[i]=weights[i]-momentum*prev_v_w[i]
                biases[i]=biases[i]-momentum*prev_v_b[i]
            
            dw,db=backwardpass(h,preact)
            
            for i in range(num_hidden,-1,-1):
                dw[i]=dw[i]+lambdaa*weights[i]
                vw[i]=momentum*prev_v_w[i]+lr*dw[i]
                vb[i]=momentum*prev_v_b[i]+lr*db[i]
                
                weights[i]=weights[i]-vw[i]
                biases[i]=biases[i]-vb[i]
                prev_v_w[i]=vw[i]
                prev_v_b[i]=vb[i]
            
            step+=1
        
        #ADAM        
        elif opt=='adam':
            #update rule ADAM
            dw,db=backwardpass(h,preact)
            for i in range(num_hidden,-1,-1):
                dw[i]=dw[i]+lambdaa*weights[i]
                mw[i]=beta1*mw[i]+(1-beta1)*dw[i]
                mb[i]=beta1*mb[i]+(1-beta1)*db[i]
                  
                vw[i]=beta2*vw[i]+(1-beta2)*dw[i]**2
                vb[i]=beta2*vb[i]+(1-beta2)*db[i]**2
                
                weights[i]=weights[i]-(lr/(np.sqrt(np.linalg.norm(vw[i])+eps)))*mw[i]
                biases[i]=biases[i]-(lr/(np.sqrt(np.linalg.norm(vb[i])+eps)))*mb[i]
            step+=1
        #Momentum based gradient descent        
        elif opt=='momentum':
            dw,db=backwardpass(h,preact)
        
            #update rule
            for i in range(num_hidden,-1,-1):
                dw[i]=dw[i]+lambdaa*weights[i]
                vw[i]=momentum*prev_v_w[i]+lr*dw[i]
                vb[i]=momentum*prev_v_b[i]+lr*db[i]
            
                weights[i]=weights[i]-vw[i]
                biases[i]=biases[i]-vb[i]

                prev_v_w[i]=vw[i]
                prev_v_b[i]=vb[i]
            step+=1
    
    #get the loss on train set and validation set       
    if loss_func=='sq': 
        loss_train[epochs]=squared_error(train_set)
        loss_valid[epochs]=squared_error(valid_set)
        loss_test[epochs]=squared_error(test_set)

    elif loss_func=='ce':
        loss_train[epochs]=cross_entropy(train_set)
        loss_valid[epochs]=cross_entropy(valid_set)
        loss_test[epochs]=cross_entropy(test_set)

    #anneal the loss
    if anneal=='true':
        if loss_valid[epochs]>loss_valid_prev:
            epochs=epochs-1
            lr=lr*0.7
            load()
            print 'annealed'
            count=count+1
        else:
            count=0
    loss_valid_prev=loss_valid[epochs]
    if count>10:
        break
    print "Epoch:",epochs,"Accuracy -- train set: ",accuracy(train_set)," test set: ",accuracy(test_set)
f1.close()
f2.close()
f3.close()
f4.close()
f5.close()
f6.close()
labels_valid=predict(valid_set,weights,biases)
labels_valid_string= '\n'.join(map(str, labels_valid[0]))
labels_test=predict(test_set,weights,biases)
labels_test_string= '\n'.join(map(str, labels_test[0]))
f7.write(labels_valid_string)
f8.write(labels_test_string)
f7.close()
f8.close()


