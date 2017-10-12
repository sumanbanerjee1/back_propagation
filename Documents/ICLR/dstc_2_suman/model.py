#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 15:03:21 2017

@author: suman
"""
import tensorflow as tf
import numpy as np

class VariableEmbedder(object):
<<<<<<< HEAD
    def __init__(self, params, initializer=None, glove=True, name="variable_embedder"):
        V, d = params['vocab_size'],params['hidden_size']
        embeddings_matrix = params['emb_matrix']
        with tf.variable_scope(name):
            nil_word_slot = tf.constant(np.zeros([1, d]), dtype=tf.float32)
            if not glove:
                self.E = tf.get_variable("emb_mat", dtype=tf.float32, shape=[V, d], initializer=initializer)
            else:
                self.E = tf.get_variable("emb_mat", dtype=tf.float32, shape=[V, d], initializer=tf.constant_initializer(np.array(embeddings_matrix)))
            
            self.emb_mat = tf.concat([nil_word_slot,self.E],0)
=======
    def __init__(self, params, embeddings_matrix, wd=0.0, initializer=None, glove=True, name="variable_embedder"):
        V, d = params.vocab_size[0], params.hidden_size
        with tf.variable_scope(name):
            if not glove:
                self.emb_mat = tf.get_variable("emb_mat", dtype='float', shape=[V, d], initializer=initializer)
            else:
                self.emb_mat = tf.get_variable("emb_mat", dtype='float', shape=[V, d], initializer=tf.constant_initializer(np.array(embeddings_matrix)))
>>>>>>> b9142bdc5bb8c7718091544202fa6c1f1d987ffd

    def __call__(self, word, name="embedded_content"):
        out = tf.nn.embedding_lookup(self.emb_mat, word, name=name)
        return out


class PositionEncoder(object):
    def __init__(self, max_sent_size, hidden_size):
        self.max_sent_size, self.hidden_size = max_sent_size, hidden_size
        J, d = max_sent_size, hidden_size
        with tf.name_scope("pe_constants"):
            b = [1 - k/d for k in range(1, d+1)]
            w = [[j*(2*k/d - 1) for k in range(1, d+1)] for j in range(1, J+1)]
            self.b = tf.constant(b, shape=[d])
            self.w = tf.constant(w, shape=[J, d])

<<<<<<< HEAD
    def __call__(self, Ax, scope=None):
        with tf.name_scope(scope or "position_encoder"):
            shape = Ax.get_shape().as_list()
            length_dim_index = len(shape) - 2
            #length = tf.reduce_sum(tf.cast(mask, 'float'), length_dim_index)
            #length = tf.maximum(length, 1.0)  # masked sentences will have length 0
            #length_aug = tf.expand_dims(tf.expand_dims(length, -1), -1)
            # l = self.b + self.w/length_aug
            l = self.b + self.w/self.max_sent_size
            #mask_aug = tf.expand_dims(mask, -1)
            #mask_aug_cast = tf.cast(mask_aug, 'float')
            l_cast = tf.cast(l, 'float')
            f = tf.reduce_sum(Ax * l_cast, length_dim_index, name='f')  # [N, S, d]
=======
    def __call__(self, Ax, mask, scope=None):
        with tf.name_scope(scope or "position_encoder"):
            shape = Ax.get_shape().as_list()
            length_dim_index = len(shape) - 2
            length = tf.reduce_sum(tf.cast(mask, 'float'), length_dim_index)
            length = tf.maximum(length, 1.0)  # masked sentences will have length 0
            length_aug = tf.expand_dims(tf.expand_dims(length, -1), -1)
            # l = self.b + self.w/length_aug
            l = self.b + self.w/self.max_sent_size
            mask_aug = tf.expand_dims(mask, -1)
            mask_aug_cast = tf.cast(mask_aug, 'float')
            l_cast = tf.cast(l, 'float')
            f = tf.reduce_sum(Ax * l_cast * mask_aug_cast, length_dim_index, name='f')  # [N, S, d]
>>>>>>> b9142bdc5bb8c7718091544202fa6c1f1d987ffd

            return f


<<<<<<< HEAD
class Model(object):
    def __init__(self,sess,params):
        self.sess=sess
        self.batch_size, self.max_sent_size = params['batch_size'], params['max_sent_size']
        self.vocab_size, self.emb_dim = params['vocab_size'], params['hidden_size']
        self.max_pre_size, self.max_kb_size, self.max_post_size = params['max_pre_size'], params['max_kb_size'], params['max_post_size']
        initializer = tf.random_uniform_initializer(-np.sqrt(3), np.sqrt(3))
        self.build_inputs()
        
        
        with tf.variable_scope("embedding"):
            A = VariableEmbedder(params, initializer=initializer, name='Embedder')
            self.pre_emb = A(self.pre, name='embedded_pre')
            self.kb_emb = A(self.kb, name='embedded_kb')
            self.post_emb = A(self.post, name='embedded_post')
            self.query_emb = A(self.query, name='embedded_query')
            self.response_emb = A(self.response, name='embedded_response')
            
        with tf.name_scope("encoding"):
            encoder = PositionEncoder(self.max_sent_size, self.emb_dim)
            self.pre_enc = encoder(self.pre_emb)
            self.kb_enc = encoder(self.kb_emb)
            self.post_enc = encoder(self.post_emb)
            self.query_enc = encoder(self.query_emb)
            self.response_enc = encoder(self.response_emb)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
    def batch_fit(self,batch_dict):
        flags= tf.app.flags
        feed_dict={self.pre: batch_dict['pre'],
                   self.kb: batch_dict['kb'],
                   self.post: batch_dict['post'],
                   self.query: batch_dict['query'],
                   self.response: batch_dict['response']
                
                }
        return self.sess.run(self.pre_enc,feed_dict=feed_dict)
    
    def build_inputs(self):
        flags=tf.app.flags
        self.pre = tf.placeholder(tf.int64,[None,self.max_pre_size,self.max_sent_size], name="PRE")
        self.kb = tf.placeholder(tf.int64,[None,self.max_kb_size,3], name="KB")
        self.post = tf.placeholder(tf.int64,[None,self.max_post_size,self.max_sent_size], name="POST")
        self.query = tf.placeholder(tf.int64,[None,self.max_sent_size], name="QUERY")
        self.response = tf.placeholder(tf.int64,[None,self.max_sent_size], name = "RESPONSE")
        
        
        
=======
class model(object):
    def __init__(params):
        N, J, Q = 32,21,23
>>>>>>> b9142bdc5bb8c7718091544202fa6c1f1d987ffd
        
      
    
