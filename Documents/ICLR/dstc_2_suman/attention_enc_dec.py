#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 22:52:34 2017

@author: suman
"""

import tensorflow as tf
<<<<<<< HEAD
import numpy as np
from tensorflow.python.util import nest
from tensorflow.contrib.rnn.python.ops.rnn_cell import _linear


class VariableEmbedder(object):
    def __init__(self, FLAGS, initializer=None, glove=True, name="variable_embedder"):
        V, d = FLAGS.vocab_size,FLAGS.hidden_size
        embeddings_matrix = FLAGS.emb_matrix
        with tf.variable_scope(name):
            nil_word_slot = tf.constant(np.zeros([1, d]), dtype=tf.float32)
            if not glove:
                self.E = tf.get_variable("emb_mat", dtype=tf.float32, shape=[V, d], initializer=initializer)
            else:
                self.E = tf.get_variable("emb_mat", dtype=tf.float32, shape=[V, d], initializer=tf.constant_initializer(np.array(embeddings_matrix)))
            
            self.emb_mat = tf.concat([nil_word_slot,self.E],0)

    def __call__(self, word, name="embedded_content"):
        out = tf.nn.embedding_lookup(self.emb_mat, word, name=name)
        return out
    def get_emb_mat():
        return self.emb_mat

class PositionEncoder(object):
    def __init__(self, max_sent_size, hidden_size):
        self.max_sent_size, self.hidden_size = max_sent_size, hidden_size
        J, d = max_sent_size, hidden_size
        with tf.name_scope("pe_constants"):
            b = [1 - k/d for k in range(1, d+1)]
            w = [[j*(2*k/d - 1) for k in range(1, d+1)] for j in range(1, J+1)]
            self.b = tf.constant(b, shape=[d])
            self.w = tf.constant(w, shape=[J, d])

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

            return f

     
def rnn_cell(FLAGS, dropout, scope,decoder_cell=False):
=======
from tensorflow.python.util import nest

def rnn_cell(FLAGS, dropout, scope):
>>>>>>> b9142bdc5bb8c7718091544202fa6c1f1d987ffd

    with tf.variable_scope(scope):
        # Get the cell type
        if FLAGS.rnn_unit == 'gru':
            rnn_cell_type = tf.nn.rnn_cell.GRUCell
        elif FLAGS.rnn_unit == 'lstm':
            rnn_cell_type = tf.nn.rnn_cell.BasicLSTMCell
        else:
            raise Exception("Choose a valid RNN unit type.")

        # Single cell
<<<<<<< HEAD
        if decoder_cell:
            d=FLAGS.decoder_num_hidden_units
        else:
            d=FLAGS.num_hidden_units
        single_cell = rnn_cell_type(d)
=======
        single_cell = rnn_cell_type(FLAGS.num_hidden_units)
>>>>>>> b9142bdc5bb8c7718091544202fa6c1f1d987ffd

        # Dropout
        single_cell = tf.nn.rnn_cell.DropoutWrapper(single_cell,
            output_keep_prob=1-dropout)

        # Each state as one cell
        stacked_cell = tf.contrib.rnn.MultiRNNCell(
            [single_cell] * FLAGS.num_layers)

        return stacked_cell

#def rnn_inputs(FLAGS, input_data, vocab_size, scope):
#
#    with tf.variable_scope(scope, reuse=True):
#        W_input = tf.get_variable("W_input",
#            [vocab_size, FLAGS.num_hidden_units])
#
#        # embeddings will be shape [input_data dimensions, num_hidden units]
#        embeddings = tf.nn.embedding_lookup(W_input, input_data)
#        return embeddings

def _extract_argmax_and_embed(W_embedding, output_projection,
    update_embedding=True):
    """
    Extract the argmax from the decoder outputs and use the output_projection
    weights to project the predicted output as an input to the next
    decoder cell. update_embedding will either allow or freeze W_embedding.

    Return will be a loop function.
    """

    def loop_function(prev, _):
        """
        prev is the previous decoder output. _ is just a placeholder
        for something like an index i.
        """

        # xW + b to convert decoder ouput [N, H] to shape [N, C]
        ''' Recall that decoder inputs are time-major so the output from any one
            cell is [N, H] '''
        prev = tf.matmul(prev, output_projection[0]) + output_projection[1]

        # Extract argmax
        prev_symbol = tf.argmax(prev, dimension=1)

        # Need to embed the prev_symbol before feeding into next decoder cell
        embedded_prev_symbol = tf.nn.embedding_lookup(W_embedding, prev_symbol)

<<<<<<< HEAD
        # Stop the gradient update if update_embedding is False
        ''' This means the embedding the output projection will not alter the
=======
        # Stop the gradient update is update_embedding is False
        ''' This mean the embedding the output projection will not alter the
>>>>>>> b9142bdc5bb8c7718091544202fa6c1f1d987ffd
            decoder input embeddings (W_embedding). '''
        if not update_embedding:
            embedded_prev_symbol = tf.stop_gradient(embedded_prev_symbol)

        return embedded_prev_symbol

    return loop_function

<<<<<<< HEAD
def attention_decoder(decoder_inputs, initial_state, pre_attention_states,
    post_attention_states,kb_attention_states,cell, output_size,
    num_heads=1, loop_function=None, dtype=None,scope=None,
    initial_state_attention=False):
=======
def attention_decoder(decoder_inputs, initial_state, attention_states,
    cell, output_size, num_heads=1, loop_function=None, dtype=None,
    scope=None, initial_state_attention=False):
>>>>>>> b9142bdc5bb8c7718091544202fa6c1f1d987ffd
    """
    The decoder with an attentional interface.
    """

    # Set the scope
    with tf.variable_scope(scope or 'attention_decoder', dtype=dtype) as scope:

        # set the dtype
        dtype = scope.dtype

        # Get sizes
        batch_size = tf.shape(decoder_inputs[0])[0] # decoder_inputs is a list
<<<<<<< HEAD
        pre_attn_length = pre_attention_states.get_shape()[1].value # [N, <max_len> H]
        if pre_attn_length == None: # encoder inputs placeholder had None for 2nd D
            pre_attn_length = tf.shape(pre_attention_states)[1]
        pre_attn_size = pre_attention_states.get_shape()[2].value


        post_attn_length = post_attention_states.get_shape()[1].value # [N, <max_len> H]
        if post_attn_length == None: # encoder inputs placeholder had None for 2nd D
            post_attn_length = tf.shape(post_attention_states)[1]
        post_attn_size = post_attention_states.get_shape()[2].value
=======
        attn_length = attention_states.get_shape()[1].value # [N, <max_len> H]
        if attn_length == None: # encoder inputs placeholder had None for 2nd D
            attn_length = tf.shape(attention_states)[1]
        attn_size = attention_states.get_shape()[2].value
>>>>>>> b9142bdc5bb8c7718091544202fa6c1f1d987ffd

        # Use a 1x1 convolution to process encoder hidden states into features
        ''' This is an additional step we use (not in the paper). Instead of
            using attention on the raw hidden state outputs from the encoder,
            we use convolution to extract meaningful features from the hidden
            states. We will then apply attention on our ouputs from the conv.

            Recall, to calculate e_ij we need to feed in our encoder hidden
            states and the previous hidden state from the decoder into a tanh.
            We could feed in the raw encoder hidden states and previous decoder
            hidden state in raw into the tanh and apply the nonlinearity weights
            and then take the tanh of that. But, instead, we split it up. The
            nonlinearity is applied to the encoder hidden states via this conv
            operations and the previous decoder hidden state (query) goes
            through its own separate weights with _linear. Finally both results
            are added and then the regular tanh is applied. We multiply this
            value by v (which acts as softmax weights)

            Transformation in shape:
                original hidden state:
                    [N, max_len, H]
                reshaped to 4D hidden:
                    [N, max_len, 1, H] = N images of [max_len, 1, H]
                    so we can apply filter
                filter:
                    [1, 1, H, H] = [height, width, depth, # num filters]
                Apply conv with stride 1 and padding 1:
                    H = ((H - F + 2P) / S) + 1 =
                        ((max_len - 1 + 2)/1) + 1 = height'
                    W = ((W - F + 2P) / S) + 1 = ((1 - 1 + 2)/1) + 1 = 3
                    K = K = H
                    So we just converted a
                        [N, max_len, H] into [N, height', 3, H]

        '''
<<<<<<< HEAD
        hidden_pre = tf.reshape(pre_attention_states,
            [-1, pre_attn_length, 1, pre_attn_size]) # [N, max_len, 1, H]
        hidden_post = tf.reshape(post_attention_states,
            [-1, post_attn_length, 1, post_attn_size]) # [N, max_len, 1, H]
        hidden_features_pre = []
        hidden_features_post = []
        V1 = []
        V3 = []
        for a in range(num_heads):
            # filter
            k1 = tf.get_variable("AttnW1_%d" % a,
                [1, 1, pre_attn_size, pre_attn_size]) # [1, 1, H, H]
            k3= tf.get_variable("AttnW3_%d" % a,
                [1, 1, post_attn_size, post_attn_size]) # [1, 1, H, H]
            
            hidden_features_pre.append(tf.nn.conv2d(hidden_pre, k1, [1,1,1,1], "SAME"))
            hidden_features_post.append(tf.nn.conv2d(hidden_post, k3, [1,1,1,1], "SAME"))

            V1.append(tf.get_variable(
                "V1_attention_softmax_%d" % a, [pre_attn_size]))
            V3.append(tf.get_variable(
                "V3_attention_softmax_%d" % a, [post_attn_size]))
=======
        hidden = tf.reshape(attention_states,
            [-1, attn_length, 1, attn_size]) # [N, max_len, 1, H]
        hidden_features = []
        attention_softmax_weights = []
        for a in range(num_heads):
            # filter
            k = tf.get_variable("AttnW_%d" % a,
                [1, 1, attn_size, attn_size]) # [1, 1, H, H]
            hidden_features.append(tf.nn.conv2d(hidden, k, [1,1,1,1], "SAME"))
            attention_softmax_weights.append(tf.get_variable(
                "W_attention_softmax_%d" % a, [attn_size]))
>>>>>>> b9142bdc5bb8c7718091544202fa6c1f1d987ffd

        state = initial_state

        def attention(query):
            """
            Places an attention mask on hidden states from encoder
            using hidden and query. Query is a state of shape [N, H]
            """
            # results of the attention reads
            cs = [] # context vectors c_i

            # Flatten the query if it is a tuple
            if nest.is_sequence(query):
                # converts query from [N, H] to list of size N if [H, 1]
                query_list = nest.flatten(query)
            query = tf.concat(1, query_list) # becomes [H, N]

<<<<<<< HEAD
            for i in range(num_heads):
                with tf.variable_scope("Attention_%d" % i) as scope:
                    y = _linear(
                        args=query, output_size=pre_attn_size, bias=True)

                    # Reshape into 4D
                    y = tf.reshape(y, [-1, 1, 1, pre_attn_size]) # [N, 1, 1, H]

                    # Calculating alpha
                    s = tf.reduce_sum(V1[i] * tf.nn.tanh(hidden_features_pre[i] + y), [2, 3])
=======
            for a in range(num_heads):
                with tf.variable_scope("Attention_%d" % a) as scope:
                    y = tf.nn.rnn_cell._linear(
                        args=query, output_size=attn_size, bias=True)

                    # Reshape into 4D
                    y = tf.reshape(y, [-1, 1, 1, attn_size]) # [N, 1, 1, H]

                    # Calculating alpha
                    s = tf.reduce_sum(
                        attention_softmax_weights[a] *
                        tf.nn.tanh(hidden_features[a] + y), [2, 3])
>>>>>>> b9142bdc5bb8c7718091544202fa6c1f1d987ffd
                    a = tf.nn.softmax(s)

                    # Calculate context c
                    c = tf.reduce_sum(tf.reshape(
<<<<<<< HEAD
                        a, [-1, pre_attn_length, 1, 1])*hidden_pre, [1,2])
                    cs.append(tf.reshape(c, [-1, pre_attn_size]))
=======
                        a, [-1, attn_length, 1, 1])*hidden, [1,2])
                    cs.append(tf.reshape(c, [-1, attn_size]))
>>>>>>> b9142bdc5bb8c7718091544202fa6c1f1d987ffd

            return cs

        outputs = []
        prev = None
        batch_attn_size = tf.pack([batch_size, attn_size])
        attns = [tf.zeros(batch_attn_size, dtype=dtype)
            for _ in range(num_heads)]
        for a in attns:
            a.set_shape([None, attn_size])

        # Process decoder inputs one by one
        for i, inp in enumerate(decoder_inputs):

            if i > 0:
                tf.get_variable_scope().reuse_variables()

            if loop_function is not None and prev is not None:
                with tf.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)

            # Merge the input and attentions together
            input_size = inp.get_shape().with_rank(2)[1]
<<<<<<< HEAD
            x = _linear(
=======
            x = tf.nn.rnn_cell._linear(
>>>>>>> b9142bdc5bb8c7718091544202fa6c1f1d987ffd
                args=[inp]+attns, output_size=input_size, bias=True)

            # Decoder RNN
            cell_outputs, state = cell(x, state) # our stacked cell

            # Attention mechanism to get Cs
            attns = attention(state)

            with tf.variable_scope('attention_output_projection'):
<<<<<<< HEAD
                output = _linear(
=======
                output = tf.nn.rnn_cell._linear(
>>>>>>> b9142bdc5bb8c7718091544202fa6c1f1d987ffd
                    args=[cell_outputs]+attns, output_size=output_size,
                    bias=True)
            if loop_function is not None:
                prev = output
            outputs.append(output)

        return outputs, state

def embedding_attention_decoder(decoder_inputs, initial_state,
<<<<<<< HEAD
    pre_attention_states,post_attention_states,kb_attention_states,
    cell, num_symbols, embedding_size, output_projection,W_embedding,
=======
    attention_states, cell, num_symbols, embedding_size, output_projection,
>>>>>>> b9142bdc5bb8c7718091544202fa6c1f1d987ffd
    feed_previous, num_heads=1, update_embedding_for_previous=True,
    scope=None, dtype=None, initial_state_attention=False):
    """
    Embedding attention decoder which will produce the decoder outputs
    and state. Feed previous will be used to determine wether to feed in the
    previous decoder output into consideration for the next prediction or not.

    Args:
    decoder_inputs: <list> in time-major [<max_len>, N],
    initial_state: [N, H],
    attention_states: [batch_size, attn_length, attn_size] = [N, <max_len> H],
    cell: decoder layered cell,
    num_symbols: sp vocab size,
    embedding_size: embedding size (usually just H),
    output_projection: (W, b) for softmax and projection of decoder outputs
    feed_previous: boolean (True if feeding in previous decoder output)
    num_heads: number of attention heads (usually just 1)
    update_embedding_for_previous: boolean, if feed_previous is False, this
        variable has no effect. If feed_previous is True and this is True,
        then the softmax/output_projection (which are same set of weights)
        weights will be updated when using the decoder output, embedding it,
        and feeding into the next decoder rnn cell to use for aid in prediction
        of the next decoder output.  If feed_previous is True and this is False,
        the weights will not be altered when we do the output_projection except
        for the GO token's embedding weights.
    initial_state_attention: if you want initialize attentions (False).
    dtype: None defaults to tf.float32

    Return:
<<<<<<< HEAD
    N: batch_size
    H: num_hidden_units
=======
>>>>>>> b9142bdc5bb8c7718091544202fa6c1f1d987ffd
    outputs: [<max_len>, N, H]
    state: [N, H]

    """

    # Get the output dimension from the cell
    output_size = cell.output_size

    # Set up scope
    with tf.variable_scope(scope or "embedding_attention_decoder",
        dtype=dtype) as scope:

        # Embed the decoder inputs (which is a list)
<<<<<<< HEAD
#        W_embedding = tf.get_variable("W_embedding",
#            shape=[num_symbols, embedding_size])
#        embedded_decoder_inputs = [
#            tf.nn.embedding_lookup(W_embedding, i) for i in decoder_inputs]
=======
        W_embedding = tf.get_variable("W_embedding",
            shape=[num_symbols, embedding_size])
        embedded_decoder_inputs = [
            tf.nn.embedding_lookup(W_embedding, i) for i in decoder_inputs]
>>>>>>> b9142bdc5bb8c7718091544202fa6c1f1d987ffd

        # Loop function if using decoder outputs for next prediction
        loop_function = _extract_argmax_and_embed(
            W_embedding, output_projection,
            update_embedding_for_previous) if feed_previous else None

        return attention_decoder(
<<<<<<< HEAD
            decoder_inputs,
            initial_state,
            pre_attention_states,post_attention_states,kb_attention_states,
=======
            embedded_decoder_inputs,
            initial_state,
            attention_states,
>>>>>>> b9142bdc5bb8c7718091544202fa6c1f1d987ffd
            cell,
            output_size=output_size,
            num_heads=num_heads,
            loop_function=loop_function,
            initial_state_attention=initial_state_attention)

def rnn_softmax(FLAGS, outputs, scope):
    with tf.variable_scope(scope, reuse=True):
        W_softmax = tf.get_variable("W_softmax",
            [FLAGS.num_hidden_units, FLAGS.sp_vocab_size])
        b_softmax = tf.get_variable("b_softmax", [FLAGS.sp_vocab_size])

        logits = tf.matmul(outputs, W_softmax) + b_softmax
        return logits

class AttentionModel(object):

<<<<<<< HEAD
    def __init__(self,sess, FLAGS,forward_only,dropout):

        
        self.sess=sess
        self.batch_size =FLAGS.batch_size,
        self.max_sent_size = FLAGS.max_sent_size
        self.vocab_size = FLAGS.vocab_size
        self.emb_dim = FLAGS.hidden_size
        
        self.max_pre_size,self.max_kb_size,self.max_post_size = \
        FLAGS.max_pre_size,FLAGS.max_kb_size,FLAGS.max_post_size
        
        initializer = tf.random_uniform_initializer(-np.sqrt(3), np.sqrt(3))
        self.build_inputs()
        
        with tf.variable_scope("embedding") as scope:
            A = VariableEmbedder(FLAGS, initializer=initializer, name='Embedder')
            self.pre_emb = A(self.pre, name='embedded_pre')
            self.kb_emb = A(self.kb, name='embedded_kb')
            self.post_emb = A(self.post, name='embedded_post')
            self.query_emb = A(self.query, name='embedded_query')
            self.decoder_inputs_emb = A(self.decoder_inputs, name='embedded_response_inputs')
            self.decoder_targets_emb = A(self.decoder_targets, name='embedded_response_targets')
            
        with tf.name_scope("encoding") as scope:
            encoder = PositionEncoder(self.max_sent_size, self.emb_dim)
            self.pre_enc = encoder(self.pre_emb)
            self.kb_enc = encoder(self.kb_emb)
            self.post_enc = encoder(self.post_emb)
            self.query_enc = encoder(self.query_emb)
           
             
        self.pre_enc_seq_len = FLAGS.max_pre_size
        self.post_enc_seq_len = FLAGS.max_post_size
        self.kb_len = FLAGS.max_kb_size
        self.dec_seq_len = FLAGS.max_sent_size
        self.dropout = dropout

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
     
        with tf.variable_scope('pre_encoder') as scope:

            # PRE Encoder RNN cell
            self.pre_encoder_stacked_cell = rnn_cell(FLAGS, self.dropout,
                scope=scope)

            # Outputs from PRE encoder RNN
            self.pre_encoder_outputs, self.pre_encoder_state = tf.nn.dynamic_rnn(
                cell=self.pre_encoder_stacked_cell,
                inputs=self.pre_enc,
                sequence_length=self.pre_enc_seq_len, time_major=False,
                dtype=tf.float32)

        with tf.variable_scope('post_encoder') as scope:

            # POST Encoder RNN cell
            self.post_encoder_stacked_cell = rnn_cell(FLAGS, self.dropout,
                scope=scope)

            # Outputs from POST encoder RNN
            self.post_encoder_outputs, self.post_encoder_state = tf.nn.dynamic_rnn(
                cell=self.post_encoder_stacked_cell,
                inputs=self.post_enc,
                sequence_length=self.post_enc_seq_len, time_major=False,
                dtype=tf.float32)

            #PRE attention encoder states
        with tf.variable_scope('pre_attention') as scope:
            # Need attention states to be [batch_size, attn_length, attn_size]
            self.pre_attention_states = self.pre_encoder_outputs

            #POST attention encoder states
        with tf.variable_scope('post_attention') as scope:
            # Need attention states to be [batch_size, attn_length, attn_size]
            self.post_attention_states = self.post_encoder_outputs
=======
    def __init__(self, FLAGS, forward_only,encoder_inputs,decoder_inputs,targets,
                 enc_seq_len,dec_seq_len,dropout):

        # Placeholders
        self.encoder_inputs = encoder_inputs
        self.decoder_inputs = decoder_inputs
        self.targets = targets
        self.enc_seq_lens = enc_seq_len
        self.dec_seq_lens = dec_seq_len
        self.dropout = dropout

        with tf.variable_scope('encoder') as scope:

            # Encoder RNN cell
            self.encoder_stacked_cell = rnn_cell(FLAGS, self.dropout,
                scope=scope)

            # Embed encoder inputs
#            W_input = tf.get_variable("W_input",
#                [FLAGS.en_vocab_size, FLAGS.num_hidden_units])
#            self.embedded_encoder_inputs = rnn_inputs(FLAGS,
#                self.encoder_inputs, FLAGS.en_vocab_size, scope=scope)
            #initial_state = encoder_stacked_cell.zero_state(FLAGS.batch_size, tf.float32)

            # Outputs from encoder RNN
            self.all_encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(
                cell=self.encoder_stacked_cell,
                inputs=self.encoder_inputs,
                sequence_length=self.enc_seq_lens, time_major=False,
                dtype=tf.float32)


        with tf.variable_scope('attention') as scope:
            # Need attention states to be [batch_size, attn_length, attn_size]
            self.attention_states = self.all_encoder_outputs
>>>>>>> b9142bdc5bb8c7718091544202fa6c1f1d987ffd


        with tf.variable_scope('decoder') as scope:

            # Initial state is last relevant state from encoder
<<<<<<< HEAD
            self.decoder_initial_state = tf.concat([self.pre_encoder_state,
                                                   self.post_encoder_state],1)
            
            # Decoder RNN cell
            self.decoder_stacked_cell = rnn_cell(FLAGS, self.dropout,
                scope=scope,decoder_cell=True)

            # Need input to embedding_attention_decoder to be time-major
            self.decoder_inputs_time_major = tf.transpose(
                self.decoder_inputs_emb, [1, 0, 2])
=======
            self.decoder_initial_state = self.encoder_state

            # Decoder RNN cell
            self.decoder_stacked_cell = rnn_cell(FLAGS, self.dropout,
                scope=scope)

            # Need input to embedding_attention_decoder to be time-major
            self.decoder_inputs_time_major = tf.transpose(
                self.decoder_inputs, [1, 0])
>>>>>>> b9142bdc5bb8c7718091544202fa6c1f1d987ffd

            # make decoder inputs into a batch_size list of inputs
            self.list_decoder_inputs = tf.unpack(
                self.decoder_inputs_time_major, axis=0)

            # Output projection weights (for softmax and embedding predictions)
            W_softmax = tf.get_variable("W_softmax",
<<<<<<< HEAD
                shape=[FLAGS.decoder_num_hidden_units, FLAGS.vocab_size],
                dtype=tf.float32)
            b_softmax = tf.get_variable("b_softmax",
                shape=[FLAGS.vocab_size],
                dtype=tf.float32)
            output_projection = (W_softmax, b_softmax)

            W_emb_mat=A.get_emb_mat()
            self.all_decoder_outputs, self.decoder_state = embedding_attention_decoder(
                    decoder_inputs=self.list_decoder_inputs,
                    initial_state=self.decoder_initial_state,
                    pre_attention_states=self.pre_attention_states,
                    post_attention_states=self.post_attention_states,
                    kb_attention_states=self.kb_enc,
                    cell=self.decoder_stacked_cell,
                    num_symbols=FLAGS.vocab_size,
                    embedding_size=FLAGS.decoder_num_hidden_units,
                    output_projection=output_projection,
                    W_embedding=W_emb_mat,
=======
                shape=[FLAGS.num_hidden_units, FLAGS.sp_vocab_size],
                dtype=tf.float32)
            b_softmax = tf.get_variable("b_softmax",
                shape=[FLAGS.sp_vocab_size],
                dtype=tf.float32)
            output_projection = (W_softmax, b_softmax)

            self.all_decoder_outputs, self.decoder_state = \
                embedding_attention_decoder(
                    decoder_inputs=self.list_decoder_inputs,
                    initial_state=self.decoder_initial_state,
                    attention_states=self.attention_states,
                    cell=self.decoder_stacked_cell,
                    num_symbols=FLAGS.sp_vocab_size,
                    embedding_size=FLAGS.num_hidden_units,
                    output_projection=output_projection,
>>>>>>> b9142bdc5bb8c7718091544202fa6c1f1d987ffd
                    feed_previous=forward_only)

            # Logits
            self.decoder_outputs_flat = tf.reshape(self.all_decoder_outputs,
                [-1, FLAGS.num_hidden_units])
            self.logits_flat = rnn_softmax(FLAGS, self.decoder_outputs_flat,
                scope=scope)

            # Loss with masking
            targets_flat = tf.reshape(self.targets, [-1])
            losses_flat = tf.nn.sparse_softmax_cross_entropy_with_logits(
                self.logits_flat, targets_flat)
            mask = tf.sign(tf.to_float(targets_flat))
            masked_losses = mask * losses_flat
            masked_losses = tf.reshape(masked_losses,  tf.shape(self.targets))
            self.loss = tf.reduce_mean(
                tf.reduce_sum(masked_losses, reduction_indices=1))

        # Optimization
        self.lr = tf.Variable(0.0, trainable=False)
        trainable_vars = tf.trainable_variables()
        # clip the gradient to avoid vanishing or blowing up gradients
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.loss, trainable_vars), FLAGS.max_gradient_norm)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_optimizer = optimizer.apply_gradients(
            zip(grads, trainable_vars))

        # For printng results from training
        self.logits = tf.reshape(self.logits_flat, [FLAGS.batch_size,
            FLAGS.sp_max_len, FLAGS.sp_vocab_size])
        self.y_pred = tf.argmax(self.logits, 2)

        # Save all the variables
        self.global_step = tf.Variable(0, trainable=False)
        self.saver = tf.train.Saver(tf.all_variables())


<<<<<<< HEAD
    def build_inputs(self):
        
        self.pre = tf.placeholder(tf.int64,[None,self.max_pre_size,self.max_sent_size], name="PRE")
        self.kb = tf.placeholder(tf.int64,[None,self.max_kb_size,3], name="KB")
        self.post = tf.placeholder(tf.int64,[None,self.max_post_size,self.max_sent_size], name="POST")
        self.query = tf.placeholder(tf.int64,[None,self.max_sent_size], name="QUERY")
        self.decoder_inputs = tf.placeholder(tf.int64,[None,self.max_sent_size], name = "RESPONSE_inputs")
        self.decoder_targets = tf.placeholder(tf.int64,[None,self.max_sent_size], name = "RESPONSE_targets")
    
=======
>>>>>>> b9142bdc5bb8c7718091544202fa6c1f1d987ffd
    def step(self, sess, FLAGS, batch_encoder_inputs, batch_decoder_inputs,
        batch_targets, batch_enc_seq_lens, batch_dec_seq_lens, dropout,
        forward_only, sampling=False):

        if not forward_only and not sampling:
            input_feed = {
                self.encoder_inputs: batch_encoder_inputs,
                self.decoder_inputs: batch_decoder_inputs,
                self.targets: batch_targets,
                self.enc_seq_lens: batch_enc_seq_lens,
                self.dec_seq_lens: batch_dec_seq_lens,
                self.dropout: dropout}
            #output_feed = [self.loss, self.train_optimizer]
            output_feed = [self.y_pred, self.loss, self.train_optimizer]
            outputs = sess.run(output_feed, input_feed)
            return outputs[0], outputs[1], outputs[2]
        elif forward_only and not sampling:
            input_feed = {
                self.encoder_inputs: batch_encoder_inputs,
                self.decoder_inputs: batch_decoder_inputs,
                self.targets: batch_targets,
                self.enc_seq_lens: batch_enc_seq_lens,
                self.dec_seq_lens: batch_dec_seq_lens,
                self.dropout: dropout}
            output_feed = [self.loss]
            outputs = sess.run(output_feed, input_feed)
            return outputs[0]
        elif forward_only and sampling:
            input_feed = {
                self.encoder_inputs: batch_encoder_inputs,
                self.decoder_inputs: batch_decoder_inputs,
                self.enc_seq_lens: batch_enc_seq_lens,
                self.dropout: dropout}
            output_feed = [self.y_pred]
            outputs = sess.run(output_feed, input_feed)
            return outputs[0]
<<<<<<< HEAD
     
  
=======





>>>>>>> b9142bdc5bb8c7718091544202fa6c1f1d987ffd
