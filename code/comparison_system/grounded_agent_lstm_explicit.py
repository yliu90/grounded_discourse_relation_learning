import tensorflow as tf
from datetime import datetime
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np

import random
import math
import os
import time

from collections import OrderedDict

import copy
import json
from data_construction import *
from event_occurs import *



class TemporalCluesNetwork(object):

    def __init__(self, config):

        # a dict object config contains all hyper-parameters

        self.state_dim = config['state_dim']
        self.word_dim = config['word_dim']

        self.word_vocab_size = config['word_vocab_size']
        self.grounded_vocab_size = config['grounded_vocab_size']
        self.inner_dim = config['inner_dim']

        self.num_steps = config['num_steps']
        self.sess = config['sess']
        self.batch_size = config['batch_size']
        self.rel_dim = config['rel_dim']

        self.f_mtrx = config['f_mtrx'] # following matrix
        self.m_mtrx = config['m_mtrx'] # meanwhile matrix
        self.p_mtrx = config['p_mtrx'] # previous matrix

        self.temporal_dim = config['temporal_dim']

        self.seen_event_memory = tf.constant(config['seen_event_memory'])
        self.seen_event_lens = tf.constant(config['seen_event_lens'])

        self.arg1 = tf.placeholder(tf.int32, [None, self.num_steps])
        self.arg2 = tf.placeholder(tf.int32, [None, self.num_steps])

        self.arg1_len = tf.placeholder(tf.int32, [None])
        self.arg2_len = tf.placeholder(tf.int32, [None])

        self.true_rel = tf.placeholder(tf.int32, [None])

        self.word_embedding = tf.get_variable("embedding", [self.word_vocab_size, self.word_dim])
        self.arg1_emb = tf.nn.embedding_lookup(self.word_embedding, self.arg1)
        self.arg2_emb = tf.nn.embedding_lookup(self.word_embedding, self.arg2)

        with tf.variable_scope('lstm') as scope:
            # 
            cell_fw = tf.contrib.rnn.LSTMCell(self.inner_dim, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
            # logic memory part
            self._initial_state = cell_fw.zero_state(tf.shape(self.arg1)[0], tf.float32)
            # arg1 part
            self.arg1_output, self.arg1_state = tf.nn.dynamic_rnn(cell=cell_fw, \
                    inputs=self.arg1_emb, \
                    sequence_length=self.arg1_len, \
                    initial_state=self._initial_state)
            scope.reuse_variables()
            # arg2 part
            self.arg2_output, self.arg2_state = tf.nn.dynamic_rnn(cell=cell_fw, \
                    inputs=self.arg2_emb, \
                    sequence_length=self.arg2_len,\
                    initial_state=self._initial_state)

        # now we 
        self.arg1_lo = self.extract_axis_1(self.arg1_output, self.arg1_len-1)
        self.arg2_lo = self.extract_axis_1(self.arg2_output, self.arg2_len-1)
        
        # related event 
        #  
        self.eA = tf.placeholder(tf.int32, [None, self.num_steps])
        self.eB = tf.placeholder(tf.int32, [None, self.num_steps]) 
        self.eA_len = tf.placeholder(tf.int32, [None])
        self.eB_len = tf.placeholder(tf.int32, [None])

        self.grounded_embedding = tf.get_variable("grounded_embedding", [self.grounded_vocab_size, self.word_dim])

        self.seen_event_memory_emb = tf.nn.embedding_lookup(self.grounded_embedding, self.seen_event_memory)
        with tf.variable_scope('event_lstm') as scope:
            # 
            event_cell_fw = tf.contrib.rnn.LSTMCell(self.inner_dim, state_is_tuple=True, reuse=tf.get_variable_scope().reuse)
            # logic memory part

            self._initial_state_seen_event = event_cell_fw.zero_state(tf.shape(self.seen_event_memory)[0], tf.float32)
            self.seen_event_output, self.seen_event_state = tf.nn.dynamic_rnn(cell=event_cell_fw, \
                    inputs=self.seen_event_memory_emb, \
                    sequence_length=self.seen_event_lens,\
                    initial_state=self._initial_state_seen_event)

            pass

        self.eA_emb = tf.nn.embedding_lookup(self.grounded_embedding, self.eA)
        self.eB_emb = tf.nn.embedding_lookup(self.grounded_embedding, self.eB)

        with tf.variable_scope('event_lstm') as scope:

            scope.reuse_variables()
            self._initial_state_event = event_cell_fw.zero_state(tf.shape(self.eA)[0], tf.float32)
            
            # eA part
            self.eA_output, self.eA_state = tf.nn.dynamic_rnn(cell=event_cell_fw, \
                    inputs=self.eA_emb, \
                    sequence_length=self.eA_len, \
                    initial_state=self._initial_state_event)
            
            # eB part
            self.eB_output, self.eB_state = tf.nn.dynamic_rnn(cell=event_cell_fw, \
                    inputs=self.eB_emb, \
                    sequence_length=self.eB_len, \
                    initial_state=self._initial_state_event)


        # using encoding representation to access the temporal clues matrix
        # is that necessary?
        # loaded in temporal clues matrix
        # use three memory matrix and a mapping memory to build up the temporal memory
        # fact no need to learn
        seen_events = None
        # get correlation score
        # given event eB seen_event difference

        # eA => batch, eA_representation => eA_match_idx
        # eB => batch, eB_representation => eB_match_idx
        self.eA_lo = self.extract_axis_1(self.eA_output, self.eA_len-1)
        self.eB_lo = self.extract_axis_1(self.eB_output, self.eB_len-1)

        # net 
        self.seen_event_memory_lo = self.extract_axis_1(self.seen_event_output, self.seen_event_lens-1)

        # search for simility memory event
        self.seen_event_memory_nor = tf.nn.l2_normalize(self.seen_event_memory_lo, dim=1)
        self.eA_nor = tf.nn.l2_normalize(self.eA_lo, dim=1)
        self.eB_nor = tf.nn.l2_normalize(self.eB_lo, dim=1)
        self.eA_cs = tf.matmul(self.eA_nor, tf.transpose(self.seen_event_memory_nor, [1,0]))
        self.eB_cs = tf.matmul(self.eB_nor, tf.transpose(self.seen_event_memory_nor, [1,0]))
        # 
        self.eA_idx = tf.argmax(self.eA_cs, 1)
        self.eB_idx = tf.argmax(self.eB_cs, 1)
        #

        # difference eA_representation between memory matched event
        # batch_size , inner_dim 
        self.eA_diff =  self.eA_lo - tf.gather(self.seen_event_memory_lo, self.eA_idx) # difference
        self.eB_diff =  self.eB_lo - tf.gather(self.seen_event_memory_lo, self.eB_idx) # 

        # access temporal memory feature eA_index&eB_index
        # 
        A_B_index = tf.stack([self.eA_idx, self.eB_idx], axis=1)
        B_A_index = tf.stack([self.eB_idx, self.eA_idx], axis=1)

        # generate temporal clues
        # access to self.f_mtrx; self.m_mtrx; self.p_mtrx

        self.f_A_B = tf.gather_nd(self.f_mtrx, A_B_index)
        self.f_B_A = tf.gather_nd(self.f_mtrx, B_A_index)
        self.m_A_B = tf.gather_nd(self.m_mtrx, A_B_index)
        self.m_B_A = tf.gather_nd(self.m_mtrx, B_A_index)
        self.p_A_B = tf.gather_nd(self.p_mtrx, A_B_index)
        self.p_B_A = tf.gather_nd(self.p_mtrx, B_A_index)

        self.temporal_clues = tf.stack([self.f_A_B, self.f_B_A, self.m_A_B, self.m_B_A, self.p_A_B, self.p_B_A], axis=1)

        # self.arg1_lo self.arg2_lo / self.temporal_clues
        # expand the temporal clues
        self.W_temporal, self.b_temporal = self._fc_variable([self.temporal_dim, self.inner_dim])
        self.temporal_feat = tf.tanh(tf.matmul(self.temporal_clues, self.W_temporal) + self.b_temporal)

        # 
        self.W_sen, self.b_sen = self._fc_variable([self.inner_dim, self.inner_dim/2])
        self.arg1_feat = tf.tanh(tf.matmul(self.arg1_lo, self.W_sen)+self.b_sen)
        self.arg2_feat = tf.tanh(tf.matmul(self.arg2_lo, self.W_sen)+self.b_sen)
        self.feat_ = tf.concat([self.arg1_feat, self.arg2_feat],axis=1) # 

        self.W_e_diff, self.b_e_diff = self._fc_variable([self.inner_dim, self.inner_dim/2])
        self.eA_diff_feat = tf.tanh(tf.matmul(self.eA_diff, self.W_e_diff)+self.b_e_diff)
        self.eB_diff_feat = tf.tanh(tf.matmul(self.eB_diff, self.W_e_diff)+self.b_e_diff)
        self.feat_diff = tf.concat([self.eA_diff_feat, self.eB_diff_feat],axis=1) # 

        # after two full-connected layers 
        self.sen_temporal_feat = tf.concat([self.feat_, self.temporal_feat, self.feat_diff], axis=1)

        self.W_f_1, self.b_f_1 = self._fc_variable([((self.inner_dim/2)*2)*2+self.inner_dim, self.rel_dim])
        # self.W_f_2, self.b_f_2 = self._fc_variable([self.inner_dim, self.rel_dim])

        # self.tmp_feat = tf.tanh(tf.matmul(self.sen_temporal_feat, self.W_f_1) + self.b_f_1)
        logit = tf.tanh(tf.matmul(self.sen_temporal_feat, self.W_f_1) + self.b_f_1)
        # logit = tf.tanh(tf.matmul(self.tmp_feat, self.W_f_2) + self.b_f_2)

        # self.W_diff, self.b_diff = self._fc_variable([self.inner_dim, self.inner_dim/2])
        # self.diff_feat_A = tf.tanh(tf.matmul(self.eA_diff, self.W_diff) + self.b_diff)
        # self.diff_feat_B = tf.tanh(tf.matmul(self.eB_diff, self.W_diff) + self.b_diff)

        # self.temporal_clues concatenate with self.inner_feat_2

        self.predict_score = tf.nn.softmax(logit)
        self.max_predict_score = tf.argmax(self.predict_score, axis=1) # 

        # cosine matching
        self.total_loss = tf.losses.sparse_softmax_cross_entropy(self.true_rel, logit)
        
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = optimizer.compute_gradients(self.total_loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        self.tf_init() 

    def tf_init(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)
        pass

    def extract_axis_1(self, data, idx):
        batch_range = tf.range(tf.shape(data)[0]) # batch
        indices = tf.stack([batch_range, idx], axis=1)
        res = tf.gather_nd(data, indices)
        return res
    
    def _fc_variable(self, weight_shape):
        input_channels  = weight_shape[0]
        output_channels = weight_shape[1]
        d = 1.0 / np.sqrt(input_channels)
        bias_shape = [output_channels]
        weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
        bias   = tf.Variable(tf.random_uniform(bias_shape,   minval=-d, maxval=d))
        return weight, bias

    def update(self, batch):
        _, tloss = self.sess.run([self.train_op, self.total_loss],
                                    feed_dict = {
                                        self.arg1:batch[0],
                                        self.arg1_len:batch[1],
                                        self.arg2:batch[2],
                                        self.arg2_len:batch[3],
                                        self.eA:batch[4],
                                        self.eA_len:batch[5],
                                        self.eB:batch[6],
                                        self.eB_len:batch[7],
                                        self.true_rel:batch[8]})
        
        return tloss
    
    def retrieval_match_event(self, batch):
        pred_ = self.sess.run([self.temporal_clues],
                                    feed_dict = {
                                        self.arg1:batch[0],
                                        self.arg1_len:batch[1],
                                        self.arg2:batch[2],
                                        self.arg2_len:batch[3],
                                        self.eA:batch[4],
                                        self.eA_len:batch[5],
                                        self.eB:batch[6],
                                        self.eB_len:batch[7]
                                        })
        return pred_

    def predict(self, batch):
        pred_ = self.sess.run([self.predict_score],
                                    feed_dict = {
                                        self.arg1:batch[0],
                                        self.arg1_len:batch[1],
                                        self.arg2:batch[2],
                                        self.arg2_len:batch[3],
                                        self.eA:batch[4],
                                        self.eA_len:batch[5],
                                        self.eB:batch[6],
                                        self.eB_len:batch[7]}
                                    
                                    )
        return pred_

    def max_predict(self, batch):
        pred_ = self.sess.run([self.max_predict_score],
                                    feed_dict = {
                                        self.arg1:batch[0],
                                        self.arg1_len:batch[1],
                                        self.arg2:batch[2],
                                        self.arg2_len:batch[3],
                                        self.eA:batch[4],
                                        self.eA_len:batch[5],
                                        self.eB:batch[6],
                                        self.eB_len:batch[7]
                                        })

        return pred_


def divides(batchs, divs):
    total = sum(divs)
    div_ratios = [float(w)/float(total) for w in divs]
    tmp = 0
    sum_ratios = []
    # sum_ratios.append(tmp)
    for i in div_ratios:
        tmp += i
        sum_ratios.append(tmp)
    sum_ratios[-1] = 1
    total_sum = len(batchs)
    seps = [int(float(total_sum)*float(w)) for w in sum_ratios]
    
    train = batchs[:seps[0]]
    dev = batchs[seps[0]:seps[1]]
    tst = batchs[seps[1]:]

    return train, dev, tst


def early_stop(acc_history,patient=300):

    stopflag = False
    # patient = 300
    dev_history = []

    for t in acc_history:
        dev = t['dev']
        dev_history.append(dev)

    maxdev = max(dev_history)
    best_tuple = OrderedDict()
    index = len(dev_history)-1
    for i,dev in enumerate(reversed(dev_history)):
        if dev != maxdev:
            index -= 1
        else:
            best_tuple = acc_history[index]
            break

    if len(dev_history) - index > patient:
        stopflag = True

    return stopflag, best_tuple




def train_model(model, batchs, index_to_rel, divs):

    # divides the train dev test for 8/1/1
    print 'total number of batchs samples ... '
    print len(batchs)

    if divs is None:
        divs = [6,2,2]
    train, dev, tst = divides(batchs,divs)

    print 'The size of corpora including train, dev, test ... '
    print 'The size of train : ', len(train)
    print 'The size of dev : ' , len(dev)
    print 'The size of tst : ' , len(tst)

    max_epoch = 1000

    acc_history = []
    train_cache = []
    test_cache = []
    dev_cache = []

    for epoch in range(max_epoch):

        # 
        print '#########################################3'
        print '#########################################3'
        print '#########################################3'

        print 'now Epochs-n : ' , epoch , ""

        total_pred = []
        true_tags = []

        for i,batch in enumerate(train):

            batch_pred = model.max_predict(batch)
            total_pred.extend(batch_pred[0])
            true_tags.extend(batch[8])

            print '++++++'
            print batch[8]
            batch_loss = model.update(batch)

            print 'batch update : ', batch_loss

            print '############################'
            # training control

            print 'Epoch-i-th ' , epoch , ' :'
            print 'Batch ', i , ':'

            print 'Epoch batch test function : '

            # print model.retrieval_match_event(batch)[1].shape
            # print model.retrieval_match_event(batch)[0].shape

            print 'Epoch batch train acc ' , compute_acc(true_tags, total_pred)
            trn_acc, trn_pred, trn_tags = compute_dataset_acc(model, train, index_to_rel)
            print 'Epoch train acc : ' , trn_acc
            
            dev_acc, dev_pred, dev_tags = compute_dataset_acc(model, dev, index_to_rel)
            print 'Epoch dev acc : ' , dev_acc

            tst_acc, tst_pred, tst_tags = compute_dataset_acc(model, tst, index_to_rel)
            print 'Epoch tst acc : ' , tst_acc

            if i%10 == 0:
                print "Dev discourse relation mapping ... "
                relation_acc(dev_tags, dev_pred, index_to_rel)

                print "Tst discourse relation mapping ... "
                relation_acc(tst_tags, tst_pred, index_to_rel)

                print "Train discourse relation mapping ... "
                relation_acc(trn_tags, trn_pred, index_to_rel)


            acc_tuple = OrderedDict()
            acc_tuple['epoch'] = epoch
            acc_tuple['batch'] = i
            acc_tuple['trn'] = trn_acc
            acc_tuple['dev'] = dev_acc
            acc_tuple['tst'] = tst_acc

            acc_history.append(acc_tuple)
            stopflag, best_tuple = early_stop(acc_history, 400)

            if stopflag:
                break

        if stopflag:
            print 'End of training ... '
            print 'Best Result : '
            print best_tuple
            break

        pass

    pass

def compute_acc(true_tag,pred_tag):
    match_count = 0
    for t,p in zip(true_tag,pred_tag):
        if t == p:
            match_count += 1
    return float(match_count)/float(len(true_tag))

def relation_acc(true_tags, total_pred, index_to_rel):
    # each relation F1 recall/precesion
    # relation confuse matrix
    # 

    # investigate the label 
    tag_axis = set(true_tags)

    F1_dict = OrderedDict()
    
    for tag in tag_axis:

        # compute recall and Precision for each class 
        # consider other class all as negative class 
        pred_cache = copy.deepcopy(total_pred)
        true_cache = copy.deepcopy(true_tags)
        tag_crt_count = 0
        for t,p in zip(true_cache, pred_cache):
            if p != tag:
                p = -1
            if t != tag:
                t = -1
            if t == p and t == tag:
                tag_crt_count += 1

        if tag_crt_count != 0:
            # now recall
            true_counts = true_cache.count(tag)
            if true_counts != 0:
                recall = float(tag_crt_count) / float(true_counts)
            else:
                recall = 0
            
            # precesion
            pred_tags = pred_cache.count(tag)
            if pred_tags != 0:
                prec = float(tag_crt_count) / float(pred_tags)
            else:
                prec = 0

            if (recall + prec) != 0:
                f1 = (2*recall*prec)/(recall+prec)
            else:
                f1 = 0
        else:
            recall = 0
            prec = 0
            f1 = 0

        tagF1 = OrderedDict()
        tagF1['recall'] = recall
        tagF1['precesion'] = prec
        tagF1['F1measure'] = f1
        F1_dict[index_to_rel[int(tag)]] = tagF1
        pass

    # metric matrix 
    print 'Metric Matrix '
    print index_to_rel
    metric_matrix = np.zeros((len(tag_axis),3))

    for rel in F1_dict:
        print '*******'
        print rel
        print F1_dict[rel]

    pred_cache = copy.deepcopy(total_pred)
    true_cache = copy.deepcopy(true_tags)
    rel_distr = OrderedDict()
    for i,t in enumerate(true_cache):
        # initial
        if t not in rel_distr:
            rel_distr[t] = OrderedDict()
        if pred_cache[i] not in rel_distr[t]:
            rel_distr[t][pred_cache[i]] = 0
        rel_distr[t][pred_cache[i]] += 1

    print 'Relation mapping ... :'
    for key in rel_distr:
        print '******'
        print key, ' : ' , index_to_rel[key]
        print rel_distr[key]

    init_matrix = np.zeros((len(tag_axis),len(tag_axis)))
    for key in rel_distr:
        for j in rel_distr[key]:
            init_matrix[key][j] = rel_distr[key][j]
            pass
    print init_matrix
    pass

def compute_dataset_acc(model, dataset, index_to_rel):
    total_pred = []
    true_tags = []
    i = 0
    for i,batch in enumerate(dataset):

        if len(batch[0]) == 0:
            continue

        batch_pred = model.max_predict(batch)

        total_pred.extend(batch_pred[0])
        true_tags.extend(batch[8])

    match_count = 0
    for t,p in zip(true_tags, total_pred):
        if t == p:
            match_count += 1
    # relation_acc(true_tags, total_pred, index_to_rel)
    
    return float(match_count)/float(len(true_tags)), total_pred, true_tags

def filter_data(datas):

    records = []
    for d in datas:
        records.append(d.split('\t')[0])
    items = set(records)
    for i in items:
        print i, ':' , records.count(i)
    pass

def build_temporal_memory(correl_path):

    cf = open(correl_path,'r').readline()
    event_correl = json.loads(cf)

    # print event_correl
    # event_correl is organzied by event , each event object keeps the three types temporal 
    # correlation memory: meanwhile , prev, following , each score in the indicate the oc-curance
    # between 0~1 we divide it in three matrix following , meanwhile and prev 
    # for example in the 'following' matrix [eA][eB] means 
    # Event memory is very strict, but associative behavior can connect different events
    # by perceiving synaesthesia. Therefore, we build up a seen event mapping, and we 
    # break each event into grounded item sequence
    # 
    seen_event = [e.strip().split('+') for e in event_correl.keys()]
    seen_event_lens = [len(e) for e in seen_event]

    index_to_event = event_correl.keys()
    event_to_index = OrderedDict()
    for i,e in enumerate(index_to_event):
        event_to_index[e] = i

    following_matrix = np.zeros((len(event_correl.keys()),len(event_correl.keys())))
    previous_matrix = np.zeros((len(event_correl.keys()),len(event_correl.keys())))
    meanwhile_matrix = np.zeros((len(event_correl.keys()),len(event_correl.keys())))

    # temporal_clues_feature => eA_count_norm, A_prev_B, A_meanwhile_B, A_following_B, eB_count_norm
    # B_prev_A, A_meanwhile_B, B_following_A

    event_list = event_correl.keys()
    
    for i in range(len(event_list)):
        e_i = event_list[i]
        for e_j in event_list[i:]:

            A_norm, A_prev_B, A_meanwhile_B, A_following_B, B_norm, B_prev_A, B_meanwhile_A, B_following_A = temporal_clues_feature(event_correl, e_i, e_j)
            # following matrix
            following_matrix[event_to_index[e_i]][event_to_index[e_j]] = A_following_B
            following_matrix[event_to_index[e_j]][event_to_index[e_i]] = B_following_A
            # prev matrix
            previous_matrix[event_to_index[e_i]][event_to_index[e_j]] = A_prev_B
            previous_matrix[event_to_index[e_j]][event_to_index[e_i]] = B_prev_A
            # meanwhile
            meanwhile_matrix[event_to_index[e_i]][event_to_index[e_j]] = A_meanwhile_B
            meanwhile_matrix[event_to_index[e_j]][event_to_index[e_i]] = B_meanwhile_A

    return following_matrix, previous_matrix, meanwhile_matrix, seen_event, seen_event_lens, index_to_event, event_to_index

def get_temporal_memory(correl_path, correl_cache_path):

    if os.path.isfile(correl_cache_path):
        # 
        print 'Use temporal memory cache ... '
        f_mtrx, p_mtrx, m_mtrx, s_e, s_e_l, i_t_e, e_t_i = json.loads(open(correl_cache_path,'r').readline())
        f_mtrx = np.asarray(f_mtrx)
        p_mtrx = np.asarray(p_mtrx)
        m_mtrx = np.asarray(m_mtrx)
        return f_mtrx, p_mtrx, m_mtrx, s_e, s_e_l, i_t_e, e_t_i
    else:
        print 'Not found cache temporal memory ... building ... '
        f_mtrx, p_mtrx, m_mtrx, s_e, s_e_l, i_t_e, e_t_i = build_temporal_memory(correl_path)
        jstr = json.dumps([f_mtrx.tolist(), p_mtrx.tolist(), m_mtrx.tolist(), s_e, s_e_l, i_t_e, e_t_i])
        f = open(correl_cache_path,'w')
        f.write(jstr)
        f.close()
        return f_mtrx, p_mtrx, m_mtrx, s_e, s_e_l, i_t_e, e_t_i

def process():

    ###################################
    # part 1 : data preparation
    # setup 
    ###################################
    data_path = "../../data/explicits"
    batch_size = 64
    
    start = time.time()
    # search for cache , build it when there is none
    correl_path = "../../data/event_correl"
    correl_cache_path = "../../data/event_correl_cache_matrix" 
    f_mtrx, p_mtrx, m_mtrx, seen_event, seen_event_lens, index_to_event, event_to_index = get_temporal_memory(correl_path,  correl_cache_path)


    # transfer the seen_event into grounded object index
    index_to_grounded, grounded_to_index = build_common_vocab(seen_event)

    # build index datas
    index_seen_event, maxlen_event = build_common_index_datas(seen_event, grounded_to_index, None)

    # build whole memory
    whole_memory = np.zeros((len(index_seen_event), maxlen_event))
    seen_event_lens = []
    for i,e in enumerate(index_seen_event):
        whole_memory[i,:len(e)] = e[:len(e)]
        seen_event_lens.append(len(e))
        pass

    end = time.time()
    print end-start

    # build up word embedding (word<->index)
    # however, in this section, the grounded information is not complete
    datas, index_to_word, word_to_index, _, _, index_to_rel, rel_to_index = build_data_vocab(data_path)
    print rel_to_index
    print index_to_rel

    # filter the data to get balance relation ratio
    filter_data(datas)
    # 
    index_datas = build_grounded_index_data(datas, word_to_index, grounded_to_index, rel_to_index)
    # build up padding and batch data
    batchs, maxlen = build_batch_grounded_data(index_datas, batch_size)

    #####################################
    # part 2 : model construction
    # setup the configuration
    #####################################
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    word_vocab_size = len(index_to_word)
    grounded_vocab_size = len(index_to_grounded)

    
    config = OrderedDict()
    config['sess'] = sess
    config['batch_size'] = 64
    config['word_vocab_size'] = len(index_to_word)
    config['grounded_vocab_size'] = len(index_to_grounded)

    print config['word_vocab_size']
    print config['grounded_vocab_size']
    config['word_dim'] = 150
    config['grounded_dim'] = 50
    config['state_dim'] = 50
    config['num_steps'] = maxlen
    config['inner_dim'] = 50
    config['rel_dim'] = len(rel_to_index)

    config['f_mtrx'] = np.asarray(f_mtrx, dtype=np.float32)
    config['p_mtrx'] = np.asarray(p_mtrx, dtype=np.float32)
    config['m_mtrx'] = np.asarray(m_mtrx, dtype=np.float32)
    config['temporal_dim'] = 6

    config['seen_event_memory'] = np.asarray(whole_memory, dtype=np.int32)
    print np.shape(config['seen_event_memory'])
    config['seen_event_lens'] = seen_event_lens


    model = TemporalCluesNetwork(config)

    ####################################
    # part 3 : model training
    ####################################

    divs = [6,2,2]
    train_model(model, batchs, index_to_rel, divs)


if __name__ == "__main__":

    process()

    # using early stop

    # train dev test
