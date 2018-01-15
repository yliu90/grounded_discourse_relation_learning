import tensorflow as tf
import os
import sys

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



class NaiveLstmNetwork(object):

    def __init__(self, sess, word_vocab_size, word_dim, state_dim, inner_dim, num_steps, batch_size, rel_dim, grad_applier):

        self.state_dim = state_dim
        self.word_dim = word_dim

        self.word_vocab_size = word_vocab_size
        self.inner_dim = inner_dim

        self.num_steps = num_steps
        self.sess = sess
        self.batch_size = batch_size
        self.rel_dim = rel_dim

        self.arg1 = tf.placeholder(tf.int32, [None, self.num_steps])
        self.arg2 = tf.placeholder(tf.int32, [None, self.num_steps])

        self.arg1_len = tf.placeholder(tf.int32, [None])
        self.arg2_len = tf.placeholder(tf.int32, [None])

        self.true_rel = tf.placeholder(tf.int32, [None])

        self.word_embedding = tf.get_variable("embedding", [self.word_vocab_size, self.word_dim])
        self.arg1_emb = tf.nn.embedding_lookup(self.word_embedding, self.arg1)
        self.arg2_emb = tf.nn.embedding_lookup(self.word_embedding, self.arg2)

        # pooling operation ... # capture the feature clues from word embedding ...
        # use max mean sum three different options for pooling tecnology

        # use tf slices to do this
        self.arg1_mask = tf.sequence_mask(self.arg1_len, self.num_steps, dtype=tf.float32) 
        self.arg2_mask = tf.sequence_mask(self.arg2_len, self.num_steps, dtype=tf.float32)

        arg1msk_shape = tf.shape(self.arg1_mask)
        arg2msk_shape = tf.shape(self.arg2_mask)

        self.arg1_mask = tf.reshape(self.arg1_mask, [arg1msk_shape[0], arg1msk_shape[1], 1])
        self.arg2_mask = tf.reshape(self.arg2_mask, [arg2msk_shape[0], arg2msk_shape[1], 1])

        self.arg1_lo = tf.reduce_sum(self.arg1_emb*self.arg1_mask, axis=1)
        self.arg2_lo = tf.reduce_sum(self.arg2_emb*self.arg2_mask, axis=1)
        
        # now we 
        self.feat_ = tf.concat([self.arg1_lo, self.arg2_lo],axis=1) # 

        # after two full-connected layers 
        self.W_1, self.b_1 = self._fc_variable([self.word_dim*2, self.word_dim]) # 
        self.inner_feat = tf.tanh(tf.matmul(self.feat_,self.W_1)+self.b_1)

        self.W_3, self.b_3 = self._fc_variable([self.word_dim*1, self.rel_dim])
        logit = tf.matmul(self.inner_feat, self.W_3) + self.b_3

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
                                        self.true_rel:batch[4]})
        
        return tloss

    def predict(self, batch):
        pred_ = self.sess.run([self.predict_score],
                                    feed_dict = {
                                        self.arg1:batch[0],
                                        self.arg1_len:batch[1],
                                        self.arg2:batch[2],
                                        self.arg2_len:batch[3]})
        return pred_

    def max_predict(self, batch):
        pred_ = self.sess.run([self.max_predict_score],
                                    feed_dict = {
                                        self.arg1:batch[0],
                                        self.arg1_len:batch[1],
                                        self.arg2:batch[2],
                                        self.arg2_len:batch[3]})

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

    # default
    if divs is None:
        divs = [6,2,2]
    train, dev, tst = divides(batchs,divs)

    print 'The size of corpora including train, dev, test ... '
    print len(train)
    print len(dev)
    print len(tst)

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
            true_tags.extend(batch[4])
            batch_loss = model.update(batch)

            print 'batch update : ', batch_loss

            print '############################'
            # training control

            print 'Epoch-i-th ' , epoch , ' :'
            print 'Batch ', i , ':'
            print 'Epoch batch train acc ' , compute_acc(true_tags, total_pred)
            trn_acc, trn_pred, trn_tags = compute_dataset_acc(model, train, index_to_rel)
            print 'Epoch train acc : ' , trn_acc
            dev_acc, dev_pred, dev_tags = compute_dataset_acc(model, dev, index_to_rel)
            print 'Epoch dev acc : ' , dev_acc
            tst_acc, tst_pred, tst_tags = compute_dataset_acc(model, tst, index_to_rel)
            print 'Epoch tst acc : ' , tst_acc

            if i%10 == 0:
                print "trn discourse relation mapping .. "
                relation_acc(trn_tags, trn_pred, index_to_rel)

                print "dev discourse relation mapping .. "
                relation_acc(dev_tags, dev_pred, index_to_rel)

                print "tst discourse relation mapping .. "
                relation_acc(tst_tags, tst_pred, index_to_rel)

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
        true_tags.extend(batch[4])

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

def process():

    ###################################
    # part 1 : data preparation
    # setup 
    ###################################
    data_path = "../../data/implicits"
    batch_size = 64

    # build up word embedding (word<->index)
    datas, index_to_word, word_to_index, index_to_grounded, grounded_to_index, index_to_rel, rel_to_index = build_data_vocab(data_path)

    # filter the data to get balance relation ratio
    filter_data(datas)
    # 
    index_datas = build_index_data(datas, word_to_index, rel_to_index)
    # build up padding and batch data
    batchs, maxlen = build_batch_data(index_datas, batch_size)

    #####################################
    # part 2 : model construction
    # setup the configuration
    #####################################
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    word_vocab_size = len(index_to_word)
    word_dim = 150
    state_dim = 100
    num_steps = maxlen
    inner_dim = 100
    rel_dim = len(rel_to_index)

    model = NaiveLstmNetwork(sess, word_vocab_size, word_dim, state_dim, inner_dim, num_steps, batch_size, rel_dim, None)


    # part 3 : model training
    #
    divs = [6,2,2]
    train_model(model, batchs, index_to_rel, divs)


if __name__ == "__main__":





    process()

    # using early stop

    # train dev test
