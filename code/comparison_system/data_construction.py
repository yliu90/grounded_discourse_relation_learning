import numpy as np
import os
import sys

import random
from collections import OrderedDict


def build_common_index_datas(datas, grounded_to_index, maxlen=None):

    if maxlen is None:
        lens = []
        for d in datas:
            lens.append(len(d))
        pass
    maxlen = max(lens)
    index_datas = []
    for d in datas:
        index_datas.append([grounded_to_index[g] for g in d])
    return index_datas, maxlen


def build_common_vocab(datas):

    # input as normal datas
    # list of symbol sequence
    index_to_grounded = None
    grounded_to_index = None
    grounded_set = []
    for d in datas:
        grounded_set.extend(d)
        pass
    grounded_set.append('unk')
    grounded_dict = OrderedDict()
    for g in grounded_set:
        if g in grounded_dict:
            grounded_dict[g] += 1
        else:
            grounded_dict[g] = 1
    index_to_grounded = [g for g in grounded_dict]
    grounded_to_index = OrderedDict()
    for i,g in enumerate(index_to_grounded):
        grounded_to_index[g] = i
    return index_to_grounded, grounded_to_index


def build_data_vocab(data_path):


    # return index_to_word word_to_index index_to_rel rel_to_index 
    datas = open(data_path).readlines()

    # language word is different from grounded object (event) which is extracted from embodied senor
    # therefore, we build different two embedding matrix for word_embedding and grounded_embed
    word_set = []
    grounded_set = []
    rel_set = []
    for d in datas:
        items = d.strip().split('\t')
        rel_set.append(items[0].strip())
        word_set.extend(items[1].strip().split())
        word_set.extend(items[2].strip().split())
        grounded_set.extend(items[3].strip().split('+'))
        grounded_set.extend(items[4].strip().split('+'))
        pass

    word_set.append('unk')
    grounded_set.append('unk')

    # natural langauge word
    word_dict = OrderedDict()
    for w in word_set:
        if w in word_dict:
            word_dict[w] += 1
        else:
            word_dict[w] = 1
    index_to_word = [w for w in word_dict]
    word_to_index = OrderedDict()
    for i,w in enumerate(index_to_word):
        word_to_index[w] = i

    # grounded objects 
    grounded_dict = OrderedDict()
    for g in grounded_set:
        if g in grounded_dict:
            grounded_dict[g] += 1
        else:
            grounded_dict[g] = 1
    index_to_grounded = [g for g in grounded_dict]
    grounded_to_index = OrderedDict()
    for i,g in enumerate(index_to_grounded):
        grounded_to_index[g] = i

    # rel 
    rel_dict = OrderedDict()
    for r in rel_set:
        if r in rel_dict:
            rel_dict[r] += 1
        else:
            rel_dict[r] = 1
    index_to_rel = [r for r in rel_dict]
    rel_to_index = OrderedDict()
    for i,r in enumerate(index_to_rel):
        rel_to_index[r] = i


    return datas, index_to_word, word_to_index, index_to_grounded, grounded_to_index, index_to_rel, rel_to_index


def build_index_data(datas, word_to_index, rel_to_index):
    index_datas = []

    for data in datas:
        items = data.split('\t')
        arg1 = items[1]
        arg2 = items[2]

        arg1_idx = [word_to_index[w] for w in arg1.strip().split()]
        arg2_idx = [word_to_index[w] for w in arg2.strip().split()]

        index_datas.append([arg1_idx, arg2_idx, rel_to_index[items[0]]])
    return index_datas


def build_grounded_index_data(datas, word_to_index, grounded_to_index, rel_to_index):
    index_datas = []

    for data in datas:
        items = data.split('\t')
        arg1 = items[1]
        arg2 = items[2]
        e1_ws = items[3]
        e2_ws = items[4]

        arg1_idx = [word_to_index[w] for w in arg1.strip().split()]
        arg2_idx = [word_to_index[w] for w in arg2.strip().split()]

        e1_idx = [grounded_to_index[w] for w in e1_ws.strip().split('+')]
        e2_idx = [grounded_to_index[w] for w in e2_ws.strip().split('+')]

        index_datas.append([arg1_idx, arg2_idx, e1_idx, e2_idx, rel_to_index[items[0]]])
    return index_datas


def build_batch_data(meta_index_data, batch_size):

    ##################
    ### batch_size ###
    ##################
    lens = []
    # random shuffle
    random.shuffle(meta_index_data)
    for data in meta_index_data:
        s1 = data[0]
        s2 = data[1]
        rel = data[2]
        lens.append(len(s1))
        lens.append(len(s2))
    maxlen = max(lens)
    # print len(meta_index_data)
    # print maxlen
    arg1_seq = np.zeros(( len(meta_index_data), maxlen))
    arg2_seq = np.zeros(( len(meta_index_data), maxlen))

    #
    new_data = []
    new_arg1 = []
    new_arg2 = []
    arg1_lens = []
    arg2_lens = []
    new_rels = []

    for i,data in enumerate(meta_index_data):

        arg1 = data[0]
        arg2 = data[1]
        rel = data[2]

        arg1_len = len(arg1)
        arg2_len = len(arg2)
        arg1_seq[i,:len(arg1)] = arg1[:len(arg1)]
        arg2_seq[i,:len(arg2)] = arg2[:len(arg2)]
        arg1_lens.append(arg1_len)
        arg2_lens.append(arg2_len)

        new_rels.append(rel)

    total_size = len(meta_index_data)
    batchs = []

    num_s = total_size/batch_size
    num_l = total_size%batch_size

    for i in range(num_s):
        batch_data = []
        batch_data.append(arg1_seq[i*batch_size:(i+1)*batch_size])
        batch_data.append(arg1_lens[i*batch_size:(i+1)*batch_size])
        batch_data.append(arg2_seq[i*batch_size:(i+1)*batch_size])
        batch_data.append(arg2_lens[i*batch_size:(i+1)*batch_size])
        
        batch_data.append(new_rels[i*batch_size:(i+1)*batch_size])
        batchs.append(batch_data)

    if num_l != 0:
        batch_data = []
        batch_data.append(arg1_seq[num_s*batch_size:])
        batch_data.append(arg1_lens[num_s*batch_size:])
        batch_data.append(arg2_seq[num_s*batch_size:])
        batch_data.append(arg2_lens[num_s*batch_size:])
        
        batch_data.append(new_rels[num_s*batch_size:])
        batchs.append(batch_data)

    return batchs, maxlen


def build_batch_grounded_data(meta_index_data, batch_size):

    ##################
    ### batch_size ###
    ##################
    lens = []
    # random shuffle
    random.shuffle(meta_index_data)
    for data in meta_index_data:
        s1 = data[0]
        s2 = data[1]
        e1 = data[2]
        e2 = data[3]
        rel = data[4]
        lens.append(len(s1))
        lens.append(len(s2))
        lens.append(len(e1))
        lens.append(len(e2))
    maxlen = max(lens)
    # print len(meta_index_data)
    # print maxlen
    arg1_seq = np.zeros(( len(meta_index_data), maxlen))
    arg2_seq = np.zeros(( len(meta_index_data), maxlen))

    e1_seq = np.zeros(( len(meta_index_data), maxlen))
    e2_seq = np.zeros(( len(meta_index_data), maxlen))
    #
    new_data = []
    new_arg1 = []
    new_arg2 = []
    new_e1 = []
    new_e2 = []
    arg1_lens = []
    arg2_lens = []
    e1_lens = []
    e2_lens = []
    new_rels = []

    for i,data in enumerate(meta_index_data):

        arg1 = data[0]
        arg2 = data[1]
        e1 = data[2]
        e2 = data[3]
        rel = data[4]

        arg1_len = len(arg1)
        arg2_len = len(arg2)
        arg1_seq[i,:len(arg1)] = arg1[:len(arg1)]
        arg2_seq[i,:len(arg2)] = arg2[:len(arg2)]
        arg1_lens.append(arg1_len)
        arg2_lens.append(arg2_len)
        
        e1_len = len(e1)
        e2_len = len(e2)
        e1_seq[i,:len(e1)] = e1[:len(e1)]
        e2_seq[i,:len(e2)] = e2[:len(e2)]
        e1_lens.append(e1_len)
        e2_lens.append(e2_len)


        new_rels.append(rel)

    total_size = len(meta_index_data)
    batchs = []

    num_s = total_size/batch_size
    num_l = total_size%batch_size

    for i in range(num_s):
        batch_data = []
        batch_data.append(arg1_seq[i*batch_size:(i+1)*batch_size])
        batch_data.append(arg1_lens[i*batch_size:(i+1)*batch_size])
        batch_data.append(arg2_seq[i*batch_size:(i+1)*batch_size])
        batch_data.append(arg2_lens[i*batch_size:(i+1)*batch_size])
        
        batch_data.append(e1_seq[i*batch_size:(i+1)*batch_size])
        batch_data.append(e1_lens[i*batch_size:(i+1)*batch_size])
        batch_data.append(e2_seq[i*batch_size:(i+1)*batch_size])
        batch_data.append(e2_lens[i*batch_size:(i+1)*batch_size])

        batch_data.append(new_rels[i*batch_size:(i+1)*batch_size])
        batchs.append(batch_data)

    if num_l != 0:
        batch_data = []
        batch_data.append(arg1_seq[num_s*batch_size:])
        batch_data.append(arg1_lens[num_s*batch_size:])
        batch_data.append(arg2_seq[num_s*batch_size:])
        batch_data.append(arg2_lens[num_s*batch_size:])
        
        batch_data.append(e1_seq[num_s*batch_size:])
        batch_data.append(e1_lens[num_s*batch_size:])
        batch_data.append(e2_seq[num_s*batch_size:])
        batch_data.append(e2_lens[num_s*batch_size:])

        batch_data.append(new_rels[num_s*batch_size:])
        batchs.append(batch_data)

    return batchs, maxlen

