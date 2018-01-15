import sys
import os
import numpy as np


class EReward(object):

    def __init__(self):

        self.observ_len = 0
        self.seen_items = []
        self.index_obj, self.obj_index = self.read_object_event('data/ItemType','data/BlockType')
        self.index_item = self.read_item_event('data/ItemType')
        self.index_block = self.read_block_event('data/BlockType')

        
        pass

    def read_object_event(self, p1, p2):

        self.index_item = self.read_item_event(p1)
        self.index_block = self.read_block_event(p2)

        index_obj = []
        index_obj.extend(self.index_item)
        index_obj.extend(self.index_block)

        # remove duplicate
        index_obj = list(set(index_obj))

        obj_index = dict()
        for i,w in enumerate(index_obj):
            obj_index[w] = i

        return index_obj, obj_index
    
    
    def read_block_event(self, pathname):

        with open(pathname) as f:
            a = f.readlines()
            pass
        index_item = []
        for w in a:
            index_item.append(w.strip())
        return index_item

    def read_item_event(self, pathname):

        with open(pathname) as f:
            a = f.readlines()
            pass
        index_item = []
        # add a additional item hand
        index_item.append("hand")
        for w in a:
            index_item.append(w.strip())
        return index_item



    def parse(self, obs_text):

        # position 
        posit_vector = np.zeros(5,)
        posit_vector[0] = obs_text['XPos']
        posit_vector[1] = obs_text['Yaw']
        posit_vector[2] = obs_text['YPos']
        posit_vector[3] = obs_text['ZPos']
        posit_vector[4] = obs_text['Pitch']

        # status
        state_vector = np.zeros(4,)
        state_vector[0] = obs_text['Life']
        state_vector[1] = obs_text['Air']
        state_vector[2] = obs_text['Food']
        state_vector[3] = obs_text['DamageTaken']

        self.items = dict()
        for item in obs_text:

            if "InventorySlot" in item and "item" in item:
                charitem = item.split("_")
                sizeitem = charitem[:2]
                sizeitem.append(u"size")

                # print '=====>' , item 
                item_name = obs_text[item]
                item_size = obs_text["_".join(sizeitem)]
                # print 'item_size', item_size
                if item_name in self.items:
                    self.items[item_name] = self.items[item_name] + item_size
                else:
                    self.items[item_name] = item_size
                pass

        item_vector = np.zeros(len(self.index_obj))
        for it in self.items: 
            item_vector[self.obj_index[it]] = self.items[it]
        return posit_vector, state_vector, item_vector

    def cal_reward(self, p_state, c_state):

        reward = 0
        p_pos, p_bod, p_itm = p_state
        c_pos, c_bod, c_itm = c_state
        _bod = c_bod[0] - p_bod[0]

        # time lost
        reward += -0.03
        
        # life drop
        if _bod < 0:
            reward += float(20)/float(_bod)

        _itm = c_itm - p_itm

        # find any new item or item change?
        newItemR = np.sum(np.logical_xor(c_itm,p_itm)) * 5
        reward += newItemR

        incItemR = np.sum(_itm)
        reward += incItemR
        return reward

