# 
## this file is a interface from script to environment  
##


import MalmoPython
import json
import logging
import os
import random
import sys
import traceback
import time
import Tkinter as tk

import numpy as np
from matplotlib import pyplot as PLT

from collections import OrderedDict
from PIL import Image

from util import *

import math
import tensorflow as tf
import threading

import json

import signal
import random

import copy

import pickle

from explore_reward import EReward
import xml.etree.ElementTree
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import fromstring, ElementTree, Element

from grounded_platform import *

# 

def static_event_detect(time_stamp_ob): # static is only for time stamp

    # for each observation 
    # DamageDealt/ MobsKilled/ Air/ grid/ TimeAlive/ XPos/ Life/ TotalTime/ DamageTaken/
    # ZPos/ Yaw/ YPos/ WorldTime/ XP/ IsAlive/ near_entites/ Score/ Food/ PlayersKilled
    # Name/ DistanceTravelled /LineOfSight / Pitch/
    # Agent action 
    # InventorySlot_(0~40)_size
    # InventorySlot_(0~40)_item
    ob = time_stamp_ob

    player = ob["Name"]

    max_item_num = 41
    static_events = []
    # agent state
    # agent inventory
    # environment
    # agent action
    # agent has ... / agent inventory
    item_str = "InventorySlot_"
    for i in range(max_item_num):

        if ob[item_str+str(i)+"_item"] != 'air':
            pass
        e_str = player + " has " + str(ob[item_str+str(i)+'_size']) + " " + ob[item_str+str(i)+'_item'] + ' inventory'
        e_f_str = player + " has some "+ ob[item_str+str(i)+'_item'] + ' inventory'
        if e_str not in static_events:
            static_events.append(e_str)
            static_events.append(e_f_str)

        if i == 0:
            static_events.append(player + " holds " + str(ob[item_str+str(i)+'_size']) + " " + ob[item_str+str(i)+"_item"] + " "+ 'hotbar')
            static_events.append(player + " holds the " + ob[item_str+str(i)+"_item"] + " "+ 'hotbar')
        pass

    actions = time_stamp_ob['action']
    # print '###### actions ###########################'
    # punch/attack action: agent attack object with tool
    # punch/attack action: agent attack object (without version?)
    for acts in actions:
        its = acts.split()
        if "craft" in its:
            materials = its[4:]
            ms = []
            for m in materials:
                ms.append(m.split('*')[0])
            ms_str = " ".join(ms)
            # complete version however in actual scene agent does not know the condition (with nothing)
            act_str = player + " makes "+ its[2] + " with " + ms_str
            time_stamp_ob['action'].append(act_str)
            act_str = player + " makes "+ its[2] # without version 
            time_stamp_ob['action'].append(act_str)

        if "attacks" in its:
            materials = its[4:]
            ms = []
            for m in materials:
                ms.append(m.split('*')[0])
            act_str = player + " attacks " + its[2]
            if acts == act_str:
                continue

            if act_str not in time_stamp_ob['action']:
                time_stamp_ob['action'].append(act_str)
            else:
                continue
        # static_events.extend()
    # 

    static_events.extend(time_stamp_ob['action'])
    # agent hold ... in hand/ with ... 

    # agent has more / agent inventory
    # agent doing attack/move

    # there is a ... in front of agent / environment
    # print ob['grid']
    for item in ob['grid']:
        if item != 'air' and item != 'obsidian':
            static_events.append('' + item + ' in front of '+ player+ '  grid_block')

    # print ob['near_entities']
    for item in ob['near_entities']:
        static_events.append('' + item['name'] + ' exist near_entities')

        if 'life' in item and item['life'] == 0:
            static_events.append(item['name'] + ' is killed')

    # there is less ... in environment / environment

    
    # agent gains more experiences /temporal 
    # print ob['XP']
    static_events.append(player + ' has ' + str(ob['XP']) + ' XP points')
    static_events.append(player + ' has ' + 'some XP points')
    # agent gets hurts / temporal Life
    # print ob['Life']
    static_events.append(player + ' has ' + str(ob['Life']) + ' Life points')
    if ob['Life'] == 20:
        static_events.append(player + " is very healthy Life")
    elif ob['Life'] < 20 and ob['Life'] > 10:
        static_events.append(player + " fells fine Life")
    elif ob['Life'] <= 10 and ob['Life'] > 0:
        static_events.append(player + " fells very bad Life")
        pass
        # print fe

    # print ob['Food']
    static_events.append(player + ' has ' + str(ob['Food']) + ' Food points')
    if ob['Food'] == 20:
        static_events.append(player + " is very full Food")
    elif ob['Food'] < 20 and ob['Food'] > 10:
        static_events.append(player + " need more food Food")
    elif ob['Food'] <= 10 and ob['Food'] > 0:
        static_events.append(player + " is very hungry Food")
        pass


    # agent action == 

    # for e in static_events:
    #    print 'static _ events === > ' , e
    #pass

    return static_events

def dynamic_event_detect(pre_ob, cur_ob): #

    # inventory
    # agent has ... / agent inventory

    dynamic_events = []
    # get the total picture of inventory
    item_str = "InventorySlot_"
    max_item_num = 41

    player = pre_ob["Name"]

    pre_inv_dict = dict()
    cur_inv_dict = dict()

    for i in range(max_item_num):

        inv_str = item_str+str(i)+"_item"

        if pre_ob[inv_str] in pre_inv_dict:
            pre_inv_dict[pre_ob[inv_str]] += pre_ob[item_str+str(i)+"_size"]
            pass
        else:
            pre_inv_dict[pre_ob[inv_str]] = 0
            pre_inv_dict[pre_ob[inv_str]] += pre_ob[item_str+str(i)+"_size"]
            pass
        
        if cur_ob[inv_str] in cur_inv_dict:
            cur_inv_dict[cur_ob[inv_str]] += cur_ob[item_str+str(i)+"_size"]
            pass
        else:
            cur_inv_dict[cur_ob[inv_str]] = 0
            cur_inv_dict[cur_ob[inv_str]] += cur_ob[item_str+str(i)+"_size"]
            pass

    # print '+++++ dynamic event detect'
    # print pre_inv_dict
    # print cur_inv_dict

    # detect change event in inventory

    # temporal positive
    for inv in cur_inv_dict:
        if inv not in pre_inv_dict:
            e_ = player + " gains " + str(cur_inv_dict[inv]) + ' ' + inv + " inventory"
            e_f_ = player + " gains " + 'some ' + inv + " inventory"
            dynamic_events.append(e_)
            dynamic_events.append(e_f_)
            pass
        else:
            inv_def = cur_inv_dict[inv] - pre_inv_dict[inv]
            if inv_def < 0:
                e_ = player + " loss " + str(inv_def) + " " + inv + " inventory"
                e_f_ = player + " loss some " + inv + " inventory"
                dynamic_events.append(e_)
                dynamic_events.append(e_f_)

            if inv_def > 0:
                e_ = player + " gains " + str(inv_def) + " " + inv + " inventory"
                e_f_ = player + " gains some " + inv + " inventory"
                dynamic_events.append(e_)
                dynamic_events.append(e_f_)
                
    # temporal negative
    for inv in pre_inv_dict:
        if inv not in cur_inv_dict:
            e_ = player + " loss " + str(pre_inv_dict[inv]) + " " + inv + " inventory"
            e_f_ = player + " loss some " + inv + " inventory"
            dynamic_events.append(e_)
            dynamic_events.append(e_f_)
        
    # hand tool

    # environment ob[grid]
    # print '+++++++grid++++++'
    # print pre_ob['grid']
    # print cur_ob['grid']
    for i,j in zip(pre_ob,cur_ob):
        block = 'block'
        if i == 'air' or i == 'obsidian':
            i = block
        if j == 'air' or j == 'obsidian':
            j = block
        if j != i and (j != block) : # appear
            e_ = block + ' appears in front of ' +player+ ' grid_block'
            dynamic_events.append(e_)
            pass
        if j != i and (j == block): # disappear
            e_ = block + ' disappears in front of '+player+' grid_block'
            dynamic_events.append(e_)
            pass

        pass

    # near entities ob[near_entities]
    # parsing entities
    cur_ent_list = []
    for cur_ent in cur_ob['near_entities']:
        cur_ent_id = cur_ent['name']
        if 'quantity' in cur_ent: # item 
            # cur_ent_id += '_' + str(cur_ent['quantity'])
            cur_ent_id += '+quantity'
        if 'life' in cur_ent: # entity
            cur_ent_id += '+life'
            pass
        cur_ent_list.append(cur_ent_id)
        
    pre_ent_list = []
    for pre_ent in pre_ob['near_entities']:
        pre_ent_id = pre_ent['name']
        if 'quantity' in pre_ent: # item 
            pre_ent_id += '+quantity'
            pass
        if 'life' in pre_ent: # entity
            pre_ent_id += '+life'
            pass
        pre_ent_list.append(pre_ent_id)

    # detect the difference from near entities
    # print '+++++++++++++++++++++++++++'


    # print pre_ent_list
    # print cur_ent_list
    while True:
        # python index bugs
        itemflag = True
        for ent in cur_ent_list:
            if ent in pre_ent_list:
                cur_ent_list.remove(ent)
                pre_ent_list.remove(ent)
                itemflag = False
        if itemflag: # no common anymore
            break
        pass

    if len(pre_ent_list) != 0:
        for ent in pre_ent_list:
            if ent == 'XPOrb':
                continue
            # entity item block
            ents = ent.split('+')
            if ents[1] == 'life':
               e_ = ents[0] + " is killed"
               # discard function

            if ents[1] == 'quantity':
                e_ = ents[0] + " is picked up"
                dynamic_events.append(e_)
        pass
    if len(cur_ent_list) != 0:
        for ent in cur_ent_list:
            if ent == 'XPOrb':
                continue
            # entity item block
            ents = ent.split('+')
            if ents[1] == 'life':
                e_ = ents[0] + " shows_up near_entities"
            if ents[1] == 'quantity':
                e_ = ents[0] + " drops near_entities"         
            dynamic_events.append(e_)
        pass
    # entities life change 

    # XP Food Life
    XP_diff = cur_ob['XP'] - pre_ob['XP']
    if XP_diff > 0:
        e_ = player + " gains " + str(XP_diff) + " experience"
        e_f_ = player + " gains some experience"
        dynamic_events.append(e_)
        dynamic_events.append(e_f_)
    if XP_diff < 0:
        e_ = player + " lost " + str(XP_diff) + " experience"
        e_f_ = player + " lost some experience"
        dynamic_events.append(e_)
        dynamic_events.append(e_f_)

    # Life 
    Life_diff = cur_ob['Life'] - pre_ob['Life']

    if Life_diff > 0:
        e_ = player + " recovers of " + str(Life_diff)
        dynamic_events.append(e_)
        e_ = player + " is recovering"
    if Life_diff < 0:
        e_ = player + " get hurts of " + str(Life_diff)
        dynamic_events.append(e_)
        e_ = player + " get some hurts"
        dynamic_events.append(e_)

    # Food
    Food_diff = cur_ob['Food'] - pre_ob['Food']
    if Food_diff > 0:
        e_ = player + " fells more satisfied"
        dynamic_events.append(e_)
    if Food_diff < 0:
        e_ = player + " fells more hungry"
        dynamic_events.append(e_)
        
    # near entities ob[near_entities]
    # parsing entities
    cur_lifes = OrderedDict()
    for cur_ent in cur_ob['near_entities']:
        if 'life' in cur_ent:
            cur_lifes[cur_ent['name']]=cur_ent['life']

    pre_lifes = OrderedDict()
    for pre_ent in pre_ob['near_entities']:
        if 'life' in pre_ent:
            pre_lifes[pre_ent['name']]=pre_ent['life']

    for ent in cur_lifes:
        if ent in pre_lifes:
            if ent != cur_ob['Name']:
                life_diff = cur_lifes[ent] - pre_lifes[ent]
                if life_diff < 0:
                    e_ = ent + " get some hurts"
                    dynamic_events.append(e_)
                    e_ = ent + " get hurts of " + str(life_diff)
                    dynamic_events.append(e_)
                if life_diff > 0:
                    e_ = ent + " is recovering"
                    dynamic_events.append(e_)
                    e_ = ent + " recovers of " + str(life_diff)
                    dynamic_events.append(e_)

    return dynamic_events


def event_to_NLP(total_obs):
    # extend all event
    # package in every timestamp 
    events = []
    e_s = []
    for ts in total_obs:
        events.extend(ts)
        for t in ts:
            e_s.append("+".join(t.split()))
        pass
    e_sets = set(e_s)
    print '++++++++++Event+++++++++++++++++++++++++++'
    for e in e_sets:
        print e

    for e in events:
        # print e
        e_text = data_to_text_generation(e)
        e_s.append(e_text)

    return e_s



def monitor_event(scene_experience):
    # load in json
    scene_txt = open(scene_experience).readlines()
    for exp in scene_txt:
        txt = exp.strip()
    scene_txt = scene_txt[0] # the scene_mission actually contains single record
    json_obj = json.loads(scene_txt)

    if json_obj is None:
        return []

    # observations for mission interation  
    obs = json_obj[1]
    # create event generator
    # --> agent state
    # --> agent inventory
    # --> environment
    obs_seq = []
    for ob_i in range(len(obs)):
        time_stamp = []
        ob = obs[ob_i]
        # parsing for single time stamp
        # ob is a dict
        # print ob
        # print 'static event detect ... '
        static_events = static_event_detect(ob)
        # print '#######Static Events For Each Time Stamp#####'
        time_stamp.extend(static_events)
        if ob_i > 0:
            pre_ob = obs[ob_i-1]
            cur_ob = obs[ob_i]
            # print 'dynamic_event_detect ... '
            time_stamp.extend(dynamic_event_detect(pre_ob, cur_ob))
        obs_seq.append(time_stamp)
        pass
    return obs_seq


def temporal_feature(whole_obs):

    # this function is supposed to capture the cocurrence and temporal feature
    # scenes -> [*]time_stamps -> [*]event

    # the temporal feature contains four kinds of relations
    # A-B(basic) !A-B(condition) A-!B(Cause/Probabilities) !A-!B which is the naive baysian method
    # a global matrix which calculate the connection of each event 
    # we are looking for 3 types of temporal features: 
    # inverse-time-clues right-time-clues same-time-clues 
    event_correl = OrderedDict()
    for scene in whole_obs:
        # for each scene 
        # print '##########Scene###########'
        # all event is handled in same scene first store every event in a matrix
        # event dict event occurs times temporal relation between events capture all events
        # and the sequence of event appeareance 
        event_seq = []
        event_seen = []
        for t_i in range(len(scene)):
            ts = scene[t_i]
            event_ts = []
            for e in ts: # ts is short for time stamp
                # locate the current event
                # use string form event as even id

                # force all string use lower form
                e_ = "+".join(e.split()).lower()
                if e_ not in event_seen:
                    event_seen.append(e_)
                    event_ts.append(e_)
            event_seq.append(event_ts)


        # for each scene it contains the sequence of every events
        # split the action from states, and make the time stamp more clearer
        # action make sense the each time stamp
        # print "A whole time stamps for each scene ... "
        event_from_past = []
        for ts in event_seq:
            # print '#######################'
            # print '#######################'
            # print '#######################'
            for e in ts:
                # print e
                if e not in event_correl:
                    event_correl[e] = OrderedDict()
                    event_correl[e]['count'] = 1
                    event_correl[e]['prev']  = OrderedDict()
                    event_correl[e]['meanwhile'] = OrderedDict()
                    event_correl[e]['following'] = OrderedDict()
                else:
                    event_correl[e]['count'] += 1
                # handle the past event in the scene
                # handle the past event in both time direction
                for pe in event_from_past:
                    # 
                    if pe not in event_correl[e]['prev']:
                        event_correl[e]['prev'][pe] = 1
                    else:
                        event_correl[e]['prev'][pe] += 1
                    if e not in event_correl[pe]['following']:
                        event_correl[pe]['following'][e] = 1
                    else:
                        event_correl[pe]['following'][e] += 1
                # handle all events happen in same time
                for se in ts:
                    # 
                    if se != e: # record the events happens in same time
                        if se not in event_correl[e]['meanwhile']:
                            event_correl[e]['meanwhile'][se] = 1
                        else:                        
                            event_correl[e]['meanwhile'][se] += 1
            # stack all event in time stamp into a past event collection
            # it is all in the past
            event_from_past.extend(ts)
            
        #  
        # for each event A and B , we must record the temporal occurence history including
        # A-B !A-B A-!B !A-!B A(main_count) B(main_count)
        """
        for event in event_correl:
            print "############---Time Stamp-----#############"
            print event, event_correl[event]['count']
            print '======>>>>>'
            for pe in event_correl[event]['prev']:
                print pe , " : ", event_correl[event]['prev'][pe]
                pass
            print '======>>>>>'
            for se in event_correl[event]['meanwhile']:
                print se , " : ", event_correl[event]['meanwhile'][se]
                pass
            print '======>>>>>'
            for fe in event_correl[event]['following']:
                print fe, " : ", event_correl[event]['following'][fe]
            print '***********************************************'
        """
        # directly return the feature ?????

    return event_correl


def temporal_clues_feature(event_correl, eA, eB):
    # A-B B-A  
    # test event
    events = event_correl.keys()
    """
    print '####################3'
    print eA
    print eB
    """
    # to reterial the store , handle the event and event does not exist
    # deal with the input event pair
    eA_count = event_correl[eA]['count']
    if eB in event_correl[eA]['prev']:
        A_prev_B = event_correl[eA]['prev'][eB]
    else:
        event_correl[eA]['prev'][eB] = 0
        A_prev_B = event_correl[eA]['prev'][eB]
    if eB in event_correl[eA]['meanwhile']:
        A_meanwhile_B = event_correl[eA]['meanwhile'][eB]
    else:
        event_correl[eA]['meanwhile'][eB] = 0
        A_meanwhile_B = event_correl[eA]['meanwhile'][eB]
    if eB in event_correl[eA]['following']:
        A_following_B = event_correl[eA]['following'][eB]
    else:
        event_correl[eA]['following'][eB] = 0
        A_following_B = event_correl[eA]['following'][eB]

    #
    eB_count = event_correl[eB]['count']
    if eA in event_correl[eB]['prev']:
        B_prev_A = event_correl[eB]['prev'][eA]
    else:
        event_correl[eB]['prev'][eA] = 0
        B_prev_A = event_correl[eB]['prev'][eA]
    if eA in event_correl[eB]['meanwhile']:
        B_meanwhile_A = event_correl[eB]['meanwhile'][eA]
    else:
        event_correl[eB]['meanwhile'][eA] = 0
        B_meanwhile_A = event_correl[eB]['meanwhile'][eA]
    if eA in event_correl[eB]['following']:
        B_following_A = event_correl[eB]['following'][eA]
    else:
        event_correl[eB]['following'][eA] = 0
        B_following_A = event_correl[eB]['following'][eA]

    """
    #
    print '####################3'
    print eA_count
    print A_prev_B
    print A_meanwhile_B
    print A_following_B
    print '####################3'
    print eB_count
    print B_prev_A
    print B_meanwhile_A
    print B_following_A
    # print '####################3'
    """
    eA_count_norm = float(eA_count)/float(eA_count)
    A_prev_B = float(A_prev_B) / float(eA_count)
    A_meanwhile_B = float(A_meanwhile_B) / float(eA_count)
    A_following_B = float(A_following_B) / float(eA_count)
    # print '####################3'
    eB_count_norm = float(eB_count)/float(eB_count)
    B_prev_A = float(B_prev_A) / float(eB_count)
    B_meanwhile_A = float(B_meanwhile_A) / float(eB_count)
    B_following_A = float(B_following_A) / float(eB_count)

    return (eA_count_norm, A_prev_B, A_meanwhile_B, A_following_B, eB_count_norm, B_prev_A, B_meanwhile_A, B_following_A)



def isExpansion(eA, eB, kn):
    isExpansion_flag = False
    if True:
        # 1. Detail 
        if ("gains" in eA) and ("gains" in eB) and (isExpansion_flag is False):
            eA_items = eA.split('+')
            eB_items = eB.split('+')
            # with same stuff
            if eA_items[2] == "more" and eA_items[3] == eB_items[3] and eB_items[2] != "more":
                isExpansion_flag = True
                
        if ("has" in eA) and ("has" in eB) and (isExpansion_flag is False):
            eA_items = eA.split('+')
            eB_items = eB.split('+')
            # with same stuff
            if eA_items[2] == "some" and eA_items[3] == eB_items[3] and eB_items[2] != "some":
                isExpansion_flag = True
                
        if ("loss" in eA) and ("loss" in eB) and (isExpansion_flag is False):
            eA_items = eA.split('+')
            eB_items = eB.split('+')
            # with same stuff
            if eA_items[2] == "some" and eA_items[3] == eB_items[3] and eB_items[2] != "some":
                isExpansion_flag = True


        if ('hurts' in eA) and ("hurts" in eB) and (isExpansion_flag is False):
            eA_items = eA.split('+')
            eB_items = eB.split('+')

            if eA_items[1] == "get" and eB_items[1] == "some":
                isExpansion_flag = True


        # 2. Conjunction with word cohension has common words
        # blank

        # 4. Temporal 
        if isTemporal(eA, eB, kn)[0] is not None:
            isExpansion_flag = True
                    
        if isExpansion_flag:
            return ['Expansion', eA, eB]
        else:
            return [None, eA, eB]




def access_event_facts(eA, eB, env_knowledge, total_distr, limit_size):

    # contorl the knowledge access
    # in this section , we use structual knowledge to label particular 
    # pattern of discourse relation

    tmp_cache = []

    for kn in env_knowledge:

        # 1. Cause
        if len(total_distr['Cause']) < limit_size:
            case = isCause(eA, eB, kn)
            if case[0] is not None:
                total_distr['Cause'].append(case)

        # 3. Condition
        if len(total_distr['Condition']) < limit_size:
            condition_case = isCondition(eA, eB, kn)
            if condition_case[0] is not None:
                total_distr['Condition'].append(condition_case)

        # 4. Concession # with two strategies
        if len(total_distr['Concession']) < limit_size:
            concession_case = generate_Concession(eA, eB, kn)
            if concession_case[0] is not None:
                total_distr['Concession'].append(concession_case)

    return tmp_cache

## we consider the event as basic unit with the help of strucutual environment knowledge.
## we propose the script to automatically generate the micro discourse pair 
##
# input eA eB and kn return discourse label 


def match_Relation(events, env_knowledge):

    # case generation takes too long time
    # search for all knowledge

    plain_txt_total_pairs = []
    for kn in env_knowledge:
        # players 
        players = ['adam','mary']
        ##################
        ## Cause Part: contains Result/Reason
        #################
        # Cause pattern 1 : player attacks kn_target with tool& player gains some target 
        #                    
        # attacks -> some 
        if "punch" in kn[1]:
            kn_target = kn[1][1].split('*')[0]
            kn_tool = kn[1][3].split('*')[0]
            end_keys = kn[2].keys()
            start_keys = kn[0].keys()
            gains = []
            for k in end_keys:
                if kn[2][k] != 0.5 and (k not in start_keys):
                    gains.append(k)
            # match for eA
            # match for different players 
            for ply in players:
                eA_event = "+".join([ply,'attacks',kn_target,'with',kn_tool]).lower()
                eA_flag = False
                if eA_event in events:
                    eA_flag = True
                candidate_events = []
                if eA_flag and len(gains) != 0:
                    nums = ['some']
                    # match for eB
                    meta_eB_event = [ply,'gains','X','Y','inventory']
                    for n in nums:
                        for g in gains:
                            tmp_ = copy.deepcopy(meta_eB_event)
                            tmp_[2] = n
                            tmp_[3] = g
                            candidate_events.append(tmp_)
                    for c in candidate_events:
                        eB_event = ("+".join(c)).lower()
                        if eB_event in events:
                            # Cause_cases.append(["Cause",eA_event,eB_event])
                            plain_txt = "#".join(["Cause",eA_event,eB_event])
                            plain_txt_total_pairs.append(plain_txt)
                            
        # Cause pattern 2 : player attacks kn_target& player gains some target 
        if "punch" in kn[1]:
            kn_target = kn[1][1].split('*')[0]
            kn_tool = kn[1][3].split('*')[0]
            end_keys = kn[2].keys()
            start_keys = kn[0].keys()
            gains = []
            for k in end_keys:
                if k not in start_keys:
                    gains.append(k)
            for ply in players:
                eA_event = "+".join([ply,'attacks',kn_target]).lower()
                eA_flag = False
                if eA_event in events:
                    eA_flag = True
                candidate_events = []
                if eA_flag and len(gains) != 0:
                    nums = ['some']
                    nums.extend([str(x) for x in range(1,10)])
                    # match for eB
                    meta_eB_event = [ply,'gains','X','Y','inventory']
                    for n in nums:
                        for g in gains:
                            tmp_ = copy.deepcopy(meta_eB_event)
                            tmp_[2] = n
                            tmp_[3] = g
                            candidate_events.append(tmp_)
                    for c in candidate_events:
                        eB_event = ("+".join(c)).lower()
                        if eB_event in events:
                            plain_txt = "#".join(["Cause",eA_event,eB_event])
                            plain_txt_total_pairs.append(plain_txt)

        # Cause pattern 3: player makes target& player loss some target
        # if ("makes" in eA) and ("loss" in eB) and "craft" in kn[1]:
        if "craft" in kn[1]:
            kn_target = kn[1][1].split("*")[0]
            kn_materials = kn[1][3:]
            materials = []
            for k in kn_materials:
                materials.append(k.split('*')[0])
            for ply in players:
                # match for eA
                eA_event = "+".join([ply,'makes',kn_target]).lower()
                eA_flag = False
                if eA_event in events:
                    eA_flag = True
                candidate_events = []
                if eA_flag and len(materials) != 0:
                    nums = ['some']
                    nums.extend([str(x) for x in range(1,10)])
                    # match for eB
                    meta_eB_event = [ply,'loss','X','Y','inventory']
                    for n in nums:
                        for g in materials:
                            tmp_ = copy.deepcopy(meta_eB_event)
                            tmp_[2] = n
                            tmp_[3] = g
                            candidate_events.append(tmp_)
                    for c in candidate_events:
                        eB_event = ("+".join(c)).lower()
                        if eB_event in events:
                            # Cause_cases.append(["Cause",eA_event,eB_event])
                            plain_txt = "#".join(["Cause",eA_event,eB_event])
                            plain_txt_total_pairs.append(plain_txt)

           
        # Cause pattern 4: player makes target & player gains some target
        # if ("makes" in eA) and ("more" in eB) and "craft" in kn[1]:
        if "craft" in kn[1]:
            kn_target = kn[1][1].split("*")[0]
            for ply in players:
                # match for eA
                eA_event = "+".join([ply,'makes',kn_target]).lower()
                eA_flag = False
                if eA_event in events:
                    eA_flag = True
                candidate_events = []
                if eA_flag and len([kn_target]) != 0:
                    nums = ['some']
                    nums.extend([str(x) for x in range(1,10)])
                    # match for eB
                    meta_eB_event = [ply,'gains','X','Y','inventory']
                    for n in nums:
                        for g in [kn_target]:
                            tmp_ = copy.deepcopy(meta_eB_event)
                            tmp_[2] = n
                            tmp_[3] = g
                            candidate_events.append(tmp_)
                    for c in candidate_events:
                        eB_event = "+".join(c).lower()
                        if eB_event in events:
                            # Cause_cases.append(["Cause",eA_event,eB_event])
                            plain_txt = "#".join(["Cause",eA_event,eB_event])
                            plain_txt_total_pairs.append(plain_txt)
        


        ########################
        # Condition part 
        #######################
        # Condition pattern 1
        # condition is very similiar with Cause
        if "punch" in kn[1]:
            kn_target = kn[1][1].split("*")[0]
            kn_tool = kn[1][3].split("*")[0]
            for ply in players:
                eA_event = "+".join([ply,'attacks',kn_target,'with',kn_tool]).lower()
                eA_flag = False
                if eA_event in events:
                    eA_flag = True
                candidate_events = []
                if eA_flag:
                    nums = ['some']
                    # nums.extend([str(x) for x in range(1,10)])
                    # match for eB
                    meta_eB_event = [ply,'has','X','Y','inventory']
                    for n in nums:
                        for g in [kn_tool]:
                            tmp_ = copy.deepcopy(meta_eB_event)
                            tmp_[2] = n
                            tmp_[3] = g
                            candidate_events.append(tmp_)
                    
                    for c in candidate_events:
                        eB_event = "+".join(c).lower()
                        if eB_event in events:
                            # Condition_cases.append(["Condition", eB_event, eA_event])
                            plain_txt = "#".join(["Condition", eB_event, eA_event])
                            plain_txt_total_pairs.append(plain_txt)
        
        # Condition pattern 2
        if "craft" in kn[1]:
            kn_target = kn[1][1].split("*")[0]
            kn_materials = kn[1][3:]
            for ply in players:
                materials = []
                for k in kn_materials:
                    materials.append(k.split('*')[0])
                # match for eA
                eA_event = "+".join([ply,'makes',kn_target]).lower()
                eA_flag = False
                if eA_event in events:
                    eA_flag = True
                candidate_events = []
                if eA_flag and len(materials) != 0:
                    nums = ['some']
                    nums.extend([str(x) for x in range(1,10)])
                    # match for eB
                    meta_eB_event = [ply,'has','X','Y','inventory']
                    for n in nums:
                        for g in materials:
                            tmp_ = copy.deepcopy(meta_eB_event)
                            tmp_[2] = n
                            tmp_[3] = g
                            candidate_events.append(tmp_)
                    for c in candidate_events:
                        eB_event = ("+".join(c)).lower()
                        if eB_event in events:
                            # Condition_cases.append(["Condition",eB_event,eA_event])
                            plain_txt = "#".join(["Condition", eB_event, eA_event])
                            plain_txt_total_pairs.append(plain_txt)

        # Temporal_syn pattern 1: target in front of player   
        # divide into two types
        # Build Temporal Case for synchrony
        # Temporal pattern 1
        if "craft" in kn[1]:
            kn_target = kn[1][1].split("*")[0]
            kn_materials = kn[1][3:]
            # target gains / materials loss
            materials = []
            for k in kn_materials:
                materials.append(k.split('*')[0])
            # match for eA
            for ply in players:
                eA_event = "+".join([ply,'gains','some',kn_target,'inventory']).lower()
                eA_flag = False
                if eA_event in events:
                    eA_flag = True
                    
                candidate_events = []
                if eA_flag and len(materials) != 0:
                    nums = ['some']
                    # nums.extend([str(x) for x in range(1,10)])
                    # match for eB
                    meta_eB_event = [ply,'loss','X','Y','inventory']
                    for n in nums:
                        for g in materials:
                            tmp_ = copy.deepcopy(meta_eB_event)
                            tmp_[2] = n
                            tmp_[3] = g
                            candidate_events.append(tmp_)
                    for c in candidate_events:
                        eB_event = ("+".join(c)).lower()
                        if eB_event in events:
                            # Temporal_syn_cases.append(["Temporal_syn", eA_event, eB_event])
                            plain_txt = "#".join(["Temporal_syn", eA_event, eB_event])
                            plain_txt_total_pairs.append(plain_txt)


    # finish the knowledge over 
    cause_instances = []
    condition_instances = []
    for p in plain_txt_total_pairs:
        items = p.strip().split("#")
        if items[0] == "Condition":
            condition_instances.append([items[0], items[1], items[2]])
        if items[0] == "Cause":
            cause_instances.append([items[0], items[1], items[2]])


    ########################3
    # Concession part
    # Concession is mainly modified from Cause/Condition
    # Concession pattern 1
    for c in cause_instances:
        if True:
            eA = c[1].split('+')
            eB = c[2].split('+')
            eA_event = copy.deepcopy(eA)
            eB_event = copy.deepcopy(eB)
            # meta_eB_event = ['Agent','gains','X','Y','inventory']
            if "attacks" in eA_event and "gains" in eB_event:
                eB_event[1] = 'loss'
                plain_txt = "#".join(["Concession", c[1], '+'.join(eB_event)])
                plain_txt_total_pairs.append(plain_txt)
                
            # 0 cases    
            eA_event = copy.deepcopy(eA)
            eB_event = copy.deepcopy(eB)
            # meta_eB_event = ['Agent','gains','X','Y','inventory']
            if "attacks" in eA_event and "gains" in eB_event:
                eB_event[2] = '0'
                plain_txt = "#".join(["Concession", c[1], '+'.join(eB_event)])
                plain_txt_total_pairs.append(plain_txt)
            
            # 0 cases
            eA_event = copy.deepcopy(eA)
            eB_event = copy.deepcopy(eB)
            if "makes" in eA_event and "gains" in eB_event:
                eB_event[1] = 'loss'
                plain_txt = "#".join(["Concession", c[1], '+'.join(eB_event)])
                plain_txt_total_pairs.append(plain_txt)
            
            # 0 cases
            eA_event = copy.deepcopy(eA)
            eB_event = copy.deepcopy(eB)
            if "makes" in eA_event and "gains" in eB_event:
                eB_event[2] = '0'
                plain_txt = "#".join(["Concession", c[1], '+'.join(eB_event)])
                plain_txt_total_pairs.append(plain_txt)

            eA_event = copy.deepcopy(eA)
            eB_event = copy.deepcopy(eB)
            if "makes" in eA_event and "loss" in eB_event:
                eB_event[1] = "gains"
                plain_txt = "#".join(["Concession", c[1], '+'.join(eB_event)])
                plain_txt_total_pairs.append(plain_txt)
                
            # 0 cases 
            eA_event = copy.deepcopy(eA)
            eB_event = copy.deepcopy(eB)
            if "makes" in eA_event and "loss" in eB_event:
                eB_event[2] = "0"
                plain_txt = "#".join(["Concession", c[1], '+'.join(eB_event)])
                plain_txt_total_pairs.append(plain_txt)


            eA_event = copy.deepcopy(eA)
            eB_event = copy.deepcopy(eB)
            if "attacks" in eA_event and "gains" in eB_event:
                eB_event[1] = 'loss'
                plain_txt = "#".join(["Concession", c[1], '+'.join(eB_event)])
                plain_txt_total_pairs.append(plain_txt)
            
            # 0 cases 
            eA_event = copy.deepcopy(eA)
            eB_event = copy.deepcopy(eB)
            if "attacks" in eA_event and "gains" in eB_event:
                eB_event[2] = '0'
                plain_txt = "#".join(["Concession", c[1], '+'.join(eB_event)])
                plain_txt_total_pairs.append(plain_txt)

        
    for c in condition_instances:
        if True:
            eA_event = c[1].split('+')
            eB_event = c[2].split('+')
            if "has" in eA_event and "makes" in eB_event:
                eA_event[2] = '0'
                # Concession_cases.append(["Concession", '+'.join(eB_event), c[1]])
                plain_txt = "#".join(["Concession", '+'.join(eA_event), c[2]])
                plain_txt_total_pairs.append(plain_txt)
    # 
    # print "Knowledge finish ... "

    return plain_txt_total_pairs



def mirco_discourse_generator(whole_obs, event_correl, env_knowledge, store_path):

    # this function label the scene sentence to generate the mirco_discourse
    # each mirco discourse contains only two sentences, it express the certain
    # discourse relation which hold them together.
    # we consider seven main discourse relation here, each discours relation 
    # reflect different coherence meaning. 
    
    events = event_correl.keys()
    print "found " , len(events) , " in total"
    print "after remove the duplicated cases : " , len(set(events))

    total_pairs = OrderedDict()
    total_pairs['Cause'] = []
    total_pairs['Condition'] = []
    total_pairs['Concession'] = []
    total_pairs['Temporal'] = []
    total_pairs['Temporal_syn'] = []
    total_pairs['Expansion'] = []
    total_pairs['Comparison'] = []

    start_time = time.time()
    plain_txt_total_pairs = []

    # we now collecting event from each scene not whole scenes ...
    print "Now we process all scenes from agent experience ... "
    for scene in whole_obs:
        scene_event = []
        for time_stamp in scene:
            for e in time_stamp:
                plain_e = ('+'.join(e.split()).lower())
                if plain_e not in scene_event:
                    scene_event.append(plain_e)
        # scene_event
        # print "In this scene, we found " , len(scene_event) , " event "
        # for e in scene_event:
        #     print e
        # print '###########'
        scene_total_pairs = match_Relation(scene_event, env_knowledge)
        for s in scene_total_pairs:
            if s not in plain_txt_total_pairs:
                plain_txt_total_pairs.append(s)
    
    # the collection finish ... with Cause/Condition/Concession
    # build related Expansion ... 
    # use type:list from Expansion to aganist Cause/Condition/Concession
    # divides collection
    event_collection = []
    pairs_collection = []
    
    gains_events = []
    attacks_events = []
    makes_events = []
    has_events = []
    loss_events = []
    for p in plain_txt_total_pairs:
        items = p.strip().split('#')
        pairs_collection.append(items[1]+"#"+items[2])
        e1 = items[1]
        e2 = items[2]
        event_collection.append(e1)
        event_collection.append(e2)

    for e in event_collection:
        if "gains" in e:
            gains_events.append(e)
        if "attacks" in e:
            attacks_events.append(e)
        if "makes" in e:
            makes_events.append(e)
        if "has" in e:
            has_events.append(e)
        if "loss" in e:
            loss_events.append(e)
        pass

    #
    Expansion_list_pairs = []
    # generate noise instances from Cause/Condition/Concession 
    # we consider these instances as List type
    for p in plain_txt_total_pairs:

        items = p.strip().split('#')
        eA = items[1]
        eB = items[2]

        if items[0] == "Cause" or items[0] == "Concession":
            # makes/ loss
            if "makes" in eA and "with" not in eA and "loss" in eB:
                while True:
                    l_e = loss_events[random.randint(0,len(loss_events)-1)]
                    pair = '#'.join([eA,l_e])
                    eA_items = eA.strip().split('+')
                    eB_items = eB.strip().split('+')
                    le_items = l_e.strip().split('+')
                    if pair not in pairs_collection \
                            and pair not in Expansion_list_pairs \
                            and le_items[3] != eB_items[3] \
                            and le_items[0] == eB_items[0]:
                        Expansion_list_pairs.append(pair)
                        break
            # makes/ gains
            if "makes" in eA and "with" not in eA and "gains" in eB:
                while True:
                    g_e = gains_events[random.randint(0,len(gains_events)-1)]
                    pair = '#'.join([eA,g_e])
                    eA_items = eA.strip().split('+')
                    eB_items = eB.strip().split('+')
                    ge_items = g_e.strip().split('+')
                    if pair not in pairs_collection \
                            and pair not in Expansion_list_pairs \
                            and eB_items[3] != ge_items[3] \
                            and eB_items[0] == ge_items[0]:
                        Expansion_list_pairs.append(pair)
                        break

            # attacks/gains
            if "attacks" in eA and "with" not in eA and "gains" in eB:
                while True:
                    g_e = gains_events[random.randint(0,len(gains_events)-1)]
                    pair = '#'.join([eA,g_e])
                    eA_items = eA.strip().split('+')
                    eB_items = eB.strip().split('+')
                    ge_items = g_e.strip().split('+')

                    if pair not in pairs_collection \
                            and pair not in Expansion_list_pairs \
                            and eB_items[3] != ge_items[3] \
                            and eB_items[0] == ge_items[0]:
                        Expansion_list_pairs.append(pair)
                        break
                    
            # attacks/loss from Concession
            if "attacks" in eA and "with" not in eA and "loss" in eB:
                while True:
                    l_e = loss_events[random.randint(0,len(loss_events)-1)]
                    pair = '#'.join([eA,l_e])
                    eA_items = eA.strip().split('+')
                    eB_items = eB.strip().split('+')
                    le_items = l_e.strip().split('+')

                    if pair not in pairs_collection \
                            and pair not in Expansion_list_pairs \
                            and eB_items[3] != le_items[3] \
                            and eB_items[0] == le_items[0]:
                        Expansion_list_pairs.append(pair)
                        break
            pass

        if items[0] == "Condition" or items[0] == "Concession":
            # has/makes
            if "makes" in eB and "has" in eA:
                while True:
                    h_e = has_events[random.randint(0,len(has_events)-1)]
                    pair = '#'.join([h_e,eB])
                    eA_items = eA.strip().split('+')
                    eB_items = eB.strip().split('+')
                    he_items = h_e.strip().split('+')
                    if pair not in pairs_collection \
                            and pair not in Expansion_list_pairs \
                            and eA_items[3] != he_items[3] \
                            and eA_items[0] == he_items[0]:
                        Expansion_list_pairs.append(pair)
                        break
            
            if "has" in eA and "makes" in eB:
                while True:
                    m_e = makes_events[random.randint(0,len(makes_events)-1)]
                    pair = '#'.join([eA,m_e])
                    eA_items = eA.strip().split('+')
                    eB_items = eB.strip().split('+')
                    me_items = m_e.strip().split('+')
                    if pair not in pairs_collection \
                            and pair not in Expansion_list_pairs \
                            and eB_items[2] != me_items[2] \
                            and eB_items[0] == me_items[0]:
                        Expansion_list_pairs.append(pair)
                        break
            # has/attacks
            pass


    # end of Expansion
    # Test with no Expansion ...  
    for e in Expansion_list_pairs:
        plain_txt_total_pairs.append("Expansion#"+e)
        pass
    
    Comparison_list_pairs = []
    # generate noise instances from Cause/Condition/Concession 
    # we consider these instances as List type
    for p in plain_txt_total_pairs:

        items = p.strip().split('#')
        eA = items[1]
        eB = items[2]

        if items[0] == "Temporal_syn":
            # makes / loss[noise]
            if "gains" in eA and "loss" in eB:
                while True:
                    l_e = loss_events[random.randint(0,len(loss_events)-1)]
                    pair = '#'.join([eA,l_e])
                    eA_items = eA.strip().split('+')
                    eB_items = eB.strip().split('+')
                    ls_items = l_e.strip().split('+')
                    if pair not in pairs_collection \
                            and pair not in Comparison_list_pairs \
                            and eB_items[3] != ls_items[3] \
                            and eB_items[0] == ls_items[0]:
                        Comparison_list_pairs.append(pair)
                        break

                    pass

            # makes[noise] / loss
            if "gains" in eA and "loss" in eB:
                while True:
                    g_e = gains_events[random.randint(0,len(gains_events)-1)]
                    pair = '#'.join([g_e, eB])
                    eA_items = eA.strip().split('+')
                    eB_items = eB.strip().split('+')
                    ge_items = g_e.strip().split('+')

                    if pair not in pairs_collection \
                            and pair not in Comparison_list_pairs \
                            and eA_items[3] != ge_items[3] \
                            and eA_items[0] == ge_items[0]:
                        Comparison_list_pairs.append(pair)
                        break
                    pass


    # end of Comparison
    # Test with no Comparison ...  
    for e in Comparison_list_pairs:
        plain_txt_total_pairs.append("Comparison#"+e)
        pass

    # handle the ratio
    # Expansion
    # Part 1
    # The expansion is a typical relation which not related to the temporal clues. In fact, 
    # Epansion relation usually connect the proposed sentences in semantic way.
    # the Expansion happens when people talk about something, and it is not temporal, 
    # and they don't want to emphasize the difference, but to add more details.
    # in our case, Agent first describe the event it observes and add more detail about it
    
    # e.g.1 Adam gains more stuff, in fact, he gains 3 stuff (shows more details)
    # e.g.2 Adam gets hurt, he loss 20 lift points (they are actually same thing)
    # e.g.3 Adam attack cow with iron_axe, Adam attack sheep with iron axe too. (Conjunction)
    # e.g.4 (List, by definition these sentences are not unnecessarily related, this type is very
    # simliar to temporal, but without time indicator)

    # leave blank
    # 
    # finish the collecting 
    for p in plain_txt_total_pairs:
        items = p.split('#')
        # if items[0] == "Concession":
        #     print items
        # append by name 
        total_pairs[items[0]].append(items)
    
    # elapsed time
    end_time = time.time()
    gap_time = end_time - start_time
    print "cost time : " , gap_time , 's '

    # eliminate the duplicated instance
    for item in total_pairs:
        # print '################'
        # print item
        # print len(total_pairs[item])
        tmp_cache = []
        for pair in total_pairs[item]:
            txt = pair[1]+"#"+pair[2]
            tmp_cache.append(txt)
        refine = list(set(tmp_cache))

        new_ = []
        for r in refine:
            pairs = r.split('#')
            new_.append([item, pairs[0], pairs[1]])

        total_pairs[item] = new_
        # print item
        # print len(total_pairs[item])

    # generate_Expansion
    # deal with the number distribution of different discours relations
    # 1700 for 6 relations

    ## code snippet 
    # store to the disk
    # no json version 
    f = open(store_path, 'w')
    for p in total_pairs:
        for rec in total_pairs[p]:
            f.write("\t".join(rec))
            f.write("\n")
    f.close()
    
    ## code snippet
    # store to the disk in json version
    f = open(store_path+"_json" , "w")
    total_pairs_txt = json.dumps(total_pairs)
    f.write(total_pairs_txt)
    f.close()


def data_to_text_generation(event):

    # do reference to the data to text (text generation template)
    # convert static and dynamic event into natural language 
    # we use a simple text generation (data to text) solution to handle this problem 
    #  

    plm = platform()
    
    matchflag = False
    # [u'Agent', u'has', u'1', u'wooden_shovel', u'inventory']
    items = event.split("+")
    event_text = ""

    # Static
    # 01. Agent has . inventory
    if "has" in event and "inventory" in event:
        matchflag = True
        if items[3] == "air":
            event_text = "there is still space left in " + items[0] + "'s bag"
        else:
            event_text = items[0] + " has " + items[2] + ' ' + items[3]
        pass

    # 2. Agent holds . size in hand
    if "holds" in event and "hotbar" in event:
        matchflag = True
        # Agent holds 1 stone_sword hotbar
        if items[3] == 'air':
            event_text = items[0] + " has nothing in hand"
        else:
            event_text = items[0] + " holds " + items[3] + " in hand"
        pass

    # 3. in front of Agent
    # grid
    if "in+front+of" in event:
        matchflag = True
        event_text = "there is " + items[0] + " in front of " + items[0]
        # print event_text
        pass

    # 4. nearby entities event
    if "near_entities" in event and "exist" in event:
        matchflag = True 
        event_text = "there is " + items[0] + " near " + items[0]
        pass

    #
    if "drops" in event:
        matchflag = True
        event_text = items[0] + " fells " + " near " + items[0]
        pass

    # 5. is killed / is dead
    if "is+killed" in event:
        matchflag = True
        event_text = items[0] + " is killed"
        pass

    # 6. Agent has . XP points
    if "has" in event and "xp+points" in event:
        matchflag = True
        if items[2] =="some":
            event_text = items[0] + " now has some experience"
        else:
            if float(items[2]) == 0:
                event_text = items[0] + " has no experience at all"
            else:
                event_text = items[0] + " now has some experience"
        pass

    # 7. Agent has . Life points
    if "has" in event and "life+points" in event:
        matchflag = True
        if float(items[2]) == 20:
            event_text = items[0] + " is very healthy"
        elif float(items[2]) < 20 and float(items[2]) > 10:
            event_text = items[0] + " fells fine"
        elif float(items[2]) <= 10 and float(items[2]) > 0:
            event_text = items[0] + " fells very bad"
        pass

    # 8. Agent has . Food points
    if "has" in event and "food+points" in event:
        matchflag = True
        # print event
        if float(items[2]) == 20:
            event_text = items[0] + " fells full"
        else:
            event_text = items[0] + " can still eat more food"

        pass

    # 24. Agent action
    if "attacks" in event and "with" in event:
        matchflag = True
        # print event

        targettype, target = plm.isEntitiesOrBlock(items[2])

        if targettype == "EntityTypes":
            event_text = items[0] + " attacks " + items[2] + " with " + items[4]

        if targettype == "BlockType":
            event_text = items[0] + " digs " + items[2] + " with " + items[4]

        pass
    
    # 24. Agent action
    if "attacks" in event and "with" not in event:
        matchflag = True
        # print event

        targettype, target = plm.isEntitiesOrBlock(items[2])

        if targettype == "EntityTypes":
            event_text = items[0] + " attacks " + items[2]

        if targettype == "BlockType":
            event_text = items[0] + " digs " + items[2]

        pass


    if "moves" in event:
        matchflag = True
        event_text = items[0] + " moves forward"
        pass

    if "quit" in event:
        matchflag = True
        event_text = items[0] + " quits from scene"
        pass

    if "swap" in event:
        matchflag = True
        event_text = items[0] + " swaps " + items[2] + " to hand"
        pass
    
    # Dynamic 
    # 9. Agent gains more ...
    if "gains" in event and "inventory" in event:

        matchflag = True
        event_text = items[0] + " gains " + items[2] + " " + items[3] 
        pass

    # 10. Agent loss
    if "loss" in event and "inventory" in event:
        matchflag = True
        event_text = items[0] + " lost " + items[2] + " " + items[3]
        pass
    
    # 13. block appears in front of Agent
    if "appears" in event and "grid_block" in event:
        matchflag = True
        event_text = items[0] + " appears in front of player"
        pass

    # 14. disappears in front of Agent
    if "disappears" in event and "grid_block" in event:
        matchflag = True
        event_text = items[0] + " disappears in front of player"
        pass
    
    # 15. ... is picked up
    if "picked+up" in event:
        matchflag = True
        event_text = items[0] + " is picked up"
        pass

    # 16. entity shows up
    if "shows_up" in event:
        matchflag = True
        event_text = items[0] + " shows up near player" 
        pass

    # 17. entity drops
    if "drops" in event:
        matchflag = True
        event_text = items[0] + " drops near player" 
        pass
    
    # 18. Agent gains more . experience
    if "gains" in event and "experience" in event:
        matchflag = True
        event_text = items[0] + " gains some experience"
        pass

    # 19. Agent lost . experiences
    if "lost" in event and "experience" in event:
        matchflag = True
        event_text = items[0] + " lost some experience"

        pass

    # 20. Agent recovers with ... Life
    if "recovers" in event:
        matchflag = True
        event_text = items[0] + " is recovering"
        pass

    # 21. Agent get hurts with ... Life
    if "get+hurts" in event:
        matchflag = True
        event_text = items[0] + " gets hurt"
        pass

    # 22. Agent fells more satisfied
    if "fells+more+satisfied" in event:
        matchflag = True
        event_text = items[0] + " fells more satisfied"
        pass

    # 23. Agent fells more hungry
    if "fells+more+hungry" in event:
        matchflag = True
        event_text = items[0] + " fells more hungry"
        pass

    # 24.
    if "life" in event and "very+healthy" in event:
        matchflag = True
        event_text = items[0] + " is very healthy"
        pass

    # 25.
    if "food" in event and "very+full" in event:
        matchflag = True
        event_text = items[0] + " is very full"
        pass

    # 26. 
    if "craft" in event:
        matchflag = True
        materials = items[4:]
        ms = []
        for m in materials:
            ms.append(m.split('+')[0])
        event_text = items[0] + " makes " + items[2] + " with " + " ".join(ms)
        pass

    # 27.
    if "makes" in event:
        matchflag = True
        event_text = items[0] + " makes some " + items[-1]
        pass

    # 28. 
    if matchflag is False:
        print 'match sentence pattern fail ... '
        print event
        raise ValueError

    return event_text



def from_rec_to_instance(rec):

    # to Explicit and Implicit 
    # connectives 6 relations -> {}

    rec = rec.strip().split('\t')
    relation = rec[0]
    arg1 = rec[1]
    arg2 = rec[2]

    Explicit_ins = None
    Implicit_ins = None

    # Cause
    if relation == "Cause":

        arg1_text = data_to_text_generation(arg1)
        arg2_text = data_to_text_generation(arg2)

        # Explicit
        exp_connectives = ["becuase/1","therefore/2","then/2","since/1"]
        con_word = exp_connectives[random.randint(0,len(exp_connectives)-1)]
        w, p = con_word.split('/')
        if p == "1":
            ins = ["Cause", w + " " + arg1_text, arg2_text, arg1, arg2]
            Explicit_ins = ins
            pass
        if p == "2":
            ins = ["Cause", arg1_text, w + " " + arg2_text, arg1, arg2]
            Explicit_ins = ins
            pass

        # Implicit
        imp_connectives = ["as/1","and/2"]
        con_word = imp_connectives[random.randint(0,len(imp_connectives)-1)]
        w, p = con_word.split('/')
        if p == "1":
            ins = ["Cause" , w + " " + arg1_text , arg2_text, arg1, arg2]
            Implicit_ins = ins
            pass
        if p == "2":
            ins = ["Cause" , arg1_text , w + " " + arg2_text, arg1, arg2]
            Implicit_ins = ins
            pass
        pass

    # Condition
    if relation == "Condition":

        arg1_text = data_to_text_generation(arg1)
        arg2_text = data_to_text_generation(arg2)
        
        # Explicit
        exp_connectives = ["if/1","when/1"]
        con_word = exp_connectives[random.randint(0,len(exp_connectives)-1)]
        w, p = con_word.split('/')
        if p == "1":
            ins = ["Condition", w+" "+arg1_text, arg2_text, arg1, arg2]
            Explicit_ins = ins
            pass
        if p == "2":
            ins = ["Condition", arg1_text, w+" "+arg2_text, arg2, arg1]
            Explicit_ins = ins
            pass

        # Implicit
        imp_connectives = ["and/2","as/1"]
        con_word = imp_connectives[random.randint(0,len(imp_connectives)-1)]
        w, p = con_word.split('/')
        if p == "1":
            ins = ["Condition" , w + " " + arg1_text , arg2_text, arg1, arg2]
            Implicit_ins = ins
            pass
        if p == "2":
            ins = ["Condition", arg1_text , w + " " + arg2_text, arg1, arg2]
            Implicit_ins = ins
            pass
        pass

    
    # Concession
    if relation == "Concession":
        
        arg1_text = data_to_text_generation(arg1)
        arg2_text = data_to_text_generation(arg2)
        
        # Explicit
        exp_connectives = ["but/2", "however/2", "while/2", "although/1" ]
        con_word = exp_connectives[random.randint(0,len(exp_connectives)-1)]
        w, p = con_word.split('/')
        if p == "1":
            ins = ["Concession", w + " " + arg1_text, arg2_text, arg1, arg2]
            Explicit_ins = ins
            pass
        if p == "2":
            ins = ["Concession", arg1_text, w + " " + arg2_text, arg1, arg2]
            Explicit_ins = ins
            pass

        # Implicit
        imp_connectives = ["as/1","and/2"]
        con_word = imp_connectives[random.randint(0,len(imp_connectives)-1)]
        w, p = con_word.split('/')
        if p == "1":
            ins = ["Concession" , w + " " + arg1_text , arg2_text, arg1, arg2]
            Implicit_ins = ins
            pass
        if p == "2":
            ins = ["Concession", arg1_text , w + " " + arg2_text, arg1, arg2]
            Implicit_ins = ins
            pass
        pass

    # Temporal
    if relation == "Temporal":
        
        arg1_text = data_to_text_generation(arg1)
        arg2_text = data_to_text_generation(arg2)
        
        # Explicit
        exp_connectives = ["before/2", "then/2", "after/1", "before/1", "since/1"]
        con_word = exp_connectives[random.randint(0,len(exp_connectives)-1)]
        w, p = con_word.split('/')
        if p == "1":
            ins = ["Temporal", w + " " + arg1_text, arg2_text, arg1, arg2]
            Explicit_ins = ins
            pass
        if p == "2":
            ins = ["Temporal", arg1_text, w + " " + arg2_text, arg1, arg2]
            Explicit_ins = ins
            pass

        # Implicit
        imp_connectives = ["a few minutes later/2", "a moment later/2", "a few minutes before/1", "a moment before/1", "just minutes later/1", "just minutes before/2"]
        con_word = imp_connectives[random.randint(0,len(imp_connectives)-1)]
        w, p = con_word.split('/')
        if p == "1":
            ins = ["Temporal" , w + " " + arg1_text , arg2_text, arg1, arg2]
            Implicit_ins = ins
            pass
        if p == "2":
            ins = ["Temporal", arg1_text , w + " " + arg2_text, arg1, arg2]
            Implicit_ins = ins
            pass

        pass

    # Temporal 
    if relation == "Temporal_syn":
        
        arg1_text = data_to_text_generation(arg1)
        arg2_text = data_to_text_generation(arg2)
        
        # Explicit
        exp_connectives = ["when/1", "while/2" ]
        con_word = exp_connectives[random.randint(0,len(exp_connectives)-1)]
        w, p = con_word.split('/')
        if p == "1":
            ins = ["Temporal_syn", w + " " + arg1_text, arg2_text, arg1, arg2]
            Explicit_ins = ins
            pass
        if p == "2":
            ins = ["Temporal_syn", arg1_text, w + " " + arg2_text, arg1, arg2]
            Explicit_ins = ins
            pass
        # Implicit
        imp_connectives = ["and/2", "as/1"]
        con_word = imp_connectives[random.randint(0,len(imp_connectives)-1)]
        w, p = con_word.split('/')
        if p == "1":
            ins = ["Temporal_syn" , w + " " + arg1_text , arg2_text, arg1, arg2]
            Implicit_ins = ins
            pass
        if p == "2":
            ins = ["Temporal_syn", arg1_text , w + " " + arg2_text, arg1, arg2]
            Implicit_ins = ins
        pass

    # Expansion
    if relation == "Expansion":
        arg1_text = data_to_text_generation(arg1)
        arg2_text = data_to_text_generation(arg2)
        # Explicit
        exp_connectives = [ "moreover/2", "then/2", "and/2" ]
        con_word = exp_connectives[random.randint(0,len(exp_connectives)-1)]
        w, p = con_word.split('/')
        if p == "1":
            ins = ["Expansion", w + " " + arg1_text, arg2_text, arg1, arg2]
            Explicit_ins = ins
            pass
        if p == "2":
            ins = ["Expansion", arg1_text, w + " " + arg2_text, arg1, arg2]
            Explicit_ins = ins
            pass
        # Implicit
        imp_connectives = ["and/2", "as/1"]
        con_word = imp_connectives[random.randint(0,len(imp_connectives)-1)]
        w, p = con_word.split('/')
        if p == "1":
            ins = ["Expansion" , w + " " + arg1_text , arg2_text, arg1, arg2]
            Implicit_ins = ins
            pass
        if p == "2":
            ins = ["Expansion", arg1_text , w + " " + arg2_text, arg1, arg2]
            Implicit_ins = ins
            pass
    
    # Comparison
    if relation == "Comparison":
        arg1_text = data_to_text_generation(arg1)
        arg2_text = data_to_text_generation(arg2)
        # Explicit
        exp_connectives = ["but/2", "however/2", "while/2", "although/1", "on the other hand/2"]
        con_word = exp_connectives[random.randint(0,len(exp_connectives)-1)]
        w, p = con_word.split('/')
        if p == "1":
            ins = ["Comparison", w + " " + arg1_text, arg2_text, arg1, arg2]
            Explicit_ins = ins
            pass
        if p == "2":
            ins = ["Comparison", arg1_text, w + " " + arg2_text, arg1, arg2]
            Explicit_ins = ins
            pass
        # Implicit
        imp_connectives = ["and/2","as/1"]
        con_word = imp_connectives[random.randint(0,len(imp_connectives)-1)]
        w, p = con_word.split('/')
        if p == "1":
            ins = ["Comparison" , w + " " + arg1_text , arg2_text, arg1, arg2]
            Implicit_ins = ins
            pass
        if p == "2":
            ins = ["Comparison", arg1_text , w + " " + arg2_text, arg1, arg2]
            Implicit_ins = ins
            pass
        pass
    return Explicit_ins, Implicit_ins

if __name__ == "__main__":

    # Step.4 monitor the event ... 
    # --> Agent state
    # --> Agent Inventory
    # --> Environment
    scene_experience_path = "../data/scene_experience/"
    # collect all the scene experiences
    exp_list = os.listdir(scene_experience_path)

    print "Collecting " , len(exp_list) , " scene experiences ... "

    whole_obs = []
    for fe in exp_list:
        whole_obs.append(monitor_event(scene_experience_path+fe))

    # Step.5 event occurrence feature temporal
    # build_temporal_clues()
    # how to build (temporal) occurence feature ? physical temporal clues 
    # how to access , how to store such information
    #
    event_correl = temporal_feature(whole_obs)
    # store the feature matrix 
    fe = open('../data/event_correl','w')
    correl_text = json.dumps(event_correl)
    fe.write(correl_text)
    fe.close()


    # stand for observation / synthesis event 
    # store the data into file 
    # Step.6 Discourse Relation Labler 
    # Human annonator
    # Discourse Relation ---
    # load in environment knowledge
    # add more knowledge and scenes

    print "Now load in environment knowledge .... "
    knows = open("../data/environment_knowledge").readlines()
    # knows = open("../data/backup/json_refine_experience_bk").readlines()
    env_knows = []
    for k in knows:
        env_knows.append(json.loads(k))

    # generate relation distrbution and store it
    store_path = "../data/discourse_relation_distribution"
    print '##################'
    print 'number of knowledge : ' , len(env_knows)
    ###
    mirco_discourse_generator(whole_obs, event_correl, env_knows, store_path)


    # Step 6.1 change the relation ratio
    # decrease the number of Concession ... 
    
    records = open(store_path).readlines()
    new_records = []

    #
    limit_size = 979
    concession_part = []

    # 
    exp_limit_size = 3834
    expansion_part = []
    for r in records:
        items = r.strip().split('\t')
        if items[0] == "Concession":
            concession_part.append(r)
        elif items[0] == "Expansion":
            expansion_part.append(r)
        else:
            new_records.append(r)

    ####
    random_concession_idx = []
    while True:
        idx = random.randint(0,len(concession_part)-1)
        if idx not in random_concession_idx:
            random_concession_idx.append(idx)
        if len(random_concession_idx) > limit_size:
            break

    random_expansion_idx = []
    while True:
        idx = random.randint(0, len(expansion_part)-1)
        if idx not in random_expansion_idx:
            random_expansion_idx.append(idx)
        if len(random_expansion_idx) > exp_limit_size:
            break

    #####
    for i in random_concession_idx:
        new_records.append(concession_part[i])
    for i in random_expansion_idx:
        new_records.append(expansion_part[i])

    records = new_records

    # recount the instance number for each different discourse relations
    count_dict = OrderedDict()
    for r in records:
        items = r.strip().split('\t')
        if items[0] not in count_dict:
            count_dict[items[0]] = 1
        else:
            count_dict[items[0]] += 1

    # 
    print "Output discourse relation distrbution ... "
    print count_dict

    # Step.7 event to NLP ... 
    # --> map the event into Language 
    # --> 
    # collect all event from experience and convert them into natural sentence
    # simple text generation via (data->language) using sentence template
    # collect all the scene experiences
    # verify the NLP part ...
    # cross check the NLP component
    Explicits = []
    Implicits = []
    for rec in records:
        exp, imp = from_rec_to_instance(rec)
        Explicits.append(exp)
        Implicits.append(imp)
        pass

    print "number of Explicit s : " , len(Explicits)
    print "number of Implicit s : " , len(Implicits)

    target_explicit_path = "../data/explicits"
    target_implicit_path = "../data/implicits"
    fe = open(target_explicit_path,'w')
    fi = open(target_implicit_path,'w')

    for exp,imp in zip(Explicits, Implicits):
        fe.write("\t".join(exp)+'\n')
        fi.write("\t".join(imp)+'\n')
        pass
    fe.close()
    fi.close()
