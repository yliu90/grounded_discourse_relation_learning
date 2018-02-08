## 
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


import pickle

from explore_reward import EReward
import xml.etree.ElementTree
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import fromstring, ElementTree, Element

import event_occurs
# 


def with_use(agent_host, use_cmd, my_mission, my_mission_record, plm):

    # Step.1 build up the game
    # Start the game and collect experience
    # notice('craft stuff with items and materials')
    # parse the input craft command
    # print "craft ... ing"
    # print craft_cmd
    # convert experience into new style 
    # this is very imortant
    # notice('craft somes stuff === '+item)
    # from state create a new state
    # print item

    print use_cmd


    tool_ = use_cmd[1][1]
    # target = action[3]
    # materials = action[3:] # a list 
    #print action

    init_scene = OrderedDict()
    init_scene['env'] = []
    init_scene['agt'] = []

    init_scene['agt'].append(tool_)
    print '######################33'
    print tool_
    tool = tool_.split('*')[0]
    # 
    my_mission.forceWorldReset()
    my_mission.observeGrid(0,0,0,2,2,2,'grid')

    my_mission = init_agent(my_mission, init_scene['agt'])

    states = []
    event_s = None
    action = None

    # 
    startMission(agent_host, my_mission, my_mission_record)
    world_state = agent_host.getWorldState()
    obs = []
    
    time_stamp = 0
    init_detect = False
    while world_state.is_mission_running:

        world_state = agent_host.peekWorldState()

        if (world_state.has_mission_begun) and (init_detect is False):
            agent_host.sendCommand('move 0')
            action = []
            if len(world_state.observations) > 0:
                world_state = agent_host.peekWorldState()            
                obs_text = json.loads(world_state.observations[-1].text)
                print 'Begin test ... '
                print obs_text
                obs_text['action'] = action
                init_detect = True
                obs.append(obs_text)

        if len(world_state.observations) > 0: 
            action = []
            toolstr = 'InventorySlot_0_item'
            toolflag = False
            # time stamp 1 record the init state
            world_state = agent_host.peekWorldState()            
            obs_text = json.loads(world_state.observations[-1].text)
            obs_text['action'] = action
            # time_stamp += 1
            # obs_text['time_stamp'] = time_stamp
            # print 'Here!'
            # print obs_text
            obs.append(obs_text)
            # 
            # print obs_text
            # tool check code
            # time stamp 2 
            # action point
            if obs_text[toolstr] == tool:
                # the right tool is in hand
                toolflag = True
                pass
            else:
                player = obs_text["Name"]
                if tool != 'air': # consider replace hand with air to unique the code
                    for i in xrange(0,39):
                        key = 'InventorySlot_'+str(i)+'_item'
                        if obs_text[key] == tool:
                            agent_host.sendCommand('swapInventoryItems 0 '+str(i))
                            time.sleep(1)
                            action.append(player + " swap "+tool+" to hand")
                    pass
                else:
                    for i in xrange(0,39):
                        key = 'InventorySlot_'+str(i)+'_item'
                        if obs_text[key] == 'air':
                            agent_host.sendCommand('swapInventoryItems 0 '+str(i))
                            time.sleep(1)
                            action.append(player + " swap "+tool+" to hand")
                    pass
                
            world_state = agent_host.peekWorldState()            
            obs_text = json.loads(world_state.observations[-1].text)
            obs_text['action'] = action
            # time_stamp += 1
            # obs_text['time_stamp'] = time_stamp
            obs.append(obs_text)


            # update video and world state
            # print 'obs text ... ==== ???'
            # print obs_text
            # print len(world_state.observations)
            # update video and world state again
            attackflag = False
            if toolflag:
                # agent attack
                agent_host.sendCommand('use 1')
                player = obs_text["Name"]
                action.append(player + " attacks "+target+" with "+tool)
                obs_text['action'] = action
                attackflag = True
                time.sleep(1)
            
            world_state = agent_host.peekWorldState()            
            obs_text = json.loads(world_state.observations[-1].text)
            obs_text['action'] = action
            # time_stamp += 1
            # obs_text['time_stamp'] = time_stamp
            obs.append(obs_text)


            # check for target block or entity to monitor the mission end 
            breakflag = False
            # update action for each time
            world_state = agent_host.peekWorldState()            
            obs_text = json.loads(world_state.observations[-1].text)
            obs_text['action'] = action
            # time_stamp += 1
            # obs_text['time_stamp'] = time_stamp
            obs.append(obs_text)

            if breakflag: # collect all breakflag here !
                break

    world_state = agent_host.peekWorldState()
    obs_text = json.loads(world_state.observations[-1].text)
    action = []
    player = obs_text['Name']
    action.append(player + " quit the scene")
    obs_text['action'] = action
    obs.append(obs_text)

    # previous state, action, next state unit state
    return [use_cmd, obs]



def with_craft(agent_host, craft_cmd, my_mission, my_mission_record, plm):

    # Step.1 build up the game
    # Start the game and collect experience
    # notice('craft stuff with items and materials')

    # parse the input craft command
    # print "craft ... ing"
    # print craft_cmd

    # convert experience into new style 
    # this is very imortant

    # notice('craft somes stuff === '+item)
    # from state create a new state
    # print item

    action = craft_cmd[1]
    target = action[1]
    materials = action[3:] # a list 
    print action

    init_scene = OrderedDict()
    init_scene['env'] = []
    init_scene['agt'] = []

    for item in materials:
        init_scene['agt'].append(item)


    # 
    my_mission.forceWorldReset()
    my_mission.observeGrid(0,0,0,2,2,2,'grid')

    my_mission = init_agent(my_mission, init_scene['agt'])

    states = []
    event_s = None
    action = None

    # 
    startMission(agent_host, my_mission, my_mission_record)
    world_state = agent_host.getWorldState()
    obs = []

    time_stamp = 0
    init_detect = False

    while world_state.is_mission_running:
        world_state = agent_host.peekWorldState()

        if (world_state.has_mission_begun) and (init_detect is False):
            agent_host.sendCommand('move 0')
            action = []
            if len(world_state.observations) > 0:
                world_state = agent_host.peekWorldState()
                obs_text = json.loads(world_state.observations[-1].text)
                print 'Begin test ... '
                print obs_text
                obs_text['action'] = action
                init_detect = True
                obs.append(obs_text)

        if len(world_state.observations) > 0:
            action = []
            toolstr = 'InventorySlot_0_item'
            toolflag = False

            # time stamp 1 record the init state
            world_state = agent_host.peekWorldState()
            obs_text = json.loads(world_state.observations[-1].text)
            obs_text['action'] = action
            obs.append(obs_text)

            # craft command does not need to swap tools
            if True:
                print target
                agent_host.sendCommand('craft '+str(target))
                player = obs_text["Name"]
                action.append(player + " craft "+target+ " with "+ " ".join(materials))
                obs_text['action'] = action
                craftflag = True
                time.sleep(1)

            # check for target / inventory item
            breakflag = False

            if len(obs) > 1:
                pre_ob = obs[-2]
                cur_ob = obs[-1]
                print '#############################333'
                events_detect = event_occurs.dynamic_event_detect(pre_ob, cur_ob)

                # detect target event happens
                for e in events_detect:
                    items = e.split()
                    if "gains" in items and target in items:
                        # mission finish 
                        time.sleep(1)
                        breakflag = True
            if breakflag:
                break
        
    world_state = agent_host.peekWorldState()
    obs_text = json.loads(world_state.observations[-1].text)
    action = []
    player = obs_text["Name"]
    action.append(player + " quit the scene")
    obs_text['action'] = action
    obs.append(obs_text)

    # previous state, action, next state unit state
    return [craft_cmd, obs]





# 
# send command to virtual environment to run the game
# 
def exp_command(cmd):
    inps, exps, outs = cmd
    rewarder = EReward()
    agent_host = MalmoPython.AgentHost()
    agent_state = np.zeros(len(rewarder.index_obj))
    if "punch" in exps:
        item, block = inps.keys()
        agent_state[rewarder.obj_index[item]] = inps[item]
        p_state, behavior, n_state, flag = with_punch(agent_host, rewarder, agent_state, item, block)
        p_dict = overall_state(rewarder, p_state)
        n_dict = overall_state(rewarder, n_state)
    if "craft" in exps:
        for item in inps:
            agent_state[rewarder.obj_index[item]] = inps[item]
        p_state, behavior, n_state = with_craft(agent_host, rewarder, agent_state, outs)
        p_dict = overall_state(rewarder, p_state)
        n_dict = overall_state(rewarder, n_state)
    return p_dict, n_dict 


def state_to_agent(rewarder, my_mission, state):
    strxml = my_mission.getAsXML(True)
    # print "############my_mission###############"
    # print strxml

    root = ET.fromstring(strxml)

    # compute the state for agent
    tmp = OrderedDict()
    for i,n in enumerate(state):
        if n != 0:
            tmp[rewarder.index_obj[i]] = int(n)
    init_item = []
    init_slot = 35
    if len(tmp) != 0:
        for item_name in tmp:
            if item_name == 'hand':
                # init_slot = init_slot - 1
                # do nothing
                continue # 
            sn = tmp[item_name]/64
            sl = tmp[item_name]%64
            if tmp[item_name] > 64:
                # loop for 64
                for i in sn:
                    t = {'slot':str(init_slot), 'type':item_name, 'quantity':"64"}
                    init_item.append(t)
                    init_slot = init_slot - 1
                    if init_slot < 9:
                        raise Exception('init_slot Error')
                        break
            t = {'slot':str(init_slot),'type':item_name,'quantity':str(sl)}
            init_item.append(t)
            init_slot = init_slot - 1
            if init_slot < 9:
                raise Exception('init_slot Error')
                break
            pass

        # add it into agent mission configuration
        for child in root.iter('{http://ProjectMalmo.microsoft.com}AgentStart'):
            child.append(Element('{http://ProjectMalmo.microsoft.com}Inventory'))
            for c in child:
                if "Inventory" in c.tag:
                    # pass 9-35
                    for i_ in init_item:
                        c.append(Element('{http://ProjectMalmo.microsoft.com}InventoryItem',i_))
    xmlstr = ET.tostring(root, encoding='utf8', method='xml')
    my_mission = MalmoPython.MissionSpec(xmlstr,True)
    
    return my_mission


def init_agent( my_mission, agent_init_inv):

    strxml = my_mission.getAsXML(True)
    # print "############my_mission###############"
    # print strxml
    # print agent_init_inv
    root = ET.fromstring(strxml)

    # to string and rebuild it from string of XML

    # compute the state for agent
    tmp = OrderedDict()

    for item in agent_init_inv:
        pair = item.split('*')
        tmp[pair[0]] = int(pair[1])

    # print '*****************'
    # print tmp

    init_item = []
    init_slot = 35

    if len(tmp) != 0:
        for item_name in tmp:
            if item_name == 'hand':
                # init_slot = init_slot - 1
                # do nothing
                continue # 
            sn = tmp[item_name]/64
            sl = tmp[item_name]%64
            if tmp[item_name] > 64:
                # loop for 64
                for i in range(sn):
                    t = {'slot':str(init_slot), 'type':item_name, 'quantity':"64"}
                    init_item.append(t)
                    init_slot = init_slot - 1
                    if init_slot < 9:
                        raise Exception('init_slot Error')
                        break
            t = {'slot':str(init_slot),'type':item_name,'quantity':str(sl)}
            init_item.append(t)
            init_slot = init_slot - 1
            if init_slot < 9:
                raise Exception('init_slot Error')
                break
            pass

    
        # add it into agent mission configuration
        for child in root.iter('{http://ProjectMalmo.microsoft.com}AgentStart'):
            child.append(Element('{http://ProjectMalmo.microsoft.com}Inventory'))
            for c in child:
                if "Inventory" in c.tag:
                    # pass 9-35
                    for i_ in init_item:
                        c.append(Element('{http://ProjectMalmo.microsoft.com}InventoryItem',i_))

    xmlstr = ET.tostring(root, encoding='utf8', method='xml')
    my_mission = MalmoPython.MissionSpec(xmlstr,True)
    
    return my_mission


def with_punch(agent_host, punch_cmd, my_mission, my_mission_record, plm):

    # Step.1 build up the game
    # Start game and collect experience 
    # notice('with tool punch block')

    # parse the input punch command
    # print "punch ... ing "
    # print punch_cmd

    # convert experience into new style envir/block & agent/item
    # this is very important
    action = punch_cmd[1]

    env_block = action[1] 
    agt_item = action[3]

    # start point of action
    init_scene = OrderedDict()
    init_scene['env'] = []
    init_scene['agt'] = []

    init_scene['agt'].append(agt_item)
    init_scene['env'].append(env_block)

    # end point of action
    end_state = punch_cmd[2]
    end_scene = OrderedDict()
    end_scene['env'] = []
    end_scene['agt'] = []

    # print '########'
    # print end_state

    for item in end_state:
        end_scene['agt'].append(item+'*'+str(end_state[item]))

    # 
    # test the new form
    new_scene = [init_scene, action, end_scene]
    # env_target = 

    # 
    target = init_scene['env'][0].split('*')[0]
    tool = action[-1].split('*')[0]
    # print 'punch => ', target, 'with : ', item
    if tool == 'hand':
        tool = 'air'
    behavior = ['punch', tool, target]
    # =========== # build the scene # with script 
    # world_state = agent_host.getWorldState() # get from world_state
    # print '##########'
    # print world_state.observations[-1].text
    # planetbox = ['wheat']

    # environment object
    # if block not in planetbox:
    #     my_mission.drawBlock(5,5,5, block)
    # if block in planetbox:
    #     my_mission.drawBlock(5,4,5, block)

    # Step.2 build up the basic scene
    # init the environment scene ! 
    my_mission.forceWorldReset() # force the world to reset
    my_mission.observeGrid(0,0,0,2,2,2,'grid')
    # my_mission.observeHotBar()


    # block or entity
    # print "PLM"
    blockflag = None
    for item in plm.types_dict:
        if item == "EntityTypes":
            for word in plm.types_dict[item]:
                if target == word.lower():
                    blockflag = "EntityTypes"
                    target = word
        if item == "BlockType":
            for word in plm.types_dict[item]:
                if target == word.lower():
                    blockflag = "BlockType"
                    target = word

    if blockflag == None:
        print 'target is ', target , 'wrong types ,.,..'
        return None
        # raise ValueError;
    print "air"
    print 'target == >' , target

    # print blockflag # Block and Entity is totally different!
    # block = "Stone"
    """
    # if blockflag == "BlockType":
    if False:
    # if True:
        # Block 
        dropbox = ['sand', 'gravel']
        if block in dropbox:
            for i in range(5, 5+10):
                my_mission.drawBlock(5,i,5, str(block))
        else:
            my_mission.drawBlock(5,5,5, str(block))
            
        my_mission.drawBlock(4,5,5, 'stone')
        my_mission.drawBlock(6,5,5, 'stone')
        if block in dropbox:
            my_mission.drawBlock(5,4,5, 'stone')
        my_mission.drawBlock(5,4,6, 'stone')
        my_mission.startAtWithPitchAndYaw(5.5,4,4,0,0)
        # my_mission.endAt(5,4,6,1)
        strxml = my_mission.getAsXML(True)
        root = ET.fromstring(strxml)
        MalmoPython.MissionSpec(strxml,True)
    """

    # Entity or Block is fine
    # update the block into Entity?
    # if blockflag == "EntityTypes":
    # if blockflag == "BlockType":
    if True:

        # print block
        # build up the fence or ironblock using stone 
        # pig x5,y5,z5
        # build fence to limit the move
        fence = 'sand'

        my_mission.drawCuboid(-0,4,-0,10,4,10,'stone')
        my_mission.drawCuboid(-0,4,-0,10,9,-0,'sand')
        my_mission.drawCuboid(-0,4,-0,-0,9,10,'sand')
        my_mission.drawCuboid(-0,4,10,10,9,10,'sand')
        my_mission.drawCuboid(10,4,-0,10,9,10,'sand')

        """
        my_mission.drawBlock(5,4,6,fence)
        my_mission.drawBlock(4,4,6,fence)
        my_mission.drawBlock(6,4,6,fence)
        
        my_mission.drawBlock(5,5,6,fence)
        my_mission.drawBlock(4,5,6,fence)
        my_mission.drawBlock(6,5,6,fence)
        
        # my_mission.drawBlock(5,4,6,fence)
        # around side
        my_mission.drawBlock(4,4,5,fence)
        my_mission.drawBlock(6,4,5,fence)
        
        my_mission.drawBlock(4,5,5,fence)
        my_mission.drawBlock(6,5,5,fence)
        
        my_mission.drawBlock(4,4,4,fence)
        my_mission.drawBlock(6,4,4,fence)
        
        my_mission.drawBlock(4,5,4,fence)
        my_mission.drawBlock(6,5,4,fence)
        
        my_mission.drawBlock(4,4,3,fence)
        my_mission.drawBlock(6,4,3,fence)        

        my_mission.drawBlock(4,5,3,fence)
        my_mission.drawBlock(6,5,3,fence)


        # back of wall
        my_mission.drawBlock(5,4,2,fence)
        my_mission.drawBlock(4,4,2,fence)
        my_mission.drawBlock(6,4,2,fence)
        
        my_mission.drawBlock(5,5,2,'iron_bars')
        my_mission.drawBlock(4,5,2,'iron_bars')
        my_mission.drawBlock(6,5,2,'iron_bars')

        # build orak floor?
        my_mission.drawBlock(5,3,5,fence)
        my_mission.drawBlock(5,3,4,fence)
        my_mission.drawBlock(5,3,3,fence)
        """
        # 
        if blockflag == "BlockType":
            my_mission.drawBlock(5,4,5,target)
            # print 'skip the block and directly test the entity ... '
            # return None

        if blockflag == "EntityTypes":
            strxml = my_mission.getAsXML(True)
            root = ET.fromstring(strxml)
            # load in the Entity
            t = {'pitch':str(0),'type':target,'x':"5.5","xVel":"0","yaw":"0","y":"5","yVel":"0","z":"5","zVel":"0"}
            init_item = []
            init_item.append(t)
            for child in root.iter('{http://ProjectMalmo.microsoft.com}ServerHandlers'):
                edd = Element('{http://ProjectMalmo.microsoft.com}DrawingDecorator')
                edd.append(Element('{http://ProjectMalmo.microsoft.com}DrawEntity',t))
                child.append(edd)
            
            xmlstr = ET.tostring(root, encoding='utf8', method='xml')
            my_mission = MalmoPython.MissionSpec(xmlstr,True)

    print '++++++++++++++++++++++'
    print '++++++++++++++++++++++'
    print '++++++++++++++++++++++'
    print '++++++++++++++++++++++'
    
    print 'Now running ... ' , punch_cmd 
    
    # Step.3 init the agent inventory
    # init the agent inventory ! 
    # print "####################"
    # print init_scene['agt']
    my_mission = init_agent(my_mission, init_scene['agt'])
    # my_mission = state_to_agent(rewarder, my_mission, state)

    print '#####################'
    print '#####################'
    print '#####################'
    # print my_mission.getAsXML(True)

    states = []
    event_s = None
    action = None

    # my_mission.forceWorldReset()
    states = []
    event_s = None
    action_ = None

    # Step.4 run the scene(script) and collect experience 
    # 
    # ======== # start the Mission # with the scene # ===== #
    # agent_host, my_mission, my_mission_record = setup_env(params)
    startMission(agent_host, my_mission, my_mission_record)
    world_state = agent_host.getWorldState()
    obs = []
    # the fence already limit the mob , kill it and then move forward ... 
    # tool

    # make sure the time stamp record the sense states

    time_stamp = 0
    init_detect = False
    while world_state.is_mission_running:

        world_state = agent_host.peekWorldState()

        if (world_state.has_mission_begun) and (init_detect is False):
            agent_host.sendCommand('move 0')
            action = []
            if len(world_state.observations) > 0:
                world_state = agent_host.peekWorldState()            
                obs_text = json.loads(world_state.observations[-1].text)
                print 'Begin test ... '
                print obs_text
                obs_text['action'] = action
                init_detect = True
                obs.append(obs_text)

        

        if len(world_state.observations) > 0: 
            action = []
            toolstr = 'InventorySlot_0_item'
            toolflag = False
            # time stamp 1 record the init state
            world_state = agent_host.peekWorldState()            
            obs_text = json.loads(world_state.observations[-1].text)
            obs_text['action'] = action
            # time_stamp += 1
            # obs_text['time_stamp'] = time_stamp
            # print 'Here!'
            # print obs_text
            obs.append(obs_text)


            # 
            # print obs_text
            # tool check code
            # time stamp 2 
            # action point
            if obs_text[toolstr] == tool:
                # the right tool is in hand
                toolflag = True
                pass
            else:
                if tool != 'air': # consider replace hand with air to unique the code
                    for i in xrange(0,39):
                        key = 'InventorySlot_'+str(i)+'_item'
                        if obs_text[key] == tool:
                            agent_host.sendCommand('swapInventoryItems 0 '+str(i))
                            time.sleep(1)
                            player = obs_text["Name"]
                            action.append(player + " swap "+tool+" to hand")
                    pass
                else:
                    for i in xrange(0,39):
                        key = 'InventorySlot_'+str(i)+'_item'
                        if obs_text[key] == 'air':
                            agent_host.sendCommand('swapInventoryItems 0 '+str(i))
                            time.sleep(1)
                            player = obs_text["Name"]
                            action.append(player + " swap "+tool+" to hand")
                    pass
                
            world_state = agent_host.peekWorldState()            
            obs_text = json.loads(world_state.observations[-1].text)
            obs_text['action'] = action
            # time_stamp += 1
            # obs_text['time_stamp'] = time_stamp
            obs.append(obs_text)


            # update video and world state
            # print 'obs text ... ==== ???'
            # print obs_text
            # print len(world_state.observations)
            # update video and world state again
            attackflag = False
            if toolflag:
                # agent attack
                agent_host.sendCommand('attack 1')
                player = obs_text["Name"]
                action.append(player + " attacks "+target+" with "+tool)
                obs_text['action'] = action
                attackflag = True
                time.sleep(1)
            
            world_state = agent_host.peekWorldState()            
            obs_text = json.loads(world_state.observations[-1].text)
            obs_text['action'] = action
            # time_stamp += 1
            # obs_text['time_stamp'] = time_stamp
            obs.append(obs_text)


            # check for target block or entity to monitor the mission end 
            breakflag = False
            if blockflag == "EntityTypes":
                # detect nearest entity
                # print obs_text['near_entities']
                entflag = False
                for ent in obs_text['near_entities']:
                    if ent['name'] == target:
                        entflag = True
                
                if not entflag:
                    agent_host.sendCommand('move 0.5') # move forward to collect drops 
                    player = obs_text["Name"]
                    action.append(player + " moves forward")
                    obs_text['action'] = action
                    time.sleep(1)
                    breakflag = True
                pass
            
            if blockflag == "BlockType":
                # print '#####'
                # print obs_text
                entflag = False
                if obs_text['grid'][6] == target:
                    entflag = True
                if not entflag:
                    agent_host.sendCommand('move 0.5')
                    player = obs_text["Name"]
                    action.append(player + " moves forward")
                    obs_text['action'] = action
                    time.sleep(1)
                    breakflag = True
                pass
            # update action for each time
            world_state = agent_host.peekWorldState()            
            obs_text = json.loads(world_state.observations[-1].text)
            obs_text['action'] = action
            # time_stamp += 1
            # obs_text['time_stamp'] = time_stamp
            obs.append(obs_text)


            if breakflag: # collect all breakflag here !
                break

    # detect and collect the state
    world_state = agent_host.peekWorldState()            
    obs_text = json.loads(world_state.observations[-1].text)
    # print 'obs text ... ==== ???'
    # print obs_text
    # print len(world_state.observations)
    agent_host.sendCommand('quit')

    player = obs_text["Name"]
    action = []
    action.append(player + " quit the scene")
    obs_text['action'] = action
    # time_stamp += 1
    # obs_text['time_stamp'] = time_stamp
    obs.append(obs_text)
    
    print '++++++++++++++++'
    print '+++++OBS++++++++'
    print '++++++++++++++++'

    print '',len(obs)
    for e in obs:
        print '#####'
        print obs

    plm.count += 1
    print '---- ', str(plm.count) , 'th scene running ---- '
    print '###############3'
    print '###############3'

    # 


    event_e = None
    # next_state = states[-1]

    # if overall_state(rewarder, state) == overall_state(rewarder, next_state):
    #     flag = False
    # else:
    #     flag = True
    state = None
    next_state = None
    flag = None
    
    # return missionflag, obs
    missionflag = True
    package = [punch_cmd, obs]
    return package



def excute_scene(exp,plm,agent_config):

    # total action from agent (id , action , experience ) // { craft attack use }
    # scene creation 

    agent_host = MalmoPython.AgentHost()

    # default scene path
    scene_path = agent_config
    rewarder = None
    state = None
    item = 'iron_axe'
    block = 'log'

    target_path = "../data/scene_experience/"
    agent_name = agent_config.split("/")[-1].split(".")[0]

    print "Agent_name : ", agent_name

    # common mission setting 
    my_mission_record = MalmoPython.MissionRecordSpec()

    print "Now read configuration from " , scene_path
    my_mission = MalmoPython.MissionSpec(open(scene_path).read(), True)
    list_exp = json.loads(exp)
    print list_exp
    if "punch" in list_exp[1]:
        # punch is the most complicated action in environment 

        """
        # skip 
        file_name = ('_').join(list_exp[1])
        files_dir = os.listdir(target_path)
        for fn in files_dir:
            fns = fn.split('#')
            if fns[0] == file_name:
                return 0
        """

        
        scene_record = with_punch(agent_host, list_exp, my_mission, my_mission_record, plm)
        # 
        print "package ... "
        #
        file_name = ('_').join(list_exp[1])
        file_name = agent_name + "#" + file_name
        files_dir = os.listdir(target_path)
        count = 0
        for fn in files_dir:
            fns = fn.split('#')
            if "#".join(fns[:-1]) == file_name:
                count += 1

        print '#############%%%%%%%%%%%%%%%%%%%%'
        print target_path+file_name+"#"+str(count)
        fp = open(target_path+file_name+'#'+str(count),'w')
        # search for file name 
        json_txt = json.dumps(scene_record)
        fp.write(json_txt)
        fp.write('\n')
        fp.close()
        pass

    if "craft" in list_exp[1]:

        """
        file_name = ('_').join(list_exp[1])
        files_dir = os.listdir(target_path)
        for fn in files_dir:
            fns = fn.split('#')
            if fns[0] == file_name:
                return 0
        """

        scene_record = with_craft(agent_host, list_exp, my_mission, my_mission_record, plm)
        print 'package ... '
        # skip 
        file_name = ('_').join(list_exp[1])
        file_name = agent_name + '#' + file_name
        # Adam#craft_activator_rail_with_iron_ingot*6_stick*2_redstone_torch*1
        files_dir = os.listdir(target_path)
        count = 0
        for fn in files_dir:
            fns = fn.split('#')
            if "#".join(fns[:-1]) == file_name:
                count += 1
        print '#############%%%%%%%%%%%%%%%%%%%%'
        print target_path+file_name+"#"+str(count)
        fp = open(target_path+file_name+'#'+str(count),'w')
        # search for file name
        json_txt = json.dumps(scene_record)
        fp.write(json_txt)
        fp.write('\n')
        fp.close()
        pass

    """
    if "use" in list_exp[1]:
        scene_record = with_use(agent_host, list_exp, my_mission, my_mission_record, plm)
        pass
    """

    #  record the scene_record into file

    pass


def parse_Types_xsd(path):

    # collecting different types in environment
    e = xml.etree.ElementTree.parse('../data/Types_refine.xsd').getroot()
    types = OrderedDict()
    for child in e:
        types[child.attrib['name']] = [] 
        a = child
        for i_child in child:
            if i_child.tag == "{http://www.w3.org/2001/XMLSchema}restriction":
                for ii_c in i_child:
                    types[child.attrib['name']].append(ii_c.attrib['value'])

    # types is a dict of game objects types and items name
    return types


def scene_play(path,plm):
    scenes = open(path,'r').readlines()
    exp_s = []
    agent_configs = []
    agent_configs.append("../data/Adam.xml")
    agent_configs.append("../data/Mary.xml")
    # do some filter
    # Step.2

    # random shuffle
    # random.shuffle(scenes)
    for s in scenes:
        # excute the agent action via (api command)
        # update 1 : agent_id_owns/ environment_owns/ is different, which is not apply to the fact
        # take scene as input , excute the scene , collect the episodic experience
        # update 2  

        inner_loop = 10;
        # inner_loop = 3;
        for i in range(inner_loop):
            for agent in agent_configs:
                excute_scene(s,plm,agent)
        pass
    pass


class platform():

    def __init__(self):
        self.types_dict = parse_Types_xsd(None)
        self.count = 0
        pass

    def isEntitiesOrBlock(self,target):
        matched_word = None
        blockflag = None
        for item in self.types_dict:
            if item == "EntityTypes":
                for word in self.types_dict[item]:
                    if target.lower() == word.lower():
                        blockflag = "EntityTypes"
                        matched_word = word
            if item == "BlockType":
                for word in self.types_dict[item]:
                    if target.lower() == word.lower():
                        blockflag = "BlockType"
                        matched_word = word
        return blockflag, matched_word


if __name__ == "__main__":

    # step 1 parse the text knowledge in single step
    # 
    # skip from step 1
    # build_json_event(org_path, json_path)

    # Step 2, Build up Scene and Collect the Physical Obs (event temporal information )
    # verify for standard experience 2017-11-17
    plm = platform()
    types_dict = parse_Types_xsd(None)
    # program control > 
    
    # Step.3 unpackage the experience and scene into a json object and sersizlize the data as model feature training materials
    # experience including orginial scene script and interaction experience  
    # scene script description
    # the example experience contains 
    # --> 1. scene description
    # --> 2. mission interaction eposides experiences
    #
    
    scene_script_path = "../data/environment_knowledge"
    scene_play(scene_script_path, plm)
    
