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


def startMission(agent_host, my_mission, my_mission_record):
    max_retries = 3
    for retry in range(max_retries):
        try: 
            agent_host.startMission(my_mission, my_mission_record)
            break
        except RuntimeError as e:
            if retry == max_retries -1:
                print "Error starting mission :", e
                exit(1)
            else:
                time.sleep(5)
    print 'Waiting for the mission to start ... '
    world_state = agent_host.peekWorldState()
    print '### check is mission running ... '
    while not world_state.has_mission_begun:
        sys.stdout.write(".")
        time.sleep(1)
        world_state = agent_host.peekWorldState()
        for error in world_state.errors:
            print "Errors : " , error.text
    print 
    print "Mission running ... "
    pass

# 
def notice(tip):

    print '###########################################'
    print '#>>>>>>' , tip
    print '###########################################'
    pass


def load_mission(mission_file):

    with open(mission_file, 'r') as f:
        mission_xml = f.read()
        my_mission = MalmoPython.MissionSpec(mission_xml,True)
        print 'read xml to generate the mission'
    return my_mission


def setup_env(params):

    notice('\nBuild up the agent host server ... \n '+ 'Build up the Mission Record ... ')

    agent_host = MalmoPython.AgentHost()
    # agent_host.setRewardsPolicy(MalmoPython.RewardsPolicy.KEEP_ALL_REWARDS)

    my_mission_record = MalmoPython.MissionRecordSpec()

    notice('\nnow build server and host ... from '+params['mission'])

    
    mission_file = params['mission']
    my_mission = load_mission(mission_file)
    
    if my_mission.isVideoRequested(0):
        vwidth = my_mission.getVideoWidth(0)
        vheight = my_mission.getVideoHeight(0)
        vchannels = my_mission.getVideoChannels(0)
    else:
        print 'Error! Video is requested !'
        exit()
        return 0
    
    
    params['vwidth'] = vwidth
    params['vheight'] = vheight
    params['vchannels'] = vchannels
    
    
    return agent_host, my_mission, my_mission_record


