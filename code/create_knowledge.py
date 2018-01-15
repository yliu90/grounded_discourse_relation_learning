import copy
from collections import OrderedDict
import json

def punch_code():

    # case1 animals
    verb = ['punch']
    tuples = []
    materials = ['wooden_','stone_','iron_','diamond_']
    tools = ['sword','axe','pickaxe','hoe','shovel']
    tools.append('air')
    # perp
    prep = ['with']
    # purpose
    # forbidden tool, target, gains
    e1 = [ [], ['cow*1'], ['beef*1','leather*0.5'] ]
    e2 = [ [], ['sheep*1'], ['mutton*1','wool*0.5'] ]
    e3 = [ [], ['pig*1'], ['porkchop*1'] ]
    e4 = [ [], ['chicken*1'], ['chicken_meat*1','feather*0.5'] ]
    e5 = [ [], ['rabbit*1'], ['rabbit_meat*0.5','rabbit_hide*0.5'] ]
    e6 = [ [], ['red_mushrooms*1'] ,['red_mushroom*1'] ]
    e7 = [ [], ['brown_mushrooms*1'] ,['brown_mushroom*1'] ]
    e8 = [ [], ['tallgrass*1'] ,['wheat_seeds*0.5'] ]
    e9 = [ [], ['wheat*1'] ,['wheat_seeds*1'] ]
    e10 = [ [], ['melons*1'] ,['melon*1'] ]
    e11 = [ [], ['potatoes*1'] ,['potato*1'] ]
    e12 = [ [], ['sugar_canes*1'] ,['sugar_cane*1'] ]
    e13 = [ [], ['pumpkin*1'] ,['pumpkin*1'] ]
    e14 = [ [], ['skeleton*1'] ,['bone*0.5','bow*0.5','arrow*0.5'] ]
    e15 = [ [], ['zombie*1'] ,['rotten_flesh*0.5', 'carrots*0.5', 'potatoes*0.5'] ]
    e15 = [ [], ['spider*1'] ,['string*0.5', 'spider_eye*0.5'] ]
    e16 = [ [], ['carrots*1'] ,['carrot*1'] ]
    e17 = [ [], ['log1*1'] ,['log*1'] ]
    e18 = [ [], ['log2*1'] ,['log*1'] ]
    e19 = [ [], ['log2*1'] ,['log*1'] ]
    e20 = [ [[],['axe','sword','hand','pickaxe','hoe']], ['gravel*1'] ,['gravel*1','flint*0.5'] ]
    e21 = [ [], ['sand*1'] ,['sand*1'] ]
    e22 = [ [], ['dirt*1'] ,['dirt*1'] ]
    e23 = [ [[],['axe','sword','hoe','hand','shovel']], ['cobblestone*1'] ,['cobblestone*1'] ]
    e24 = [ [[],['axe','sword','hoe','hand','shovel']], ['coal_ore*1'] ,['coal*1']]
    e25 = [ [['wooden_'],['axe','sword','hoe','hand','shovel']], ['iron_ore*1'] ,['iron_ore*1'] ]
    e26 = [ [['wooden_','stone_'],['axe','sword','hoe','hand','shovel']], ['gold_ore*1'] ,['gold_ore*1'] ]
    e27 = [ [['wooden_','stone_'],['axe','sword','hoe','hand','shovel']], ['diamond_ore*1'] ,['diamond_ore*1'] ]
    e28 = [ [['wooden_','stone_'],['axe','sword','hoe','hand','shovel']], ['redstone_ore*1'] ,['redstone*1'] ]
    e29 = [ [['wooden_','stone_'],['axe','sword','hoe','hand','shovel']], ['lit_redstone_ore*1'] ,['redstone*1'] ]
    e30 = [ [], ['horse*1'] ,['leather*0.5'] ]

    # load in all script meta-data
    total_exps = []
    total_punch_sens = []
    for i in range(30):
        index = i+1
        total_exps.append(eval("e"+str(index)))
        pass

    def expand_words(left,right):

        total = []
        for l in left:
            for r in right:

                if type(l) is str:
                    tmp = [copy.deepcopy(l)]
                    tmp.append(copy.deepcopy(r))
                    total.append(tmp)
                if type(l) is list:
                    tmp = copy.deepcopy(l)
                    tmp.append(copy.deepcopy(r))
                    total.append(tmp)
        return total
    # [ [['wooden_'],['axe','sword','hoe','hand','shovel']], ['iron_ore*1'] ,['iron_ore*1'] ]
    # [{}, ["punch", "log2*1", "with", "hand*1"], {"log2": 1}]
    # ["punch", "log2*1", "with", "hand*1"]

    for exp in total_exps:
        tmp = []
        # name
        # verb punch
        tmp.append(verb[0])

        # target exp[1]
        target = exp[1][0]
        tmp.append(target)
        tmp.append(prep[0])

        # filter for forbidden list
        if len(exp[0]) == 0:
            m_ = copy.deepcopy(materials)
            t_ = copy.deepcopy(tools)
            m_tools = expand_words(m_,t_)
            m_t_set = []
            for i in m_tools:
                if i[1] == 'air':
                    m_t_set.append(i[1])
                else:
                    m_t_set.append(i[0]+i[1])
            m_t_set = list(set(m_t_set))

        else:
            m_ = list(set(materials) - set(exp[0][0]))
            t_ = list(set(tools) - set(exp[0][1]))
            m_tools = expand_words(m_,t_)
            m_t_set = []
            for i in m_tools:
                if i[1] == "air":
                    m_t_set.append(i[1])
                else:
                    m_t_set.append(i[0]+i[1])

            m_t_set = list(set(m_t_set))
        # tool for 1
        m_t_set = [m+'*1' for m in m_t_set] 

        for mt in m_t_set:
            tmp_ = copy.deepcopy(tmp)
            tmp_.append(mt)

            # pack up the new experience
            # action 
            # start status notice the start status and end status only describe the agent state
            # not includes the environment status in this version
            # end status
            start_state = dict()
            item, num = tmp_[3].split('*')
            if item != "air":
                start_state[item] = float(num)

            end_state = dict()
            for r in exp[2]:
                item, num = r.split('*')
                end_state[item] = float(num)

            # meanwhile keep the tool in the bag
            for k in start_state:
                end_state[k] = start_state[k]
            total_punch_sens.append([start_state, tmp_, end_state])

    return total_punch_sens



def main():
    # generate punch knowledge by meta_data
    punchs = punch_code()

    # merge together use json (punch_knowledge, craft_knowledge)
    # punch_knowledge = open('../data/tool/punch_knowledge').readlines()
    punch_knowledge = [json.dumps(t)+'\n' for t in punchs]
    craft_knowledge = open('../data/tool/craft_knowledge').readlines()

    tmp = []
    tmp.extend(punch_knowledge)
    tmp.extend(craft_knowledge)

    # store as each line
    f = open('../data/environment_knowledge','w')
    for t in tmp:
        f.write(t)
    f.close()



if __name__ == '__main__':
    main()
