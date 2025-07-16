import json
import numpy as np
import asyncio
import fastapi_poe as fp
import sys
few_shot = bool(sys.argv[1])
top_k_flag  = bool(sys.argv[2])
log_name = sys.argv[3]
async def get_responses(api_key, messages, bot_name):
    response = ''
    async for partial in fp.get_bot_response(messages=messages, bot_name=bot_name, api_key=api_key):

        response += partial.text
  
    return response


def convert_to_decision_tree(node, indent="", features=None):
    if "prediction" in node:
        return f"{indent}return {node['prediction']}"
    
    feature_index = node["feature"]

    
    feature = features[feature_index]
    
    true_branch = convert_to_decision_tree(node["true"], indent + "    ", features)
    false_branch = convert_to_decision_tree(node["false"], indent + "    ", features)
    
    return f"{indent}if {feature} then:\n{true_branch}\n{indent}else:\n{false_branch}"


def getTrees():
    trees = []

    feature_values = ['last_quality<=0.122093022', 'last_quality<=0.22674419', 'last_quality<=0.354651153', 'last_quality<=0.831395388', 'curr_buffer<=0.97983253', 'curr_buffer<=1.01593244', 'curr_buffer<=1.0500865', 'curr_buffer<=1.06336725', 'curr_buffer<=1.09612775', 'curr_buffer<=1.10626936', 'curr_buffer<=1.16871178', 'curr_buffer<=1.17780685', 'curr_buffer<=1.21303642', 'curr_buffer<=1.30972278', 'curr_buffer<=1.33054829', 'curr_buffer<=1.36694241', 'curr_buffer<=1.40048122', 'curr_buffer<=1.47480035', 'curr_buffer<=1.90814781', 'curr_buffer<=2.03355861', 'curr_buffer<=2.28890634', 'curr_buffer<=2.42415929', 'curr_buffer<=2.46704865', 'curr_buffer<=2.63507009', 'curr_buffer<=2.6948216', 'curr_buffer<=2.9963398', 'curr_buffer<=3.10570002', 'curr_buffer<=3.24202442', 'curr_buffer<=3.49860668', 'tput_4<=0.0744417161', 'tput_5<=0.0688233823', 'tput_5<=0.0846394077', 'tput_6<=0.118215263', 'tput_7<=0.0609433018', 'tput_7<=0.0934210494', 'tput_7<=0.10941115', 'tput_7<=0.175559834', 'tput_7<=0.195492133', 'tput_7<=0.255806863', 'tput_7<=0.324703485', 'delay_7<=0.0943199024', 'tput_7<=0.15317139']
    f = open(your_path,'r')
    for line in f.readlines():
        if line[0] != '{':
            continue
        # print(line)
        line = line.replace("'", '"') 
        json_data = json.loads(line)
        decision_tree = convert_to_decision_tree(json_data, features=feature_values)


        trees.append( decision_tree )
    return trees
def makeMessage(tree1,tree2):

    few_shot_mse = ''
    if few_shot:
        few_shot_mse = 'As a developer in the field of streaming media, we typically assess the interpretability of a decision tree for bitrate adaptation algorithms in the following ways. First and foremost, we consider the number of layers in the tree; the fewer the layers, the stronger the interpretability. Secondly, we prefer features that are more intuitive. Generally speaking, we desire important features such as last_quality, buffer, and tput. Furthermore, we aspire for the decision tree to be well-organized, with features within the same layer or subtree being as consistent as possible. The aforementioned methods are merely examples of evaluating interpretability. In practice, interpretability often implies better comprehension for developers.'
    
    message = few_shot_mse +'Which of the following two decision trees has higher interpretability? Please use \'TREEONE\' as the first word in your answer to indicate that the first decision tree has higher interpretability, or use \'TREETWO\' to indicate that the second decision tree has higher interpretability, followed by an explanation.\nDecision tree 1: {0}.\nDecision tree 2: {1}.\n'.format(tree1,tree2)
 
    return message
def getBetter(tree1,tree2):
    api_key = ""
    bot_names = [] # your llm
    message = fp.ProtocolMessage(role="user", content=makeMessage(tree1,tree2))
    flag = -1
    responses = {}
    for bot_idx in range(2):
        response = asyncio.run(get_responses(api_key, [message], bot_names[bot_idx]))
    
        if top_k_flag:
            response2 = asyncio.run(get_responses(api_key, [message], bot_names[bot_idx]))
            response3 = asyncio.run(get_responses(api_key, [message], bot_names[bot_idx]))
            count_treeone = 0
                    
            if response.startswith('TREEONE'):
                count_treeone += 1
            if response2.startswith('TREEONE'):
                count_treeone += 1
            if response3.startswith('TREEONE'):
                count_treeone += 1
            response = response + response2 + response3

            
            if count_treeone >= 2:
                if flag == -1 or flag == 1:
                    flag = 1
                else:
                    flag = 0
            else:
                if flag == -1 or flag == 2:
                    flag = 2
                else:
                    flag = 0
        else:
       
            if response.startswith('TREEONE'):
                if flag==-1 or flag==1:
                    flag = 1
                else:
                    flag = 0
            elif response.startswith('TREETWO'):
                if flag==-1 or flag==2:
                    flag = 2
                else:
                    flag = 0
            else:
                flag=0
        responses[bot_names[bot_idx]] = response
    return flag, json.dumps(responses)
def battle(t1_idx,t2_idx, trees):
    flag, response = getBetter(trees[t1_idx], trees[t2_idx])
    if flag==1:
        win_idx = (t1_idx,)
    elif flag==2:
        win_idx = (t2_idx,)
    else:
        win_idx = (t1_idx,t2_idx)
    battle_log = 'Battle: tree1_idx:{0}, tree2_idx:{1}, winner_idx:{2}, responses:{3}'.format(t1_idx,t2_idx,win_idx,response)
    return win_idx,battle_log

trees = getTrees()

pool1 = [i for i in range(len(trees))]
pool2 = []
battle_f = open('log/'+log_name,'w+')
round_idx = 0
while len(pool1)>1:
    round_idx +=1
    battle_f.write('Round {0} start, pool size: {1}\n'.format(round_idx, len(pool1)))
    battle_f.flush()
    np.random.shuffle(pool1)
    if len(pool1)%2==1:
        pool2.append(pool1[-1])
    for i in range(0,len(pool1)-1,2):
        win_idx,battle_log = battle(pool1[i],pool1[i+1],trees)
        for j in win_idx:
            pool2.append(j)
        battle_f.write(battle_log+'\n')
        battle_f.flush()
    
    if len(pool1) == len(pool2):
        battle_f.write('doubel check\n')
        pool2 = []
        if len(pool1)%2==1:
            pool2.append(pool1[-2])
        for i in range(0,len(pool1)-1,2):
            win_idx,battle_log = battle(pool1[i],pool1[i-1],trees)
            for j in win_idx:
                pool2.append(j)
            battle_f.write(battle_log+'\n')
            battle_f.flush()

    if len(pool1) == len(pool2):
           battle_f.write('Round {0} end, winner: {1}\n\n'.format(round_idx,json.dumps(pool1)))
           break
    else:
        pool1 = pool2
        pool2 = []
        battle_f.write('Round {0} end, winner: {1}\n\n'.format(round_idx,json.dumps(pool1)))
        battle_f.flush()
for idx in range(len(pool1)):
    battle_f.write('final winner: {0}\nTree:{1}\n'.format(pool1[idx],trees[pool1[idx]]))
    battle_f.flush()
battle_f.close()
    
