import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib.pyplot import plot, savefig
import matplotlib
import os
from get_reward import get_reward
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300] 
def box_plot(data,name):
    fig, ax = plt.subplots(figsize=(9, 6))
    plt.rcParams['axes.labelsize'] = 20
    font = {'size': 18}
    matplotlib.rc('font', **font)
    plt.subplots_adjust(left=0.1, bottom=0.05, right=0.97, top=0.98)
    

    num_boxes = len(data)
    positions = np.arange(1, num_boxes + 1) 
    positions[4:] += 1 
    positions[8:] += 1 
    width = 0.6
    

    bp = ax.boxplot(data, positions=positions, widths=width, patch_artist=True, showfliers=False,
                    boxprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(color='black', linewidth=2),
                    capprops=dict(color='black', linewidth=2),
                    medianprops=dict(color='red', linewidth=2))
    

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, (box, color) in enumerate(zip(bp['boxes'], colors * 3)):
        box.set(facecolor=color, alpha=0.5)

    ax.set_xticks([], fontsize=20)

    labels = ['Norway', 'Oboe', 'Puffer Oct.17-21', 'Puffer Feb.18-22']
    plt.ylabel('QoE Improvement Ratio', fontsize=20)
    plt.yticks(fontsize=20)
    

    category_labels = ['lin', 'hd']
    
    for i, label in enumerate(category_labels):
        ax.text((i * 4.8) + 2.4, ax.get_ylim()[1] * 0.9, label, ha='center', va='center', fontsize=30)
    

    ax.axhline(y=0, color='black', linewidth=2)

    ax.grid(linestyle='--', linewidth=1.5)

    legend_elements = [plt.Line2D([0], [0], marker='s', color='w', 
                                  markerfacecolor=colors[0], markersize=24,label='Norway', alpha=0.5),
                       plt.Line2D([0], [0], marker='s', color='w',  alpha=0.5,
                                  markerfacecolor=colors[1], markersize=24,  label='Oboe'),
                       plt.Line2D([0], [0], marker='s', color='w',  alpha=0.5,
                                  markerfacecolor=colors[2], markersize=24,  label='Puffer Oct.17-21'),
                       plt.Line2D([0], [0], marker='s', color='w',  alpha=0.5,
                                  markerfacecolor=colors[3], markersize=24,  label='Puffer Feb.18-22')]

    ax.legend(handles=legend_elements, fontsize=20,loc='lower left')
    plt.savefig('img/fig5/'+name+'.pdf')



def get_data(SCHEMES):
    TEST_TRACES = [ 'norway', 'oboe', 'puffer-2110','puffer-2202']
    reward_types = []
    reward_type_this = []
    for reward_type in ['lin',  'hd']:

        for test_trace in TEST_TRACES:

            file_path = 'simulation/'+test_trace+'/'
            save_name  = test_trace 
            dirs = os.listdir(file_path)
            rewards = []
            reward_sch_all = []
            for scheme in SCHEMES:
                reward_sch = []
                reward_this = []
                for cooked_file in dirs:
        
                    if scheme in cooked_file:
                        reward_this = []
                        file_path_now = file_path + cooked_file
                        with open(file_path_now, 'r') as f:
                            last_quality = 1
                            for line in f.readlines()[1:]:
                                line = line.strip() 
                                if line: 
                                    parse = line.split()
                                    bit_rate = int(parse[1])
                                    bit_rate = VIDEO_BIT_RATE.index(bit_rate)
                                    rebuff = float(parse[3])
                                    reward_now = get_reward(bit_rate,rebuff,last_quality,reward_type)
                                    last_quality = bit_rate
                                    reward_this.append(reward_now)
                        
                        reward_sch.append(np.mean(reward_this[:]))
                reward_sch_all.append(reward_sch)
            print(test_trace, len(reward_sch_all),np.mean(reward_sch_all[0]),np.mean(reward_sch_all[1]))
            rewards = [(a-b)/abs(b) for a,b in zip(reward_sch_all[0],reward_sch_all[1])]
            print(np.mean(rewards),np.min(rewards),np.max(rewards))
            reward_type_this.append(rewards)
   
    return reward_type_this
SCHEMES = [['ComTreep', 'rl'],['ComTreep','pensiedtp']]
names = ['rl','trl']
for scheme,name in zip(SCHEMES,names):
    data = get_data(scheme)
    box_plot(data,name)
