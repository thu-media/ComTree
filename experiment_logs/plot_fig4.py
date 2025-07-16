
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib.pyplot import plot, savefig
import matplotlib
import sys
import os
import numpy as np
import color_util
SCHEMES = ['ComTreep','pensiedtp', 'bb','robustmpc','rl','llm','ghent','bola']
TEST_TRACES = os.listdir('./simulation/')
alg = {}
for test_trace in TEST_TRACES:
    print(test_trace)
    file_path = 'simulation/'+test_trace+'/'
    save_name  = test_trace 
    dirs = os.listdir(file_path)
    rewards = []
    for scheme in SCHEMES:
        reward_sch = []
        reward_this = []
        for cooked_file in dirs:
            reward_this = []
            if 'log_'+scheme in cooked_file:
                file_path_now = file_path + cooked_file
 

                with open(file_path_now, 'r') as f:
                    for line in f.readlines()[1:]:
                        line = line.strip()  
                        if line: 
                            parse = line.split()
                            reward_this.append(float(parse[-1]))
                #print(file_path_now, np.mean(reward_this[:]))
                reward_sch.append(np.mean(reward_this[:]))
                if scheme not in alg:
                    alg[scheme] = []
                alg[scheme].append(np.mean(reward_this[:]))


        rewards.append(reward_sch)

       
        
        


    LW = 3.4
    NUM_BINS = 1000000
    plt.rcParams['axes.labelsize'] = 20
    font = {'size': 18}
    matplotlib.rc('font', **font)

    fig, ax = plt.subplots(figsize=(9, 5))

    datas = rewards

    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.97, top=0.98)
    for (data,  _scheme) in zip(datas, SCHEMES):
        if _scheme == 'ComTreep':
            continue
        label_, color_, line_, marker_ = color_util.map_cc[_scheme]
        values, base = np.histogram(data, bins=NUM_BINS)
        cumulative = np.cumsum(values)
        cumulative = 1.0*cumulative / np.max(cumulative)
        print(np.mean(data),np.std(data),label_)

        ax.plot(base[:-1], cumulative, line, color=color_, lw=LW, label=label_)
    _scheme = 'ComTreep'
    data = datas[0]
    label_, color_, line_, marker_ = color_util.map_cc[_scheme]
    values, base = np.histogram(data, bins=NUM_BINS)
    cumulative = np.cumsum(values)
    cumulative = 1.0*cumulative / np.max(cumulative)
    print(np.mean(data),np.std(data),label_)

    ax.plot(base[:-1], cumulative, line, color=color_, lw=LW+0.5, label=label_)
    ax.legend(framealpha=1,
                frameon=True, fontsize=20)
    plt.yticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.xlim(-1,4.5)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)

    plt.ylabel('CDF', fontsize=20)
    ax.grid(linestyle='--', linewidth=1.5)
    plt.xlabel('QoE_lin', fontsize=20)
    savefig('img/fig4/'+save_name+'.pdf')
for scheme in SCHEMES:
    # print(scheme,alg[scheme])
    print(scheme, np.mean(alg[scheme]), np.std(alg[scheme]))