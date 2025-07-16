import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, savefig
import matplotlib
import sys
from collections import OrderedDict
import scipy.stats
import color_util
plt.switch_backend('agg')

NUM_BINS = 10000
BITS_IN_BYTE = 8.0
MILLISEC_IN_SEC = 1000.0
M_IN_B = 1000000.0
VIDEO_LEN = 48
K_IN_M = 1000.0
REBUF_P = 4.3
SMOOTH_P = 1

# labels = SCHEMES#, 'RB']
LW = 2.

SCHEMES = ['ComTreep','pitreep','pensieve','BB','robustMPC','bola','genet']


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h #m, m-h, m+h

def inlist(filename, traces):
    ret = False
    for trace in traces:
        if trace in filename:
            ret = True
            break
    return ret


def plot_qoe(pref,outputs,y_min, y_max):
    datas = []
    datas_std = []
    for idx_2 , scheme in enumerate(SCHEMES):
        bit_rates = []
        rebufs = []
        smooths = []
        qoes = []
        

        for files in os.listdir('real_world'):
            if  files.startswith(scheme+pref):
                print(files)
                bit_rate = []
                rebuf = []
                qoe = []
                file_scehem = 'real_world/' + files
                f = open(file_scehem, 'r')
                arr = []
            
                for line in f:
                    sp = line.split()
                    if len(sp) > 1:
                        # bit = 
                        rebuf.append(float(sp[3])*4.3)
                        qoe.append(float(sp[-1]))
                       
                        bit_rate.append(int(sp[1])/1000.0)
               

                    else:
                        break
            
                f.close()
                bit_rates.append(np.mean(bit_rate[1:]))
                rebufs.append(np.mean(rebuf[1:]))
                qoes.append(np.mean(qoe[1:]))
                smooths.append(np.mean(np.abs(np.diff(bit_rate))))
        print(qoes,bit_rates,rebufs,smooths)
        print([np.mean(qoes),np.mean(bit_rates),np.mean(rebufs),np.mean(smooths)])
        datas.append([np.mean(qoes),np.mean(bit_rates),np.mean(rebufs),np.mean(smooths)])
        datas_std.append([mean_confidence_interval(qoes),mean_confidence_interval(bit_rates),mean_confidence_interval(rebufs),mean_confidence_interval(smooths)])
         
        #datas_std.append([np.std(qoe),np.std(bit_rates),np.std(rebufs),np.std(smooths)])

    width = 0.16
    beta = 1.05
    ind = np.linspace(0., 3.2, 4)
    sp = np.arange(-3.6, 3.5, 1)

    def rgb_to_hex(rr, gg, bb):
        rgb = (rr, gg, bb)
        return '#%02x%02x%02x' % rgb
    edgecolors = [rgb_to_hex(237, 65, 29), '#3E85BA', rgb_to_hex(122, 122, 122), rgb_to_hex(102, 49, 160), '#FE3287', '#A76831', '#50CB93', '#FFA0A0', '#28FFBF',\
    '#00C1D4', '#B980F0', '#628395', '#787A91', '#7C83FD', '#96BAFF', '#7DEDFF', '#D54C4C', '#F08FC0', '#EDF6E5']
    hatchs = ['++++++', None, 'xxxxxxxx', '\\\\\\\\\\', '***', 'ooooo', 'OOOOO', '....', '+++++', '////', '****', \
        'ooooo', 'OOOOO', '....', '+++++', '////', '****', 'ooooo', 'OOOOO', '....', '+++++', '////', '****']
    width = 0.1  # the width of the bars

    plt.rcParams['axes.labelsize'] = 17
    font = {'size': 17}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(figsize=(9,5))
    plt.subplots_adjust(left=0.08, bottom=0.1, right=0.97, top=0.82)
    # plt.tight_layout()
    capsize = 1
    lw = 1.2
    beta = 1.05
    index = 0
    # print(datas)
    # print(datas_std)
    error_params=dict(elinewidth=1,ecolor='black',capsize=2)
    for _sp, _edgecolor, _hatch,_scheme in zip(sp, edgecolors, hatchs,SCHEMES):
        label_, color_, line_, marker_ = color_util.map_cc[_scheme]
        rects = ax.bar(ind + _sp * width * beta, datas[index], width, edgecolor= color_, hatch=_hatch, color='white',lw=lw,yerr=datas_std[index],error_kw=error_params,
                    label=label_)
        
        index += 1
                                
                                        




    ax.set_ylabel('QoE')
    ax.set_xticks(ind + 0.5 * width - 0.05)
    ax.set_xticklabels(['QoE','Bitrate','Rebuff Penalty','Smooth Penalty'])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, axis='y')
    ax.set_axisbelow(True)
    ax.legend(loc='upper center', ncol=4,
            bbox_to_anchor=(0.47, 1.2), frameon=False, fontsize=16)
    savefig(outputs + '.pdf')

   
if __name__ == '__main__':



    plot_qoe('_new_real_test_2m','img/fig6/net',0,4.3)


    