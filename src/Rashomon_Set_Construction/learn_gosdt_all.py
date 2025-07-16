import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
import pickle as pk
import csv
import os
import pensieve
import pensiedt_all as pensiedt
import pandas as pd
import argparse
import load_trace
import fixed_env as env
from multiprocessing import Pool
import time
import h5py
from model.gosdt import GOSDT
S_LEN = 8  # take how many frames in the past
A_DIM_P = 6
A_DIM_H = 2
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
BITRATE_REWARD = [1, 2, 3, 12, 15, 20]
HD_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
BITRATE_LEVELS = 6
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
DEFAULT_PREFETCH = 0 # default prefetch decision without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = './results'
LOG_FILE = './results/log_pensieve'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
NN_MODEL = './models/pretrain_linear_reward.ckpt'

NN_MODEL = './models/model.ckpt'


first_columns =  ['last_quality', 'curr_buffer', 'tput_0', 'tput_1', 'tput_2', 'tput_3', 'tput_4', 'tput_5', 'tput_6', 'tput_7', 'delay_0', 'delay_1', 'delay_2', 'delay_3', 'delay_4', 'delay_5', 'delay_6', 'delay_7', 'size_0', 'size_1', 'size_2', 'size_3', 'size_4', 'size_5', 'chunk_til_video_end']

def get_rollouts(env, policy, args, n_batch_rollouts, dt_policy=None):
    rollouts = []
    if dt_policy is None:
        for i in range(n_batch_rollouts):
            rollouts.extend(policy.main(args, env))
    else:
        for i in range(n_batch_rollouts):
            rollouts.extend(policy.main(args, env, dt_policy))
    return rollouts



def resample(states, actions, serials, max_pts):
    if len(states) <= 200000:
        idx = np.random.choice(len(states), size=max_pts)
    else:

        last_10000_idx = np.arange(-10000, 0)

        remaining_samples = len(states) - 10000
    

        remaining_idx = np.random.choice(remaining_samples, size=190000, replace=False)
        idx = np.concatenate((last_10000_idx, remaining_idx))
        
    return states[idx], actions[idx], serials[idx]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--abr', metavar='ABR', choices=['pensieve', 'robustmpc', 'hotdash'])
    parser.add_argument('-q', '--qoe-metric', choices=['lin', 'log', 'hd'], default='lin')
    parser.add_argument('-t', '--traces', choices=['norway', 'fcc', 'oboe'])
    parser.add_argument('-item', '--item', type=int, default=False)
    parser.add_argument('-l', '--log', action='store_true')
    args = parser.parse_args()
    n_batch_rollouts = 220
    max_iters = 100 
    pts = 200000
    train_frac = 0.2
    np.random.seed(RANDOM_SEED)

    trees = []
    precision = []
    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(args.traces)
 
    net_env = env.Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw,
                              all_file_names=all_file_names)

    time_calc = np.zeros((max_iters, 3))
    df =h5py.File('save/datac'+str(args.item), 'r') 
    states =list(df['state'])
    actions =  list(df['action'])
    serials=  list(np.array(df['serial']))
    df2 =h5py.File('save/columnsc'+str(args.item), 'r') 
    columns = [item.decode('utf-8') for item in list(df2['c'])]
    if args.abr == 'pensieve':
        teacher = pensieve.Pensieve()
        student = pensiedt.PensieveDT(columns)
        predict = teacher.predict

    else:
        raise NotImplementedError

    t1 = time.time()




    i = int(args.item)

    print('Iteration {}/{}'.format(i, max_iters))
    model_name = 'gosdtc'+str(i)
   
    dt_policy = GOSDT()
    dt_policy.load("file/"+str(model_name))

    t4 = time.time()

    t5 = time.time()

    reward = 0

    t2 = time.time()
    time_calc[i][0] = t2 - t1 + t4 - t5

    student_trace = get_rollouts(env=net_env, policy=student, args=args, n_batch_rollouts=n_batch_rollouts,
                                    dt_policy=dt_policy)
    student_states = [state for state, _, _ in student_trace]
    student_actions = [action for _, action, _ in student_trace]
    student_serials = [serial for _, _, serial in student_trace]

    t3 = time.time()
    time_calc[i][1] = t3 - t2

    teacher_actions = map(predict, student_states)

    states.extend(student_states)

   

    actions.extend(teacher_actions)

    serials.extend(student_serials)

    t1 = time.time()
    time_calc[i][2] = t1 - t3
    cur_states, cur_actions, cur_serials = resample(np.array(states), np.array(actions), np.array(serials), pts)

    cur_serials = pd.DataFrame(cur_serials, columns=first_columns)
    cur_actions =  pd.Series( cur_actions,name= 'test')
    serials_train, serials_val, actions_train, actions_val = train_test_split(cur_serials, cur_actions, test_size=train_frac, random_state=42)

    dfs = {
                   'X_train': serials_train,
            'X_test': serials_val,
            'y_train': actions_train,
            'y_test':actions_val,
            'state': states,
            'action': actions,
                'serial':  serials,



        }
    with h5py.File("save/datac"+str(i+1), "w") as f:
        for file_name, df in dfs.items():
            f[file_name] = df
    