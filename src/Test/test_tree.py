import numpy as np
import fixed_env as env
import load_trace
import ComTree as decision_tree
# import ComTree_C as decision_tree
import os
import sys

S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000000
RESEVOIR = 5  # BB
CUSHION = 10  # BB
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
TEST_TRACES = ['oboe','norway','puffer-2110','puffer-2202']
SUMMARY_DIR = './test/'

import sys
feature_num = 25 


def main():
  
    np.random.seed(RANDOM_SEED)
    assert len(VIDEO_BIT_RATE) == A_DIM

    for idx, TEST_TRACE in enumerate(TEST_TRACES):

        all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(SUMMARY_DIR+TEST_TRACE+'/')

        net_env = env.Environment(all_cooked_time=all_cooked_time,
                                all_cooked_bw=all_cooked_bw)
        if not os.path.exists('./results/'+ TEST_TRACE ):
            os.mkdir('./results/'+ TEST_TRACE)
        LOG_FILE = './results/'+ TEST_TRACE +'/log_rsetp'
        log_path = LOG_FILE + '_' +all_file_names[net_env.trace_idx] #+ '_' + args.qoe_metric
        log_file = open(log_path, 'w')

        time_stamp = 0

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        r_batch = []
        s_batch = [np.zeros((S_INFO, S_LEN))]
        video_count = 0

        while True:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = \
                net_env.get_video_chunk(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # reward is video quality - rebuffer penalty
            reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                    - REBUF_PENALTY * rebuf \
                    - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
            
                                            VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
            
            if len(s_batch) != 1:
                r_batch.append(reward)
            last_bit_rate = bit_rate
    


            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)
            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
            serialized_state = []
            if feature_num  == 25:
                serialized_state.append(state[0, -1])
                serialized_state.append(state[1, -1])
                for i in range(S_LEN):
                    serialized_state.append(state[2, i])
                for i in range(S_LEN):
                    serialized_state.append(state[3, i])
                for i in range(A_DIM):
                    serialized_state.append(state[4, i])
                serialized_state.append(state[5, -1])
        

            bit_rate = decision_tree.predict(serialized_state)

            s_batch.append(state)
            log_file.write(str(time_stamp / M_IN_K) + '\t' +
                       str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                       str(buffer_size) + '\t' +
                       str(rebuf) + '\t' +
                       str(video_chunk_size) + '\t' +
                       str(delay) + '\t' +
                       str(bit_rate)+ '\t' +

                       str(reward) + '\n')
            log_file.flush()
            if end_of_video:
                log_file.write('\n')
                log_file.close()

                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here

                del s_batch[:]


                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1

                s_batch.append(np.zeros((S_INFO, S_LEN)))

    
                video_count += 1

                if video_count >= len(all_file_names):
                    break
      
                log_path = LOG_FILE + '_' +all_file_names[net_env.trace_idx]
                log_file = open(log_path, 'w')
        print( TEST_TRACE, np.mean(r_batch))
        

if __name__ == '__main__':
    main()