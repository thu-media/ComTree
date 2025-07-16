def predict(state):
    feature_map = {'last_quality': 0, 'curr_buffer': 1, 'tput_0': 2, 'tput_1': 3, 'tput_2': 4, 'tput_3': 5, 'tput_4': 6, 'tput_5': 7, 'tput_6': 8, 'tput_7': 9, 'delay_0': 10, 'delay_1': 11, 'delay_2': 12, 'delay_3': 13, 'delay_4': 14, 'delay_5': 15, 'delay_6': 16, 'delay_7': 17, 'size_0': 18, 'size_1': 19, 'size_2': 20, 'size_3': 21, 'size_4': 22, 'size_5': 23, 'chunk_til_video_end': 24}
    
    lq = state[feature_map['last_quality']]
    b = state[feature_map['curr_buffer']]
    t_7 = state[feature_map['tput_7']]
    t_5 = state[feature_map['tput_5']]
    
    if t_7 <= 0.15317139:
        if b <= 2.63507009:
            if t_7 <= 0.0934210494:
                if lq <= 0.122093022:
                    if b <= 1.47480035:
                        return 0
                    else:
                        return 1
                else:
                    if b <= 1.0500865:
                        return 0
                    else:
                        return 1
            else:
                if b <= 1.17780685:
                    if lq <= 0.122093022:
                        return 0
                    else:
                        return 1
                else:
                    if lq <= 0.22674419:
                        return 1
                    else:
                        return 3
        else:
            return 3
    else:
        if lq <= 0.354651153:
            if t_7 <= 0.255806863:
                if b <= 2.28890634:
                    if t_5 <= 0.0688233823:
                        return 0
                    else:
                        return 1
                else:
                    return 3
            else:
                if t_7 <= 0.324703485:
                    if b <= 1.47480035:
                        return 1
                    else:
                        return 3
                else:
                    if b <= 0.97983253:
                        return 1
                    else:
                        return 3
        else:
            if b <= 0.97983253:
                if t_7 <= 0.195492133:
                    return 1
                else:
                    return 3
            else:
                if lq <= 0.831395388:
                    if b <= 3.49860668:
                        return 3
                    else:
                        return 5
                else:
                    return 5