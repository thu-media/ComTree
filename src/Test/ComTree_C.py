def predict(state):
    feature_map = {
        'last_quality': 0, 'curr_buffer': 1, 'tput_0': 2, 'tput_1': 3,
        'tput_2': 4, 'tput_3': 5, 'tput_4': 6, 'tput_5': 7, 'tput_6': 8,
        'tput_7': 9, 'delay_0': 10, 'delay_1': 11, 'delay_2': 12,
        'delay_3': 13, 'delay_4': 14, 'delay_5': 15, 'delay_6': 16,
        'delay_7': 17, 'size_0': 18, 'size_1': 19, 'size_2': 20,
        'size_3': 21, 'size_4': 22, 'size_5': 23, 'chunk_til_video_end': 24
    }

    lq = state[feature_map['last_quality']]
    b = state[feature_map['curr_buffer']]
    t_7 = state[feature_map['tput_7']]

    if lq <= 0.23:
        if lq <= 0.122:
            if b <= 1.48:
                return 0
            else:
                return 1
        else:
            if b <= 2.28:
                if b <= 1.06:
                    if t_7 <= 0.11:
                        return 0
                    else:
                        return 1
                else:
                    if t_7 <= 0.26:
                        return 1
                    else:
                        return 3
            else:
                if b <= 2.64:
                    if t_7 <= 0.15:
                        return 1
                    else:
                        return 3
                else:
                    return 3
    else:
        if b <= 3.50:
            if lq <= 0.83:
                if t_7 <= 0.18:
                    if b <= 1.10:
                        return 1
                    else:
                        return 3
                else:
                    return 3
            else:
                if b <= 1.01:
                    return 4
                else:
                    return 5
        else:
            return 5