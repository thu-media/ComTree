import pandas as pd
import numpy as np
import time
import pathlib
from treefarms import TREEFARMS
import h5py
import numpy as np
import sys
seed = int(sys.argv[1])
np.random.seed(seed)
max_size = int(sys.argv[2])

model_file = path_model
columns_f =  path_columns
df1 =h5py.File(columns_f , 'r') 
columns = [item.decode('utf-8') for item in list(df1['c'])]

df =h5py.File(model_file, 'r') 
X_train_guessed = pd.DataFrame(df['X_train_guessed'], columns=columns)
X_test_guessed = pd.DataFrame(df['X_test_guessed'], columns=columns)
y_train =  pd.Series(df['y_train'],name= 'test')
y_test =  pd.Series(df['y_test'],name= 'test')

config = {
    "regularization": 0.0005,  # regularization penalizes the tree with more leaves. We recommend to set it to relative high value to find a sparse tree.
   "rashomon_bound_multiplier": 0.05,  # rashomon bound multiplier indicates how large of a Rashomon set would you like to get
    "max_hsize": max_size,
    "verbose": True,
    "depth_budget": 6,
    "worker_limit": 220,
    "model":"file/model3",
    "profile":"file/profile3",
    

    "feature_exchange":True,
    "continuous_feature_exchange":True,

}

model = TREEFARMS(config)

model.fit(X_train_guessed , y_train)
print('find num',model.get_tree_count())

set_num = model.get_tree_count()
last_cnt = 0
for idx in range(64):
    cnt = model.get_tree_at_idx_raw_instance(idx)
    if cnt<0:
        break
    print('cnt',last_cnt,cnt)
   
    for idx2 in range(last_cnt,cnt):
        print(idx2)
        print(model.get_tree_at_idx_raw(idx2))
    print('find acc',model.get_tree_metric_at_idx(cnt))
    last_cnt = cnt


