import os
for item in range(0,100):
    cmd ='python learn_gosdt_all.py -a pensieve -t fcc  -q lin -item '+str(item)
    print(cmd)
    os.system(cmd)
    cmd = 'python test_xgboost_all.py '+str(item+1)
    print(cmd)
    os.system(cmd)
