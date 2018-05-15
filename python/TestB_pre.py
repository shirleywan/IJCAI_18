import pandas as pd
import numpy as np 
import pickle
import time


train=pd.read_csv('../data/round2_train.txt',sep=' ')#10432036
test_a=pd.read_csv('../data/round2_ijcai_18_test_a_20180425.txt',sep=' ')#519888
test_b=pd.read_csv('../data/round2_ijcai_18_test_b_20180510.txt',sep=' ')#1209768
test_a['is_trade']=-1
test_b['is_trade']=-2
data=pd.concat([train,test_a,test_b],axis=0)
data['context_timestamp']=pd.to_datetime(data['context_timestamp'].apply(lambda x:time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(x))))
data['day']=data['context_timestamp'].dt.day
data['hour']=data['context_timestamp'].dt.hour
data['minute']=data['context_timestamp'].dt.minute

with open('../data/TestB_data.pickle','wb') as output:
    pickle.dump(data,output)