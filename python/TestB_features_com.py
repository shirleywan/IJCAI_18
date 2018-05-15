import pandas as pd
import numpy as np 
import pickle
import time
import gc

#对data进行预处理
def AddBaseFeatures(data):
    #data.loc[data['day']==7,'special_day']=3
    #data.loc[data['day'].isin([5,6]),'special_day']=2
    #data.loc[data['day']==4,'special_day']=1
    
    df_shop_gender_ratio = data[data['is_trade']!=-1].groupby(['shop_id'])['user_gender_id']\
                            .agg([lambda x: np.mean(x == 0)])\
                            .reset_index()\
                            .rename(columns={'<lambda>': 'shop_user_gender_ratio'})
    df_shop_avg_age_level = data[data['is_trade']!=-1].groupby(['shop_id'])['user_age_level']\
                            .mean()\
                            .reset_index()\
                            .rename(columns={'user_age_level': 'user_avg_age_level'})
                
    data=pd.merge(data,df_shop_gender_ratio,how='left',on=['shop_id'])
    data=pd.merge(data,df_shop_avg_age_level,how='left',on=['shop_id'])
    
    la=data['item_id'].value_counts()
    small_item_id=la[la<2000].index.tolist()
    data.loc[data['item_id'].isin(small_item_id),'item_id']=-999
    data['item_id']=data['item_id'].astype('category').cat.codes
    
    ls=data['shop_id'].value_counts()
    small_shop_id=ls[ls<5000].index.tolist()
    data.loc[data['shop_id'].isin(small_shop_id),'shop_id']=-999
    data['shop_id']=data['shop_id'].astype('category').cat.codes
    
    lu=data['user_id'].value_counts()
    small_user_id=lu[lu<20].index.tolist()
    data.loc[data['user_id'].isin(small_user_id),'user_id']=-999
    data['user_id']=data['user_id'].astype('category').cat.codes
    
    data['time']=(data['day']-4)*24+data['hour']+data['minute']/60.0

    return data

with open('../data/TestB_data.pickle','rb') as output:
    data=pickle.load(output)

data=AddBaseFeatures(data)

with open('../data/TestB_data_member_part2.pickle','rb') as output:
    other_1=pickle.load(output)
with open('../data/TestB_data_member_part1.pickle','rb') as output:
    other_2=pickle.load(output)
other=pd.concat([other_1,other_2],axis=1)
del other_1,other_2
with open('../data/TestB_data_member_part3.pickle','rb') as output:
    other_3=pickle.load(output)
with open('../data/TestB_data_member_part4.pickle','rb') as output:
    other_4=pickle.load(output)
other_3=other_3.reset_index(drop=True)
other_4=other_4.reset_index(drop=True)

other=pd.concat([other,other_3,other_4],axis=1)
del other_3,other_4
gc.collect()
data=data.reset_index(drop=True)
data=pd.concat([data,other],axis=1)

with open('../data/TestBdata_1.pickle','wb') as output:
    pickle.dump(data,output)

with open('../data/TestB_data_self_part1_1_Unique.pickle','rb') as output:
    self_1=pickle.load(output)
with open('../data/TestB_data_self_part1_2_Time.pickle','rb') as output:
    self_2=pickle.load(output)
with open('../data/TestB_data_self_part1_3_Count.pickle','rb') as output:
    self_3=pickle.load(output)
with open('../data/TestB_data_self_part1_4_FirstEnd.pickle','rb') as output:
    self_4=pickle.load(output)
with open('../data/TestB_data_self_part2_1_Shift_Rate.pickle','rb') as output:
    self_5=pickle.load(output)
self=pd.concat([self_1,self_2,self_3,self_4,self_5],axis=1)
del self_1,self_2,self_3,self_4,self_5
gc.collect()

with open('../data/TestBdata_2.pickle','wb') as output:
    pickle.dump(self,output)