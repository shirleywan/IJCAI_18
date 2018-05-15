from bayes_smoothing import *
import pandas as pd
import numpy as np 
import pickle
import time

#转化率特征
cols=[['user_id'],['item_id'],['shop_id'],['item_category_list'],['user_occupation_id'],['user_age_level'],['user_star_level'],['shop_star_level'],['user_id','item_category_list']]

def Pre(df,cols):
    df=df[['user_id','item_id','shop_id','item_category_list','user_occupation_id','user_age_level',
               'user_star_level','shop_star_level','context_timestamp','is_trade']]
    df['tmp_count']=1
    features=[]
    for col in cols:
        col_name='_'.join(col)+'_browse'
        features.append(col_name)
        df[col_name]=df.groupby(col)['tmp_count'].cumsum()
    df.loc[df['is_trade']==-1,'is_trade']=0
    df.loc[df['is_trade']==-2,'is_trade']=0
    df['raw_index']=[i for i in range(df.shape[0])]
    df=df.sort_values(['context_timestamp'])
    for col in cols:
        col_name='_'.join(col)+'_buy'
        features.append(col_name)
        df[col_name]=df.groupby(col)['is_trade'].cumsum()
        df[col_name]=df[col_name]-df['is_trade']
    df=df.sort_values(['raw_index'])
    return df[features]

def Ctr(df,cols):
    features=[]
    for col in cols:
        browse='_'.join(col)+'_browse'
        buy='_'.join(col)+'_buy'
        col_name='_'.join(col)+'_ctr'
        df[col_name]=bs_utilize(df[browse], df[buy])
        features.append(col_name)
    return df
start=time.time()

with open('../data/TestB_data.pickle','rb') as output:
    data=pickle.load(output)
df1=Pre(data,cols)
df2=Ctr(df1,cols)
with open('../data/TestB_data_member_part3.pickle','wb') as output:
    pickle.dump(df2,output)
    
print ('Member_features_self_part3 Time:',(time.time()-start)/60.0)