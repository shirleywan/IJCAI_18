import pandas as pd
import numpy as np 
import pickle
import time
import gc

#user_id和day的上一次及下一次时间差
def AddTimeFeatures_next(df,cols):
    df=df[cols+['context_timestamp']]
    df['index']=[i for i in range(df.shape[0])]
    df=df.sort_values('context_timestamp')
    df['_'.join(cols)+'nextClick_timedelta'] = (df.groupby(cols).context_timestamp.shift(-1) - df.context_timestamp).dt.seconds.astype(np.float32)
    df=df.sort_values('index')
    return df[['_'.join(cols)+'nextClick_timedelta']]

def AddTimeFeatures_last(df,cols):
    df=df[cols+['context_timestamp']]
    df['index']=[i for i in range(df.shape[0])]
    df=df.sort_values('context_timestamp')
    df['_'.join(cols)+'lastClick_timedelta'] = (df.groupby(cols).context_timestamp.shift(+1) - df.context_timestamp).dt.seconds.astype(np.float32)
    df=df.sort_values('index')
    return df[['_'.join(cols)+'lastClick_timedelta']]

def Time(data_df):
    tf_df=pd.DataFrame()
    cols=[['user_id','day'],['user_id','day','hour']]
    for col in cols:
        tmp=AddTimeFeatures_next(data_df,col)
        tf_df=pd.concat([tf_df,tmp],axis=1)
        tmp=AddTimeFeatures_last(data_df,col)
        tf_df=pd.concat([tf_df,tmp],axis=1)
    return tf_df


start=time.time()

with open('../data/TestB_data.pickle','rb') as output:
    data=pickle.load(output)
df1=Time(data)
df1=df1.reset_index(drop=True)
df1.to_csv('../data/TestB_data_self_part3_UseDay.csv',index=False)
    
print ('Member_features_self_part3 Time:',(time.time()-start)/60.0)