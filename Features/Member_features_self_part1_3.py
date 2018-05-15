import pandas as pd
import numpy as np 
import pickle
import time
import gc
def CountInfo(data_df,cols):
    qe=data_df.groupby(cols).size()
    qe=qe.reset_index()
    qe.columns=cols+['_'.join(cols)+'_Size']
    data_df=pd.merge(data_df,qe,how='left',on=cols)
    return data_df[['_'.join(cols)+'_Size']]
def Count(data_df,TargetCols):
    df=pd.DataFrame()
    Features=['user_id','item_id','shop_id','item_category_list','predict_category_property']
    data_df=data_df[Features+['day','hour']]
    for i in Features:
        if '_'.join([i])+'_Size' in TargetCols:
            tmp=CountInfo(data_df,[i])
            df=pd.concat([df,tmp],axis=1)
        if '_'.join([i,'day'])+'_Size' in TargetCols:
            tmp=CountInfo(data_df,[i,'day'])
            df=pd.concat([df,tmp],axis=1)
        if '_'.join([i,'day','hour'])+'_Size' in TargetCols:
            tmp=CountInfo(data_df,[i,'day','hour'])
            df=pd.concat([df,tmp],axis=1) 
    for i in range(len(Features)):
        for j in range(len(Features)):
            if j>i:
                tmp=CountInfo(data_df,[Features[i],Features[j]])
                df=pd.concat([df,tmp],axis=1)
                tmp=CountInfo(data_df,[Features[i],Features[j],'day'])
                df=pd.concat([df,tmp],axis=1)
                tmp=CountInfo(data_df,[Features[i],Features[j],'day','hour'])
                df=pd.concat([df,tmp],axis=1)

    del data_df
    gc.collect()
    return df[TargetCols]

def UniqueInfo(data_df,cols):
    qe=data_df[cols].groupby(cols[:-1])[cols[-1]].nunique()
    qe=qe.reset_index()
    qe.columns=cols[:-1]+['_'.join(cols)+'_Unique']
    data_df=pd.merge(data_df,qe,how='left',on=cols[:-1])
    return data_df[['_'.join(cols)+'_Unique']]

def Unique(data_df,TargetCols):
    Features=['user_id','item_id','shop_id','item_category_list','predict_category_property']
    df=pd.DataFrame()
    data_df=data_df[Features+['day','hour']]
    for i in range(len(Features)):
        for j in range(len(Features)):
            if j>i:
                tmp=UniqueInfo(data_df,[Features[i],Features[j]])
                df=pd.concat([df,tmp],axis=1)
                tmp=UniqueInfo(data_df,[Features[i],'day',Features[j]])
                df=pd.concat([df,tmp],axis=1)
                tmp=UniqueInfo(data_df,[Features[i],'day','hour',Features[j]])
                df=pd.concat([df,tmp],axis=1)

    del data_df
    gc.collect()
    return df[TargetCols]

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

def Time(data_df,TargetCols):
    F=['user_id']
    S=['item_id','shop_id','predict_category_property','item_category_list','item_price_level',
   'item_sales_level','item_collected_level','item_pv_level','user_gender_id','user_age_level','user_occupation_id',
    'user_star_level','context_page_id','shop_review_num_level','shop_star_level']
    tf_df=pd.DataFrame()
    for i in F:
        for j in S:
            cols=[]
            cols.append(i)
            cols.append(j)
            tmp=AddTimeFeatures_next(data_df,cols)
            tf_df=pd.concat([tf_df,tmp],axis=1)
            tmp=AddTimeFeatures_last(data_df,cols)
            tf_df=pd.concat([tf_df,tmp],axis=1)
    return tf_df[TargetCols]

def First(Series):
    min_=min(Series)
    return min_
def End(Series):
    max_=max(Series)
    return max_

def GetCol(data_df,cols,type_):
    data_df=data_df[cols+['context_timestamp']]
    if type_=='first':
        first_df=data_df.groupby(cols)['context_timestamp'].agg([('first_time',First)]).reset_index()
        data_df=pd.merge(data_df,first_df,how='left',on=cols)
        data_df['_'.join(cols)+'_first']=(data_df['context_timestamp']-data_df['first_time']).apply(lambda x:x.total_seconds()/60.0)
        return data_df[['_'.join(cols)+'_first']]
    elif type_=='end':
        end_df=data_df.groupby(cols)['context_timestamp'].agg([('end_time',End)]).reset_index()
        data_df=pd.merge(data_df,end_df,how='left',on=cols)
        data_df['_'.join(cols)+'_end']=(data_df['context_timestamp']-data_df['end_time']).apply(lambda x:x.total_seconds()/60.0)
        return data_df[['_'.join(cols)+'_end']]

def FirstEnd(data,TargetCols):
    cols=[['user_id'],['user_id','day'],['user_id','item_id'],['user_id','day','item_id'],['user_id','item_category_list'],['user_id','day','item_category_list'],
     ['user_id','predict_category_property'],['user_id','day','predict_category_property'],['user_id','day','hour'],
     ['user_id','day','hour','item_id'],['user_id','day','hour','item_category_list'],['user_id','day','hour','predict_category_property']]
    re_df=pd.DataFrame()
    for col in cols:
        df=GetCol(data,col,'first')
        re_df=pd.concat([re_df,df],axis=1)
        df=GetCol(data,col,'end')
        re_df=pd.concat([re_df,df],axis=1)
    return re_df[TargetCols]

start=time.time()

FirstEnd_list=['user_id_first','user_id_end','user_id_item_id_first','user_id_item_id_end','user_id_day_item_id_first',
              'user_id_day_item_id_end','user_id_item_category_list_first','user_id_item_category_list_end',
              'user_id_day_item_category_list_end','user_id_predict_category_property_first',
             'user_id_day_predict_category_property_end','user_id_day_hour_end','user_id_day_hour_item_id_first',
             'user_id_day_hour_item_category_list_end','user_id_day_hour_predict_category_property_end']
Time_list=[
'user_id_item_idnextClick_timedelta',
'user_id_item_idlastClick_timedelta',
'user_id_shop_idnextClick_timedelta',
'user_id_predict_category_propertynextClick_timedelta',
'user_id_predict_category_propertylastClick_timedelta',
'user_id_item_category_listnextClick_timedelta',
'user_id_item_price_levelnextClick_timedelta',
'user_id_item_sales_levelnextClick_timedelta',
'user_id_item_collected_levelnextClick_timedelta',
'user_id_item_pv_levelnextClick_timedelta',
'user_id_user_gender_idnextClick_timedelta',
'user_id_user_gender_idlastClick_timedelta',
'user_id_user_age_levelnextClick_timedelta',
'user_id_user_age_levellastClick_timedelta',
'user_id_user_occupation_idnextClick_timedelta',
'user_id_user_occupation_idlastClick_timedelta',
'user_id_user_star_levelnextClick_timedelta',
'user_id_context_page_idlastClick_timedelta',
'user_id_shop_review_num_levelnextClick_timedelta',
'user_id_shop_review_num_levellastClick_timedelta',
'user_id_shop_star_levelnextClick_timedelta']
Count_list=[
'user_id_Size',
'user_id_day_Size',
'user_id_day_hour_Size',
'item_id_Size',
'shop_id_Size',
'shop_id_day_Size',
'shop_id_day_hour_Size',
'predict_category_property_Size',
'predict_category_property_day_Size',
'predict_category_property_day_hour_Size',
'user_id_item_id_Size',
'user_id_item_id_day_Size',
'user_id_item_id_day_hour_Size',
'user_id_shop_id_Size',
'user_id_item_category_list_Size',
'user_id_item_category_list_day_Size',
'user_id_item_category_list_day_hour_Size',
'user_id_predict_category_property_day_hour_Size',
'item_id_item_category_list_Size',
'item_id_predict_category_property_Size',
'item_id_predict_category_property_day_hour_Size',
'shop_id_item_category_list_Size',
'shop_id_item_category_list_day_hour_Size',
'item_category_list_predict_category_property_Size',
'item_category_list_predict_category_property_day_hour_Size'
]
Unique_list=[
'user_id_day_item_id_Unique',
'user_id_day_hour_item_id_Unique',
'user_id_shop_id_Unique',
'user_id_day_shop_id_Unique',
'user_id_day_hour_shop_id_Unique',
'user_id_day_hour_item_category_list_Unique',
'user_id_predict_category_property_Unique',
'item_id_predict_category_property_Unique',
'item_id_day_predict_category_property_Unique',
'item_id_day_hour_predict_category_property_Unique',
'shop_id_item_category_list_Unique',
'shop_id_day_item_category_list_Unique',
'shop_id_day_hour_item_category_list_Unique',
'shop_id_predict_category_property_Unique',
'shop_id_day_predict_category_property_Unique',
'shop_id_day_hour_predict_category_property_Unique',
'item_category_list_day_predict_category_property_Unique',
'item_category_list_day_hour_predict_category_property_Unique'
]

with open('../data/TestB_data.pickle','rb') as output:
    data=pickle.load(output)
df1=Count(data,Count_list)
df1=df1.reset_index(drop=True)
with open('../data/TestB_data_self_part1_3_Count.pickle','wb') as output:
    pickle.dump(df1,output)
    
print ('Member_features_self_part1_3 Time:',(time.time()-start)/60.0)
