import pandas as pd
import numpy as np 
import pickle
import time
import gc

def AddShift(df):
    df['raw_index']=range(df.shape[0])
    df=df.sort_values(by=['user_id','context_timestamp'])
    df['next_item']=df.item_id.shift(-1)
    df['next_shop']=df.shop_id.shift(-1)
    df['next_predict']=df.predict_category_property.shift(-1)
    df['next_item_category']=df.item_category_list.shift(-1)
    df['next_click_time']=df.context_timestamp.shift(-1)
    
    df['has_next_item']=np.where(df.item_id == df.next_item, 1, 0)
    df['has_next_shop']=np.where(df.shop_id == df.next_shop, 1, 0)
    df['has_next_predict']=np.where(df.predict_category_property == df.next_predict, 1, 0)
    df['has_next_item_category']=np.where(df.item_category_list == df.next_item_category, 1, 0)
    
    df['next_same']=np.where((df.has_next_item == 1) & (df.has_next_shop == 1) &(df.has_next_predict == 1) & (df.has_next_item_category == 1) , (df.next_click_time - df.context_timestamp)/np.timedelta64(1, 's'), np.NaN)
    
    df['previous_item']=df.item_id.shift(1)
    df['previous_shop']=df.shop_id.shift(1)
    df['previous_predict']=df.predict_category_property.shift(1)
    df['previous_item_category']=df.item_category_list.shift(1)
    df['previous_click_time']=df.context_timestamp.shift(1)
    
    df['has_previous_item']=np.where(df.item_id == df.previous_item, 1, 0)
    df['has_previous_shop']=np.where(df.shop_id == df.previous_shop, 1, 0)
    df['has_previous_predict']=np.where(df.predict_category_property == df.previous_predict, 1, 0)
    df['has_previous_item_category']=np.where(df.item_category_list == df.previous_item_category, 1, 0)
    
    df['previous_same']=np.where((df.has_previous_item == 1) & (df.has_previous_shop == 1) &(df.has_previous_predict == 1) & (df.has_previous_item_category == 1) , (df.previous_click_time - df.context_timestamp)/np.timedelta64(1, 's'), np.NaN)
    df=df.sort_values(by=['raw_index'])
    
    cols=['previous_same','next_same']
    return df,cols

#得到下一次再次相同点击时间差的统计特征
def get_next_click_stat(df):
    df['row_id']=range(df.shape[0])
    
    grouping_col = ['user_id']
    target_col = ['next_same']
    
    used_cols = grouping_col.copy()
    used_cols.extend(['row_id'])
    used_cols.extend(target_col)
    all_df = df[used_cols]
    
    group_used_cols = grouping_col.copy()
    group_used_cols.extend(target_col)
    grouped = all_df[group_used_cols].groupby(grouping_col)
    
    new_names = []
    #mean
    the_mean = pd.DataFrame(grouped[target_col].mean()).reset_index()
    names = grouping_col.copy()
    new_name = 'next_click_mean'
    new_names.append(new_name)
    names.append(new_name)
    the_mean.columns = names
    #median
    the_median = pd.DataFrame(grouped[target_col].median()).reset_index()
    names = grouping_col.copy()
    new_name = 'next_click_median'
    new_names.append(new_name)
    names.append(new_name)
    the_median.columns = names
    the_stats = pd.merge(the_mean, the_median)
    #max
    the_max = pd.DataFrame(grouped[target_col].max()).reset_index()
    names = grouping_col.copy()
    new_name = 'next_click_max'
    new_names.append(new_name)
    names.append(new_name)
    the_max.columns = names
    the_stats = pd.merge(the_stats, the_max)
    
    all_df = pd.merge(all_df, the_stats)
    all_df.sort_values('row_id', inplace=True)
    
    for new_name in new_names:
        df[new_name] = np.array(all_df[new_name])
    del all_df
    print('get_next_click_stat is done')
    return df[new_names]

def Shift(data_df,cols):
    data_df,cols=AddShift(data_df)
    use_1=data_df[cols]
    use_2=get_next_click_stat(data_df)
    use_=pd.concat([use_1,use_2],axis=1)
    return use_

def RateInfo(data_df,cols1,cols2):
    qe=data_df.groupby(cols1).size()
    qe=qe.reset_index()
    qe.columns=cols1+['_'.join(cols1)+'_Size']
    data_df=pd.merge(data_df,qe,how='left',on=cols1)
    
    qe2=data_df.groupby(cols2).size()
    qe2=qe.reset_index()
    qe2.columns=cols2+['_'.join(cols2)+'_Size']
    data_df=pd.merge(data_df,qe2,how='left',on=cols2)
    data_df['_'.join(cols1)+'_Size'+'/'+'_'.join(cols2)+'_Size']=data_df['_'.join(cols1)+'_Size']/data_df['_'.join(cols2)+'_Size']
    return data_df['_'.join(cols1)+'_Size'+'/'+'_'.join(cols2)+'_Size']

def CountInfo_rate2(data_df,cols):
    qe=data_df.groupby(cols).size()
    qe=qe.reset_index()
    qe.columns=cols+['_'.join(cols)+'_Size']
    data_df=pd.merge(data_df,qe,how='left',on=cols)
    return data_df[['_'.join(cols)+'_Size']].values,'_'.join(cols)+'_Size'

def UniqueInfo_rate2(data_df,cols):
    qe=data_df[cols].groupby(cols[:-1])[cols[-1]].nunique()
    qe=qe.reset_index()
    qe.columns=cols[:-1]+['_'.join(cols)+'_Unique']
    data_df=pd.merge(data_df,qe,how='left',on=cols[:-1])
    return data_df[['_'.join(cols)+'_Unique']].values,'_'.join(cols)+'_Unique'

def Rate_2(df):
    d1,a=CountInfo_rate2(df,['user_id'])
    d2,b=UniqueInfo_rate2(df,['user_id','item_id'])
    d3,c=UniqueInfo_rate2(df,['user_id','shop_id'])
    d4,d=UniqueInfo_rate2(df,['user_id','item_category_list'])
    d5,e=UniqueInfo_rate2(df,['user_id','predict_category_property'])
    df[b+'/'+a]=d2/d1
    df[c+'/'+a]=d3/d1
    df[d+'/'+a]=d4/d1
    df[e+'/'+a]=d5/d1
    return df[[b+'/'+a,c+'/'+a,d+'/'+a,e+'/'+a]]

def Rate(data_df):
    f1=RateInfo(data_df,['shop_id'],['shop_id','day'])
    f2=RateInfo(data_df,['item_id'],['item_id','day'])
    f3=RateInfo(data_df,['item_id','day','hour'],['item_id','day','hour','minute'])
    f4=RateInfo(data_df,['user_id'],['user_id','day'])
    f5=RateInfo(data_df,['user_id','day'],['user_id','day','hour'])
    f6=RateInfo(data_df,['user_id','day','hour'],['user_id','day','hour','minute'])
    df=pd.concat([f1,f2,f3,f4,f5,f6],axis=1)
    return df

def AddTimeHow(df,cols,how,type_):
    df=df[cols+['context_timestamp']]
    df['index']=[i for i in range(df.shape[0])]
    df=df.sort_values('context_timestamp')
    if type_=='next':
        df['_'.join(cols)+'nextClick_timedelta_how'+str(how)] = (df.groupby(cols).context_timestamp.shift(-how) - df.context_timestamp).dt.seconds.astype(np.float32)
        df=df.sort_values('index')
        return df[['_'.join(cols)+'nextClick_timedelta_how'+str(how)]]
    if type_=='last':
        df['_'.join(cols)+'lastClick_timedelta_how'+str(how)] = (df.groupby(cols).context_timestamp.shift(+how) - df.context_timestamp).dt.seconds.astype(np.float32)
        df=df.sort_values('index')
        return df[['_'.join(cols)+'lastClick_timedelta_how'+str(how)]]

# 生成user_id相关的时间特征
def TimeMul(data_df):
    F=['user_id']
    S=['item_id','shop_id','predict_category_property','item_category_list','item_price_level',
   'item_sales_level','item_collected_level','item_pv_level','user_gender_id','user_age_level','user_occupation_id',
    'user_star_level','context_page_id','shop_review_num_level','shop_star_level']
    data_df=data_df[F+S+['context_timestamp']]
    tf_df=pd.DataFrame()
    for i in F:
        for j in S:
            for k in [2,3,4,5]:
                cols=[]
                cols.append(i)
                cols.append(j)
                tmp=AddTimeHow(data_df,cols,k,'next')
                tf_df=pd.concat([tf_df,tmp],axis=1)
                tmp=AddTimeHow(data_df,cols,k,'last')
                tf_df=pd.concat([tf_df,tmp],axis=1)
                del tmp
                gc.collect()
    return tf_df

start=time.time()

with open('../data/TestB_data.pickle','rb') as output:
    data=pickle.load(output)
df1=TimeMul(data)
df1=df1.reset_index(drop=True)
df1.to_csv('../data/TestB_data_self_part2_2_TimeMul.csv',index=False)
    
print ('Member_features_self_part2_2 Time:',(time.time()-start)/60.0)