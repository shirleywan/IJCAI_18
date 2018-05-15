import pandas as pd
import numpy as np 
import pickle
import time
import gc
import datetime
#构建上一次购买的时间差
def diff_to_last_buy(df_input, column_pair):
	def _get_time(s):
		date_received, dates = s.split('@')
		if dates=='-1':
			return -1
		dates = dates.split(';')
		gaps = []
		startTime= datetime.datetime.strptime(date_received,"%Y-%m-%d %H:%M:%S")
		for d in dates:
		    endTime= datetime.datetime.strptime(d,"%Y-%m-%d %H:%M:%S")  
		    if startTime>endTime:
		        this_gap =(startTime - endTime).seconds/3600
		        gaps.append(this_gap)
		if len(gaps)==0:
			return -1
		#可以返回所有的间隔，也可以返回最近的，也就是min
		return min(gaps)  
	pair_name = '_'.join(column_pair)+'_df2lb'
	column_pair_1 = column_pair + ['context_timestamp', 'is_trade']
	t6 = df_input[column_pair_1]
	t6.context_timestamp = t6.context_timestamp.astype('str')
	t6 = t6[t6.is_trade==1].groupby(column_pair)['context_timestamp'].agg(lambda x: ';'.join(x)).reset_index()
	t6.rename(columns={'context_timestamp': 'dates'}, inplace=True)
	column_pair_2 = column_pair + ['context_timestamp']
	t7 = df_input[column_pair_2]
	t7 = pd.merge(t7, t6, on=column_pair, how='left')
	t7['dates']= t7['dates'].fillna('-1')
	t7['context_timestamp'] = t7.context_timestamp.astype('str') + '@' + t7.dates
	t7 = t7[column_pair_2]
	del t6
	gc.collect()
	t7[pair_name] = t7['context_timestamp'].apply(lambda x:_get_time(x))
	t7=t7[[pair_name]]
	return t7
	
	
def all_diff_to_last_buy_fetch(data_input, global_browse_pair):
    df=pd.DataFrame()
    for c in global_browse_pair:
        column_input = c +['context_timestamp'] + ['is_trade']
        df_tmp= diff_to_last_buy(data_input[column_input], c)
        df=pd.concat([df,df_tmp],axis=1)
    return df
start=time.time()
		
		
with open('../data/TestB_data.pickle','rb') as output:
    data=pickle.load(output)
# 执行
df=all_diff_to_last_buy_fetch(data, [['user_id'],
                                           ['user_id', 'item_id'],
										   ['user_id', 'shop_id']])
with open('../data/TestB_data_member_part4.pickle','wb') as output:
    pickle.dump(df,output)
    
print ('Member_features_self_part4 Time:',(time.time()-start)/60.0)
