import pandas as pd
import numpy as np 
import lightgbm as lgb
import time
import gc
from sklearn.metrics import log_loss
from contextlib import contextmanager
import pickle

start=time.time()
with open('../data/TestB_data.pickle','rb') as output:
    data=pickle.load(output)



def GetCombina(df2):
    item_category_list_unique = list(np.unique(df2.item_category_list))
    df2.item_category_list.replace(item_category_list_unique, list(np.arange(len(item_category_list_unique))),
                                  inplace=True)

    df2['(item_category_list+shop_score_description)']=df2['item_category_list']+df2['shop_score_description']
    df2['(item_category_list-shop_score_description)']=df2['item_category_list']-df2['shop_score_description']
    df2['(item_category_list-item_city_id)']=df2['item_category_list']-df2['item_city_id']
    df2['(item_category_list-shop_id)']=df2['item_category_list']-df2['shop_id']
    df2['(item_category_list*shop_id)']=df2['item_category_list']*df2['shop_id']
    df2['(item_category_list+day)']=df2['item_category_list']+df2['day']
    df2['(item_category_list-shop_score_service)']=df2['item_category_list']-df2['shop_score_service']
    df2['(item_category_list*item_price_level)']=df2['item_category_list']*df2['item_price_level']
    df2['(item_price_level+shop_review_positive_rate)']=df2['item_price_level']+df2['shop_review_positive_rate']
    df2['(item_price_level+user_star_level)']=df2['item_price_level']+df2['user_star_level']
    df2['(item_price_level/user_star_level)']=df2['item_price_level']/df2['user_star_level']
    df2['(item_collected_level*day)']=df2['item_collected_level']*df2['day']
    df2['(item_pv_level+shop_id)']=df2['item_pv_level']+df2['shop_id']
    df2['(item_pv_level-item_brand_id)']=df2['item_pv_level']-df2['item_brand_id']
    df2['(item_pv_level*item_category_list)']=df2['item_pv_level']*df2['item_category_list']
    df2['(day-shop_id)']=df2['day']-df2['shop_id']
    df2['(shop_score_description*shop_score_description)']=df2['shop_score_description']*df2['shop_score_description']
    df2['(shop_score_description+(item_category_list*shop_id))']=df2['shop_score_description']+df2['(item_category_list*shop_id)']
    df2['(shop_score_description+(item_category_list-shop_score_service))']=df2['shop_score_description']+df2['(item_category_list-shop_score_service)']
    df2['(shop_score_description+context_page_id)']=df2['shop_score_description']+df2['context_page_id']
    #df2['(shop_score_description+user_id_item_iddiffTime_last)']=df2['shop_score_description']+df2['user_id_item_iddiffTime_last']
    df2['(shop_score_description-(item_category_list+day))']=df2['shop_score_description']-df2['(item_category_list+day)']
    df2['(shop_score_description-day)']=df2['shop_score_description']-df2['day']
    #df2['(shop_score_description/user_id_item_brand_iddiffTime_last)']=df2['shop_score_description']/df2['user_id_item_brand_iddiffTime_last']
    #df2['(shop_score_description/user_id_item_iddiffTime_last)']=df2['shop_score_description']/df2['user_id_item_iddiffTime_last']
    df2['(shop_score_service+(item_category_list*item_price_level))']=df2['shop_score_service']+df2['(item_category_list*item_price_level)']
    df2['(shop_score_service+(item_category_list-item_city_id))']=df2['shop_score_service']+df2['(item_category_list-item_city_id)']
    #df2['(user_id_item_category_listdiffTime_last/shop_review_num_level)']=df2['user_id_item_category_listdiffTime_last']/df2['shop_review_num_level']
    cols=[col for col in df2.columns if '+' in col or '-' in col or '*' in col or '/' in col]
    return df2[cols]
df=GetCombina(data)
df=df.reset_index(drop=True)
with open('../data/TestB_data_member_part2.pickle','wb') as output:
    pickle.dump(df,output)
    
print ('Member_features2 Time',time.time()-start)