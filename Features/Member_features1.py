import pandas as pd
import numpy as np 
import lightgbm as lgb
import time
import gc
from sklearn.metrics import log_loss
from contextlib import contextmanager
import pickle
import time

start=time.time()
with open('../data/TestB_data.pickle','rb') as output:
    data_df=pickle.load(output)
    

def similar(data12):
    data12['predict_propertys'] = data12['predict_category_property'].map(lambda x:str(x).replace(':',';'))
    data12['predict_propertys'] = data12['predict_propertys'].map(lambda x:str(x).replace(',',';'))
    def category_num(x):
        return len(list(set(str(x.ix['predict_propertys']).split(';')).intersection(set(str(x.ix['item_category_list']).split(';')))))
    def property_num(x):
        return len(list(set(str(x.ix['predict_propertys']).split(';')).intersection(set(str(x.ix['item_property_list']).split(';')))))
    data12['category_num']=data12.apply(category_num,axis=1) 
    data12['property_num']=data12.apply(property_num,axis=1) 
    del data12['predict_propertys']
    gc.collect()
    addlist2=[]
    addlist2.append('category_num')
    addlist2.append('property_num')
    return data12[addlist2]

def zuhe2(data):
    for col in ['user_gender_id','user_age_level','user_occupation_id','user_star_level']:
        data[col] = data[col].apply(lambda x: 0 if x == -1 else x)

    for col in ['item_sales_level', 'item_price_level', 'item_collected_level',
                'user_gender_id','user_age_level','user_occupation_id','user_star_level',
                'shop_review_num_level', 'shop_star_level','category_num','property_num']:
        data[col] = data[col].astype(str)
    print('与item组合')
    data['category_sale'] = data['item_sales_level'] + data['category_num']
    data['category_collect'] = data['item_collected_level'] + data['category_num']
    data['category_price'] = data['item_price_level'] + data['category_num']
    data['property_sale'] = data['item_sales_level'] + data['property_num']
    data['property_collect'] = data['item_collected_level'] + data['property_num']
    data['property_price'] = data['item_price_level'] + data['property_num']

    print('与user组合')
    data['category_age'] = data['category_num'] + data['user_age_level']
    data['category_occ'] = data['category_num'] + data['user_occupation_id']
    data['category_star'] = data['category_num'] + data['user_star_level']
    data['category_gender'] = data['category_num'] + data['user_gender_id']
    data['property_age'] = data['property_num'] + data['user_age_level']
    data['property_occ'] = data['property_num'] + data['user_occupation_id']
    data['property_star'] = data['property_num'] + data['user_star_level']
    data['property_gender'] = data['property_num'] + data['user_gender_id']

    print('与shop组合')
    data['category_shop_star'] = data['category_num'] + data['shop_star_level']
    data['category_review'] = data['category_num'] + data['shop_review_num_level']
    data['property_shop_star'] = data['property_num'] + data['shop_star_level']
    data['property_review'] = data['property_num'] + data['shop_review_num_level']
    
    print('两两组合')
    data['property_category'] = data['property_num'] + data['category_num']

    for col in ['item_sales_level', 'item_price_level', 'item_collected_level',
                'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level',
                'shop_review_num_level','shop_star_level',
#                21维
               'category_sale', 'category_collect', 'category_price',
                'property_sale', 'property_collect', 'property_price', 'category_age',
                'category_occ','category_star','category_gender', 'property_age', 'property_occ',
                'property_star', 'property_gender', 'category_shop_star', 'category_review',
                'property_shop_star','property_review','property_category',
                'category_num', 'property_num'
               ]:
        data[col] = data[col].astype(int)
    gc.collect()
    cols=['category_sale', 'category_collect', 'category_price',
                'property_sale', 'property_collect', 'property_price', 'category_age',
                'category_occ','category_star','category_gender', 'property_age', 'property_occ',
                'property_star', 'property_gender', 'category_shop_star', 'category_review',
                'property_shop_star','property_review','property_category',
                'category_num', 'property_num']
    return data[cols]

df1=similar(data_df)
data_df=data_df.reset_index(drop=True)
df1=df1.reset_index(drop=True)
cols=[col for col in data_df.columns if col not in df1.columns]
data_df=pd.concat([data_df[cols],df1],axis=1)

df2=zuhe2(data_df)
df=pd.concat([df1,df2],axis=1)
with open('../data/TestB_data_member_part1.pickle','wb') as output:
    pickle.dump(df,output)
    
print ('Member_features1 Time',time.time()-start)