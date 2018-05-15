import pandas as pd
import numpy as np 
import lightgbm as lgb
import time
import gc
from sklearn.metrics import log_loss
from contextlib import contextmanager
import pickle

#loading data
with open('../data/TestBdata_1.pickle','rb') as output:
    data=pickle.load(output)
gc.collect()
with open('../data/TestBdata_2.pickle','rb') as output:
    self_1=pickle.load(output)
gc.collect()
with open('../data/TestB_data_self_part1_1_Unique.pickle','rb') as output:
    self_2=pickle.load(output)
gc.collect()
self_6=pd.read_csv('../data/TestB_data_self_part2_2_TimeMul.csv')
self_7=pd.read_csv('../data/TestB_data_self_part3_UseDay.csv')
gc.collect()
data=pd.concat([data,self_1,self_2,self_6,self_7],axis=1)
del self_1,self_2,self_6,self_7
gc.collect()
print (data.shape)
data=data[data['day'].isin([6,7])]
gc.collect()

base_raw_numerical_cols=['shop_review_positive_rate','shop_score_delivery','shop_score_description','shop_score_service']
base_cate_features=['item_price_level','item_sales_level','item_collected_level', 'user_age_level','user_star_level',
               'context_page_id','shop_review_num_level','shop_star_level','day','hour','minute','item_category_list',
                    'predict_category_property','item_id','shop_id','user_id','item_property_list']

def RandomSelectItemId(random_seed,frac,train):
    np.random.seed(random_seed)
    items=train['item_id'].unique().tolist()
    idx = np.random.randint(0,len(items),size=(int(len(items)*frac)))
    select_item_id=[]
    for i in idx:
        select_item_id.append(items[int(i)])
    return select_item_id


def lgb_pre(data_df,num_cols,cate_cols,submission=True):
    data_df=data_df[cate_cols+num_cols+['is_trade']+['instance_id']]
    for i in cate_cols:
        if i not in ['day','minute','hour']:
            data_df[i]=data_df[i].astype('category').cat.codes
    predictors=cate_cols+num_cols
    target = 'is_trade'
    
    offline_test=data_df[(data_df['day']==7) & (data_df['hour']==11)]
    online_testa=data_df[data_df['is_trade']==-1]
    online_testb=data_df[data_df['is_trade']==-2]
    for random in range(5):
        offline_train=data_df[(data_df['day'].isin([6])) | ((data_df['day']==7) & (data_df['hour'].isin([0,1,2,3,4,5,6,7,8,9,10])))]
        offline_train=offline_train.reset_index()
        with open('../data/TestB_Ids_random_'+str(random)+'.pickle','rb') as output:
            indexs=pickle.load(output)
        offline_train=offline_train[offline_train['index'].isin(indexs)]
        offline_train.drop(['index'],axis=1,inplace=True)
        print ('This is ok?*************************************')
        lgbm=lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=30, reg_alpha=0.0, reg_lambda=1,
            max_depth=9, n_estimators=1500, objective='binary',
            subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
            learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=-1)
    #lgbm=lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=30, max_depth=3, n_estimators=1200,objective='binary',
    #                       learning_rate=0.05)
        lgbm.fit(offline_train[predictors],offline_train[target],
                 eval_set=[(offline_train[predictors],offline_train[target]),(offline_test[predictors],offline_test[target])],
                 eval_metric='binary_logloss',early_stopping_rounds=20)
        #sort_={}
        #for i,j in zip(cate_cols+num_cols,lgbm.feature_importances_):
        #    sort_[i]=j
        #sort_=sorted(sort_.items(),key=lambda x:x[1],reverse=True)
        #for i in sort_:
        #    print (i)
        off_pre=lgbm.predict_proba(offline_test[predictors])[:,1]
        pre=lgbm.predict_proba(online_testb[predictors])[:,1]
        score=log_loss(offline_test[target],off_pre)
        print ('Off_score:',score)
        print ('OffTrue_mean:',np.mean(offline_test[target]))
        print ('OffPre_mean:',np.mean(off_pre))
        print ('OnTrue_Mean:',0.03575892237189279)
        print ('OnPre_Mean',pre.mean())
        if submission:
            with open('../data/Pre.pickle','wb') as output:
                pickle.dump(pre,output)
            sub=pd.DataFrame()
            sub['instance_id']=online_testb['instance_id']
            sub['predicted_score']=pre
            sub.to_csv('../submission/Test_B_submission_bootstrap_'+str(random)+'.txt',index=False,sep=' ')
            sub_off=pd.DataFrame()
            sub_off['instance_id']=offline_test['instance_id']
            sub_off['predicted_score']=off_pre
            sub_off.to_csv('../submission/offline_submission_bootstrap_'+str(random)+'.txt',index=False,sep=' ')
        
lgb_pre(data,[col for col in data.columns if col not in base_cate_features+['instance_id','is_trade','context_id','context_timestamp']],base_cate_features)