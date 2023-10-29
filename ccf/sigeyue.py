import numpy as np
import pandas as pd
import os
#off:0.580220962942851,
from tqdm import tqdm, tqdm_notebook
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import datetime
import matplotlib.pyplot as plt
import time
import lightgbm as lgb
import xgboost as xgb
import warnings
nowTime=datetime.datetime.now().strftime('%Y%m%d%H:%M')
from scipy.stats import mode
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 1000)
pd.set_option("display.max_colwidth",1000)
pd.set_option('display.width',1000)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

path=r'C:\Users\cqutb301\Downloads\phw_model_lizi\chnegyonghce/'
train_sales  = pd.read_csv(path+'train_sales_data.csv')#43296
train_search = pd.read_csv(path+'train_search_data.csv')
train_user   = pd.read_csv(path+'train_user_reply_data.csv')
evaluation_public = pd.read_csv(path+'evaluation_public.csv')#7216
submit_example= pd.read_csv(path+'submit_example.csv')

data = pd.concat([train_sales, evaluation_public],ignore_index=True)
data = data.merge(train_search, 'left', on=['province', 'adcode', 'model', 'regYear', 'regMonth'])
data = data.merge(train_user, 'left', on=['model', 'regYear', 'regMonth'])
data['label'] = data['salesVolume']
data['id'] = data['id'].fillna(0).astype(int)
data['bodyType'] = data['model'].map(train_sales.drop_duplicates('model').set_index('model')['bodyType'])
train_2016 = train_sales.loc[train_sales['regYear'] == 2016]
train_2017 = train_sales.loc[train_sales['regYear'] == 2017]
train_2016.rename(columns={'salesVolume': 'sales_2016'}, inplace=True)
train_2017.rename(columns={'salesVolume': 'sales_2017'}, inplace=True)
train_2016.drop('regYear', axis=1, inplace=True)
train = pd.merge(train_2017, train_2016, on=['province', 'adcode', 'model', 'bodyType', 'regMonth'], how='left')
# province_model_1_12 = train.groupby(['model', 'province']).agg({'sales_2017': 'sum', 'sales_2016': 'sum'}).reset_index()
# province_model_1_12['12_province_model'] = province_model_1_12['sales_2017'] / province_model_1_12['sales_2016']
# data = pd.merge(data, province_model_1_12[['province','model','12_province_model']],'left',on=['province','model'])
data['model']=pd.Categorical(data['model']).codes
data['province']=pd.Categorical(data['province']).codes
#LabelEncoder
for i in ['bodyType', 'model']:
    data[i] = data[i].map(dict(zip(data[i].unique(), range(data[i].nunique()))))
data['mt'] = (data['regYear'] - 2016) * 12 + data['regMonth']


data2016=data.loc[data['regYear']==2016]##15840
data2017=data.loc[data['regYear']==2017].reset_index(drop=True)##15840

data2016_2017=pd.concat([data2016,data2017],axis=0)


def score(data, pred='pred_label', label='label', group='model'):
    data['pred_label'] = data['pred_label'].apply(lambda x: 0 if x < 0 else x).round().astype(int)
    data_agg = data.groupby('model').agg({
        pred:  list,
        label: [list, 'mean']
    }).reset_index()
    data_agg.columns = ['_'.join(col).strip() for col in data_agg.columns]
    nrmse_score = []
    for raw in data_agg[['{0}_list'.format(pred), '{0}_list'.format(label), '{0}_mean'.format(label)]].values:
        nrmse_score.append(
            mse(raw[0], raw[1]) ** 0.5 / raw[2]
        )
    # print(1 - np.mean(nrmse_score))
    return 1 - np.mean(nrmse_score)

def get_stat_feature(df_):
    df = df_.copy()
    stat_feat = []
    df['model_adcode'] = df['adcode'] + df['model']
    df['model_adcode_mt'] = df['model_adcode'] * 100 + df['mt']

    df['label_model_rank'] = df.groupby(['province', 'model', 'regMonth'])['label'].agg('rank')
    bodytypeonehot = pd.DataFrame(pd.get_dummies(df['bodyType'],prefix='A'))
    df = pd.concat([df, bodytypeonehot], axis=1)
    bodytypeonehot_index = bodytypeonehot.columns.tolist()
    for item in bodytypeonehot_index:
        stat_feat.append(item)
    provinceonehot = pd.DataFrame(pd.get_dummies(df['province'], prefix='B'))
    df = pd.concat([df, provinceonehot], axis=1)
    provinceonehot_index = provinceonehot.columns.tolist()
    for item in provinceonehot_index:
        stat_feat.append(item)
    for col in tqdm(['label','popularity']):
        # shift
        for i in [1,2,3,4,5,6,7,8]:
            stat_feat.append('shift_model_adcode_mt_{}_{}'.format(col,i))
            df['model_adcode_mt_{}_{}'.format(col,i)] = df['model_adcode_mt'] + i
            df_last = df[~df[col].isnull()].set_index('model_adcode_mt_{}_{}'.format(col,i))
            df['shift_model_adcode_mt_{}_{}'.format(col,i)] = df['model_adcode_mt'].map(df_last[col])
    for col in tqdm(['label','model']):
        # shift
        for i in [1,2,3,4,5,6,7,8]:
            stat_feat.append('shift_model_adcode_mt_{}_{}'.format(col,i))
            df['model_adcode_mt_{}_{}'.format(col,i)] = df['model_adcode_mt'] + i
            df_last = df[~df[col].isnull()].set_index('model_adcode_mt_{}_{}'.format(col,i))
            df['shift_model_adcode_mt_{}_{}'.format(col,i)] = df['model_adcode_mt'].map(df_last[col])

    for col in tqdm(['label','bodyType']):
        # shift
        for i in [1,2,3,4,5,6,7,8]:
            stat_feat.append('shift_model_adcode_mt_{}_{}'.format(col,i))
            df['model_adcode_mt_{}_{}'.format(col,i)] = df['model_adcode_mt'] + i
            df_last = df[~df[col].isnull()].set_index('model_adcode_mt_{}_{}'.format(col,i))
            df['shift_model_adcode_mt_{}_{}'.format(col,i)] = df['model_adcode_mt'].map(df_last[col])

    return df

def get_train_oneall_lgb(data):
    print('LGB:..............................')
    features = [x for x in data if x not in ['forecastVolum', 'label']]
    trian_inx=data['mt'].between(11,22)
    test_inx=data['mt'].between(23,25)
    test_x=data2016_2017[test_inx]
    # val_index=data['mt'].between(20,21)
    model=lgb.LGBMRegressor( num_leaves=31, reg_alpha=0.25, reg_lambda=0.25, objective='mse',
                                max_depth=-1, learning_rate=0.05, min_child_samples=5, random_state=2019,
                                n_estimators=2000, subsample=0.9, colsample_bytree=0.7)

    model.fit(data2016_2017[trian_inx][features],data2016_2017[trian_inx]['label'])
    test_x['pred_label']=model.predict(data2016_2017[test_inx][features])

    mae=mean_absolute_error(data2016_2017[test_inx]['label'], test_x['pred_label'])
    mymse = mse(data2016_2017[test_inx]['label'], test_x['pred_label'])
    nrmse=score(test_x)
    print('off_MAE:{}'.format(mae))
    print('off_MSE:{}'.format(mymse))
    print('off_NRMSE:{}'.format(nrmse))


def get_train_oneall_xgb(data):
    print('XGB:...........................')
    features = [x for x in data if x not in ['forecastVolum', 'label']]
    trian_inx=data['mt'].between(11,22)
    test_inx=data['mt'].between(23,24)
    test_x=data2016_2017[test_inx]
    # val_index=data['mt'].between(20,21)
    model=xgb.XGBRegressor(max_depth=5 , learning_rate=0.05, n_estimators=3000,
                                objective='reg:gamma', tree_method = 'hist',subsample=0.9,
                                colsample_bytree=0.7, min_child_samples=5,eval_metric ='rmse')
    model.fit(data2016_2017[trian_inx][features],data2016_2017[trian_inx]['label'])
    test_x['pred_label']=model.predict(data2016_2017[test_inx][features])
    mae=mean_absolute_error(data2016_2017[test_inx]['label'], test_x['pred_label'])
    mymse = mse(data2016_2017[test_inx]['label'], test_x['pred_label'])
    nrmse=score(test_x)
    print('off_MAE:{}'.format(mae))
    print('off_MSE:{}'.format(mymse))
    print('off_NRMSE:{}'.format(nrmse))


def get_train_onebyone_lgb(data):
    print('LGB_one_one:..............................')
    features = [x for x in data if x not in ['forecastVolum', 'label']]
    trian_inx=data['mt'].between(11,22)
    test_inx1=data['mt']==23
    test_inx=data['mt'].between(23,25)
    test_x=data2016_2017[test_inx]
    # val_index=data['mt'].between(20,21)
    model=lgb.LGBMRegressor( num_leaves=31, reg_alpha=0.25, reg_lambda=0.25, objective='mse',
                                max_depth=-1, learning_rate=0.05, min_child_samples=5, random_state=2019,
                                n_estimators=2000, subsample=0.9, colsample_bytree=0.7)
    model.fit(data2016_2017[trian_inx][features],data2016_2017[trian_inx]['label'])
    pred1=pd.DataFrame(model.predict(data2016_2017[test_inx1][features]))


    train_inx2=data['mt'].between(11,23)
    last_index=data['mt']==23
    test_inx2=data['mt']==24

    data2016_2017[last_index]['forecastVolum']=pred1.values
    # data2016_2017[last_index]['label'] = pred1.values

    features2=[x for x in data2016_2017.columns if x not in ['label']]
    model2 = lgb.LGBMRegressor(n_estimators=3000)
    model2.fit(data2016_2017[train_inx2][features2], data2016_2017[train_inx2]['label'])
    pred2 = pd.DataFrame(model2.predict(data2016_2017[test_inx2][features2]))
    pred_all=pd.concat([pred1,pred2],axis=0)
    test_x['pred_label']=pred_all.values
    mae=mean_absolute_error(pred_all.values,data2016_2017[test_inx]['label'])
    mymse = mse(pred_all.values,data2016_2017[test_inx]['label'])
    nrmse=score(test_x)
    print('off_MAE:{}'.format(mae))
    print('off_MSE:{}'.format(mymse))
    print('off_NRMSE:{}'.format(nrmse))
if __name__=="__main__":
    # data2016_2017=get_stat_feature(data2016_2017)
    # print(data2016_2017)
    get_train_oneall_lgb(data2016_2017)
    get_train_oneall_xgb(data2016_2017)
    get_train_onebyone_lgb(data2016_2017)