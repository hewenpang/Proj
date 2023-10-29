import sys
import numpy as np
import pandas as pd
import os
import gc
import seaborn as sns
# 0.2812262764324608
# [14:59:33] Tree method is selected to be 'hist', which uses a single updater grow_fast_histmaker.
# valid mean: 388.41415
# true  mean: 899.8204545454546
# test  mean: 406.43002

#0.61219084000
from tqdm import tqdm, tqdm_notebook
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import LabelEncoder
import datetime
from sklearn.model_selection import GridSearchCV
import time
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
nowTime = datetime.datetime.now().strftime('%Y%m%d%H:%M')
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
import tsfresh as ts
from tsfresh import extract_features
from tsfresh import extract_relevant_features
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 1000)
pd.set_option("display.max_colwidth",1000)
pd.set_option('display.width',1000)
from mlxtend.regressor import StackingRegressor
from sklearn.ensemble import GradientBoostingRegressor
unstack_data={}
yanhaicitylist=['大连','锦州','秦皇岛','天津','烟台','威海','青岛','日照','连云港','上海','杭州','宁波','台州','温州','福州','厦门','泉州',
                '汕头','广州','湛江','深圳','珠海','北海']
yixiancitylist=['浙江','江苏','广东','福建','山东','辽宁','新疆','湖北','河北','吉林']
erxiancitylist=['海南','湖南','河南','山西','黑龙江','宁夏','安徽','重庆','青海','四川','西藏','陕西','云南','江西']
shengcahncity=['北京','上海','吉林','四川','广东']
from pandas import DataFrame
from pandas import concat
varlist=[]
for i in np.arange(28):
    varlist.append('var1(t-{})'.format(i))
varlist[0]='var1(t)'

def series_to_supervised(data, n_in=1, n_out=1, dropnan=False):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
def issea(x):
    if x in yanhaicitylist:
        return 9999
    else:
        return 0
def citytype(x):
    if x in yixiancitylist:
        return 20000
    elif x in erxiancitylist:
        return 1000
    else:
        return 10
def isshengchan(x):
    if x in shengcahncity:
        return 9999
    else:
        return 0
name=[]
def getHistoryIncrease(dataset, step=1, wind=1, col='salesVolume'):
    if col not in unstack_data.keys():
        res = []
        bar = tqdm(dataset['province'].unique(), desc='history increase')
        for i in bar:
            for j in dataset['model'].unique():
                msk = (dataset['province'] == i) & (dataset['model'] == j)
                df = dataset[msk].copy().reset_index(drop=True)
                df = df[['mt', col]].set_index('mt').T
                df['province'] = i
                df['model'] = j
                res.append(df)
        res = pd.concat(res).reset_index(drop=True)
        unstack_data[col] = res.copy()
    res = unstack_data[col].copy()
    res_ = res.copy()
    for i in range(step + wind + 1, 20):
        res_[i] = (res[i - step] - res[i - (step + wind)]) / res[i - (step + wind)]
    for i in range(1, step + wind + 1):
        res_[i] = np.NaN
    res = res_.set_index(['province', 'model']).stack().reset_index()
    res.rename(columns={0: '{}_last{}_{}_increase'.format(col, step, wind)}, inplace=True)
    name.append('{}_last{}_{}_increase'.format(col, step, wind))
    dataset = pd.merge(dataset, res, 'left', on=['province', 'model', 'mt'])
    return dataset,name
def province_model(x):
    return str(x['province'])+str(x['model'])
def province_bodyType(x):
    return str(x['province'])+str(x['bodyType'])
def model_bodyType(x):
    return str(x['model'])+str(x['bodyType'])
def get_st(x):
    if x<=700:
        return 1
    elif x<=1445:
        return 2
    elif x<=2816:
        return 3
    else:
        return 0
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
train_sales  = pd.read_csv('/Users/phw/chengche/Train/train_sales_data.csv')
train_search = pd.read_csv('/Users/phw/chengche/Train/train_search_data.csv')
train_user   = pd.read_csv('/Users/phw/chengche/Train/train_user_reply_data.csv')
evaluation_public = pd.read_csv('/Users/phw/chengche/evaluation_public.csv')
submit_example    = pd.read_csv('/Users/phw/chengche/submit_example.csv')
data = pd.concat([train_sales, evaluation_public], ignore_index=True)
data = data.merge(train_search, 'left', on=['province', 'adcode', 'model', 'regYear', 'regMonth'])
data = data.merge(train_user, 'left', on=['model', 'regYear', 'regMonth'])
data['label'] = data['salesVolume']
data['id'] = data['id'].fillna(0).astype(int)
data['bodyType'] = data['model'].map(train_sales.drop_duplicates('model').set_index('model')['bodyType'])
popularity=pd.read_csv('/Users/phw/chengche/popularity54.csv',index_col=0)
data['popularity']=pd.concat([data['popularity'].iloc[0:31680],popularity['popularity']],ignore_index=True)
newsreply=pd.read_csv('/Users/phw/chengche/Train/to_Pang.csv',index_col=0)
data['newsReplyVolum']=pd.concat([data['newsReplyVolum'].iloc[0:31680],newsreply['newsReplyVolum']],ignore_index=True)
carcomment=pd.read_csv('/Users/phw/chengche/Train/to_Pang.csv',index_col=0)
data['carCommentVolum']=pd.concat([data['carCommentVolum'].iloc[0:31680],carcomment['carCommentVolum']],ignore_index=True)
train_2016 = train_sales.loc[train_sales['regYear'] == 2016]
train_2017 = train_sales.loc[train_sales['regYear'] == 2017]
train_2016.rename(columns={'salesVolume': 'sales_2016'}, inplace=True)
train_2017.rename(columns={'salesVolume': 'sales_2017'}, inplace=True)
train_2016.drop('regYear', axis=1, inplace=True)
train = pd.merge(train_2017, train_2016, on=['province', 'adcode', 'model', 'bodyType', 'regMonth'], how='left')
province_model_1_12 = train.groupby(['model', 'province']).agg({'sales_2017': 'sum', 'sales_2016': 'sum'}).reset_index()
province_model_1_12['12_province_model'] = province_model_1_12['sales_2017'] / province_model_1_12['sales_2016']
data = pd.merge(data, province_model_1_12[['province','model','12_province_model']],'left',on=['province','model'])
data['province']=pd.Categorical(data['province']).codes
#LabelEncoder
for i in ['bodyType', 'model']:
    data[i] = data[i].map(dict(zip(data[i].unique(), range(data[i].nunique()))))
data['mt'] = (data['regYear'] - 2016) * 12 + data['regMonth']


col, col2,col3= ['popularity','adcode','model']
col_per = np.percentile(data[col], (25, 75))
diff = 1.5 * (col_per[1] - col_per[0])
col_per_in = (data[col] >= col_per[0] - diff) & (data[col] <= col_per[1] + diff)

col_per2 = np.percentile(data[col2], (25, 75))
diff2 = 1.5 * (col_per2[1] - col_per2[0])
col_per_in2 = (data[col2] >= col_per2[0] - diff2) & (data[col2] <= col_per2[1] + diff2)

col_per3 = np.percentile(data[col3], (25, 75))
diff3 = 1.5 * (col_per3[1] - col_per3[0])
col_per_in3 = (data[col3] >= col_per3[0] - diff3) & (data[col3] <= col_per3[1] + diff3)


data.loc[~col_per_in, col] = col_per.mean()
data.loc[~col_per_in2, col2] = col_per2.mean()
data.loc[~col_per_in3, col3] = col_per3.mean()


# datafeature=data.iloc[0:31680]
# y=datafeature['label']
# datafeature=datafeature[['label','mt']]
# datafeature['id']=np.arange(len(datafeature))
# extract_feature=extract_relevant_features(datafeature,y,column_id='id',column_sort='mt')
# extract_feature.to_csv('extract_feature_label.csv')
def get_stat_feature(df_):
    df = df_.copy()
    stat_feat = []
    df['model_adcode'] = df['adcode'] + df['model']
    df['model_adcode_mt'] = df['model_adcode'] * 100 + df['mt']
    # df['model_adcode_mt_neg']=df['model_adcode'] * 100 *df['mt']
    df['province_model']=df['province']+df['model']
    df['province_bodytype']=df['province']+df['bodyType']
    df['province_adcode']=df['province']+df['adcode']

    df['province_mt']=df['province']*100+df['mt']
    # df['province_mt_neg'] = df['province'] * 100 *df['mt']

    df['pop_adcode']=df['popularity']+df['adcode']
    # df['pop_model']=df['popularity']+df['model']
    # df['pop_bodytype']=df['popularity']+df['bodyType']
    df['pop_mt'] = df['popularity'] * 100 + df['mt']
    # df['pop_mt_neg'] = df['popularity'] * 100 * df['mt']
    df['pop_adcode_mt']=df['pop_adcode']*100+df['mt']
    df['car_mt']=df['carCommentVolum']*100+df['mt']
    df['news_mt']=df['newsReplyVolum']*100+df['mt']
    # df['newsReplyVolum_adcode'] = df['newsReplyVolum'] + df['adcode']
    # df['newsReplyVolum_adcode_mt'] = df['newsReplyVolum_adcode'] * 100 + df['mt']
    # df['newsReplyVolum_mt'] = df['newsReplyVolum'] * 100 + df['mt']
    # df['carCommentVolum_adcode'] = df['carCommentVolum'] + df['adcode']
    # df['carCommentVolum_adcode_mt'] = df['carCommentVolum_adcode'] * 100 + df['mt']
    # df['carCommentVolum_mt'] = df['newsReplyVolum'] * 100 + df['mt']
    # df['model_weight'] = df.groupby('model')['label'].transform('mean')
    # df['is12']=df['regMonth'].map(lambda x: 1 if x==12 else 0)
    # df['pop_adcode_mt_neg'] = df['pop_adcode'] * 100 * df['mt']
    # pivot = pd.pivot_table(df, index=['adcode'], values='label', aggfunc=np.sum)
    # pivot = pd.DataFrame(pivot).rename(columns={'label': 'adcode_sales_sum'}).reset_index()
    # df = pd.merge(df, pivot, on='adcode', how='left')
    # stat_feat.append('adcode_sales_sum')
    # #
    # pivot = pd.pivot_table(df, index=['adcode'], values='popularity', aggfunc=np.sum)
    # pivot = pd.DataFrame(pivot).rename(columns={'popularity': 'adcode_pop_sum'}).reset_index()
    # df = pd.merge(df, pivot, on='adcode', how='left')
    # stat_feat.append('adcode_pop_sum')
    # df['mdoel_adcode_province_mt']=df['model_adcode_mt']+df['province_mt']
    # df['model_adcode_pop_mt']=df['model_adcode_mt']+df['pop_mt']
    # df['mdoel_adcode_pop_adcode']=df['model_adcode_mt']+df['pop_adcode']

    # df['province_pop_mt']=df['province_mt']+df['pop_mt']
    # df['province_pop_adcode_mt']=df['province_mt']+df['pop_adcode_mt']

    # df['pop_mt_adcode_mt']=df['pop_mt']+df['pop_adcode_mt']

    # df['bodytype_adcode']=df['bodyType']+df['adcode']
    # df['bodytype_mt']=df['bodyType']*100+df['mt']
    # df['bodytype_adcode_mt']=df['bodytype_adcode']*100+df['mt']




    # day_map = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    # df['daycount'] = df['regMonth'].map(day_map)
    # df['pop_sum']=df['popularity']*df['daycount']
    # df['label_count']=df['label']*df['daycount']
    # df['label_mean']=df['label']/df['daycount']
    # df['pop_count']=df['popularity']/df['daycount']
    # df['pop_mean']=df['popularity']/df['daycount']
    # df['bodytype_model']=df['bodyType']+df['model']
    # df['bodytype_adcode']=df['bodyType']+df['adcode']
    # df['bodytype_mt']=df['bodyType']*100+df['mt']
    # df['label_rank']=df.groupby(['regYear','province','model'])['label'].agg('rank')
    # df['pop_rank']=df.groupby(['regYear','province','model'])['popularity'].agg('rank')
    # df['citytype'] = df['province'].map(lambda x: citytype(x))
    # df['issea'] = df['province'].map(lambda x: issea(x))
    # df['isshengchan'] = df['province'].map(lambda x: isshengchan(x))
    # df['bt_ry_mean'] = df.groupby(['bodyType', 'province'])['salesVolume'].transform('mean')
    # df['ad_ry_mean'] = df.groupby(['adcode', 'province'])['salesVolume'].transform('mean')
    # df['md_ry_mean'] = df.groupby(['model', 'province'])['salesVolume'].transform('mean')
    #
    # df['popularity_type'] = df['popularity'].map(lambda x: get_st(x))
    # df['pop_rank']=df.groupby(['province','model','regMonth'])['popularity'].agg('rank')
    # df['dayall'] = (df['regYear'] - 2016) * 12 * 30 + (df['regMonth'] * 30)
    # day_map = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    # df['daycount'] = df['regMonth'].map(day_map)
    # df['label_rank']=df.groupby(['province','adcode','regMonth'])['label'].agg('rank')
    # df['label_mean']=df.groupby(['province','model','regYear','regMonth'])['label'].transform('mean')
    # df['label_std'] = df.groupby(['province','model','regYear','regMonth'])['label'].transform('std')
    #
    # df['pop_mean']=df.groupby(['province','model','regYear','regMonth'])['popularity'].transform('mean')
    # df['pop_std'] = df.groupby(['province','model','regYear','regMonth'])['popularity'].transform('std')
    # df['pop_model_rank'] = df.groupby(['province', 'model', 'regMonth'])['popularity'].agg('rank')
    # df['province_model']=df.apply(lambda x:province_model(x),axis=1)
    # df['province_model']=pd.Categorical(df['province_model']).codes
    # df['province_bodyType']=df.apply(lambda x:province_bodyType(x),axis=1)
    # df['province_bodyType']=pd.Categorical(df['province_bodyType']).codes
    # df['model_bodyType']=df.apply(lambda x:model_bodyType(x),axis=1)
    # df['model_bodyType']=pd.Categorical(df['model_bodyType']).codes

    # df['weight_pop_mean'] = df.groupby(['province', 'regYear', 'regMonth'])['popularity'].transform('mean') * (1 / df['mt'])
    # df['weight_label_mean'] = df.groupby(['province', 'regYear', 'regMonth'])['label'].transform('mean') * (1 / df['mt'])
    # df['label_model_rank'] = df.groupby(['province', 'model', 'regMonth'])['label'].agg('rank')
    # df['pop_model_rank']=df.groupby(['province', 'model', 'regMonth'])['popularity'].agg('rank')
    # df['label_model_mean']=df.groupby(['province','model','regMonth'])['label'].transform('mean')
    for col in tqdm(['label','popularity']):
        # shift
        for i in [1,2,3,4,5,6,7,8,9,10]:
            stat_feat.append('shift_model_adcode_mt_{}_{}'.format(col,i))
            df['model_adcode_mt_{}_{}'.format(col,i)] = df['model_adcode_mt'] + i
            df_last = df[~df[col].isnull()].set_index('model_adcode_mt_{}_{}'.format(col,i))
            df['shift_model_adcode_mt_{}_{}'.format(col,i)] = df['model_adcode_mt'].map(df_last[col])
    # for col in tqdm(['newsReplyVolum','carCommentVolum']):
    #     # shift
    #     for i in [1,2,3,4,5,6,7,8,9,10]:
    #         stat_feat.append('shift_model_adcode_mt_{}_{}'.format(col,i))
    #         df['model_adcode_mt_{}_{}'.format(col,i)] = df['model_adcode_mt'] + i
    #         df_last = df[~df[col].isnull()].set_index('model_adcode_mt_{}_{}'.format(col,i))
    #         df['shift_model_adcode_mt_{}_{}'.format(col,i)] = df['model_adcode_mt'].map(df_last[col])
    # for i in [4,6,9]:
    #     col='last_{}_label_std'.format(i)
    #     stat_feat.append(col)
    #     df[col]=df[['label']].apply(lambda x:x.rolling(i).std().shift(1),axis=0)
    # for i in [4,6,9]:
    #     col='last_{}_label_mean'.format(i)
    #     stat_feat.append(col)
    #     df[col]=df[['label']].apply(lambda x:x.rolling(i).mean().shift(1),axis=0)
    # for i in [4,6,9]:
    #     col='last_{}_pop_std'.format(i)
    #     stat_feat.append(col)
    #     df[col]=df[['popularity']].apply(lambda x:x.rolling(i).std().shift(1),axis=0)
    # for i in [4,6,9]:
    #     col='last_{}_pop_mean'.format(i)
    #     stat_feat.append(col)
    #     df[col]=df[['popularity']].apply(lambda x:x.rolling(i).mean().shift(1),axis=0)
    # for i in [4,6,9]:
    #     col='last_{}_pop_kurt'.format(i)
    #     stat_feat.append(col)
    #     df[col]=df[['popularity']].apply(lambda x:x.rolling(i).kurt().shift(1),axis=0)
    # for i in [4,6,9]:
    #     col='last_{}_pop_skew'.format(i)
    #     stat_feat.append(col)
    #     df[col]=df[['popularity']].apply(lambda x:x.rolling(i).skew().shift(1),axis=0)
    return df,stat_feat

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
    print(1 - np.mean(nrmse_score))
    return 1 - np.mean(nrmse_score)
def get_model_type(train_x,train_y,valid_x,valid_y,m_type='lgb',eval_metric='mae'):
    if m_type == 'lgb':
        model = lgb.LGBMRegressor(
                                num_leaves=2**5-1, reg_alpha=0.25, reg_lambda=0.25, objective='mse',
                                max_depth=-1, learning_rate=0.05, min_child_samples=5, random_state=2019,
                                n_estimators=2000, subsample=0.9, colsample_bytree=0.7,
                                )
        model.fit(train_x, train_y,
              eval_set=[(train_x, train_y),(valid_x, valid_y)],
              categorical_feature=cate_feat,
              early_stopping_rounds=100, verbose=100)
    elif m_type == 'xgb':
        model = xgb.XGBRegressor(
                                booster='gbtree',
                                max_depth=5 , learning_rate=0.05, n_estimators=2000,
                                objective='reg:gamma', tree_method ='hist',subsample=0.9,
                                colsample_bytree=0.7, min_child_samples=5,eval_metric =eval_metric
                                )
        model.fit(train_x, train_y,
              eval_set=[(train_x, train_y),(valid_x, valid_y)],
              early_stopping_rounds=100, verbose=100)
    return model
def get_train_model(df_, m, m_type='lgb',is_log=0):
    df = df_.copy()
    # 数据集划分
    st = 13
    all_idx   = (df['mt'].between(st , m-1))
    train_idx = (df['mt'].between(st , m-5))
    valid_idx = (df['mt'].between(m-4, m-4))
    test_idx  = (df['mt'].between(m, m ))
    print('all_idx  :',st ,m-1)
    print('train_idx:',st ,m-5)
    print('valid_idx:',m-4,m-4)
    print('test_idx :',m  ,m  )
    # 最终确认
    train_x = df[train_idx][features]
    train_y = df[train_idx]['label']
    valid_x = df[valid_idx][features]
    valid_y = df[valid_idx]['label']
    all_train_x=df[all_idx][features]
    all_train_y=df[all_idx]['label']
    test_x=df[test_idx][features]
    if is_log:
        train_y=np.log1p(train_y)
        valid_y=np.log1p(valid_y)
        all_train_y=np.log1p(all_train_y)
    # get model
    model = get_model_type(train_x,train_y,valid_x,valid_y,m_type,'mae')
    # offline
    df['pred_label'] = model.predict(df[features])
    if is_log:
        df['pred_label']=np.e**(df['pred_label'].values)-1
    best_score = score(df[valid_idx])
    # online
    if m_type == 'lgb':
        model.n_estimators = model.best_iteration_ + 100
        model.fit(df[all_idx][features], all_train_y, categorical_feature=cate_feat)
    elif m_type == 'xgb':
        # parameters = {
        #     # 'max_depth': [5, 10, 15, 20, 25],
        #     # 'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15]
        # #     # 'n_estimators': [500, 1000, 2000, 3000, 5000, 8000],
        # #     # 'min_child_weight': [0, 2, 5, 10, 20],
        # #     # 'max_delta_step': [0, 0.2, 0.6, 1, 2],
        # #     # 'subsample': [0.6, 0.7, 0.8, 0.85, 0.95],
        # #     # 'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
        # #     # 'reg_alpha': [0, 0.25, 0.5, 0.75, 1],
        # #     # 'reg_lambda': [0.2, 0.4, 0.6, 0.8, 1],
        # #     # 'scale_pos_weight': [0.2, 0.4, 0.6, 0.8, 1]
        # }
        # gsearch = GridSearchCV(model, param_grid=parameters, scoring='neg_mean_absolute_error', cv=3)
        # gsearch.fit(train_x, train_y)
        # print("Best score: %0.3f" % gsearch.best_score_)
        # print("Best parameters set:")
        # best_parameters = gsearch.best_estimator_.get_params()
        # for param_name in sorted(parameters.keys()):
        #     print("\t%s: %r" % (param_name, best_parameters[param_name]))
        model.n_estimators = model.best_iteration + 100
        model.fit(df[all_idx][features], all_train_y)
    df['forecastVolum'] = model.predict(df[features])
    if is_log:
        df['forecastVolum']=np.e**(df['forecastVolum'].values)-1
    print('valid mean:',df[valid_idx]['pred_label'].mean())
    print('true  mean:',df[valid_idx]['label'].mean())
    print('test  mean:',df[test_idx]['forecastVolum'].mean())
    print("best_score:",best_score)
    # 阶段结果
    sub = df[test_idx][['id']]
    sub['forecastVolum'] = df[test_idx]['forecastVolum'].apply(lambda x: 0 if x < 0 else x).round().astype(int)
    return sub,df[valid_idx]['pred_label']



for month in [25,26, 27, 28]:
    m_type = 'xgb'

    data_df, stat_feat = get_stat_feature(data)
    print(data_df)
    num_feat = ['regYear'] + stat_feat+['province','province_model','province_bodytype','province_adcode','province_mt'
                                        ,'pop_mt','pop_adcode_mt']
    cate_feat = ['adcode', 'bodyType', 'model','regMonth']

    if m_type == 'lgb':
        for i in cate_feat:
            data_df[i] = data_df[i].astype('category')
    elif m_type == 'xgb':
        lbl = LabelEncoder()
        for i in tqdm(cate_feat):
            data_df[i] = lbl.fit_transform(data_df[i].astype(str))

    features = num_feat + cate_feat
    print(len(features), len(set(features)))

    sub, val_pred = get_train_model(data_df, month, m_type)
    data.loc[(data.regMonth == (month - 24)) & (data.regYear == 2018), 'salesVolume'] = sub['forecastVolum'].values
    data.loc[(data.regMonth == (month - 24)) & (data.regYear == 2018), 'label'] = sub['forecastVolum'].values
sub = data.loc[(data.regMonth >= 1) & (data.regYear == 2018), ['id', 'salesVolume']]
sub.columns = ['id', 'forecastVolum']
sub[['id', 'forecastVolum']].round().astype(int).to_csv("XGB"+'CCF_sales.csv', index=False)


# y_pred=pd.read_csv('/Users/phw/chengche/XGB612mae.csv')['forecastVolum']
# eva=pd.read_csv('/Users/phw/chengche/evaluation_public.csv')
# data[data.regYear == 2018]['label'] = y_pred
# std = data[data.regYear <=2018].groupby(['adcode', 'model'])['label'].std().reset_index().rename(
#     columns={'label': 'std'}, )
# mean = data[data.regYear >= 2017].groupby(['adcode', 'model'], as_index=False)['label'].mean()
# std = std.merge(mean, on=['adcode', 'model'], how='left')
# std_top = std.loc[std['std'] >=295.7, :]
# mean_label = \
# pd.merge(left=data[data.regYear == 2018], right=std_top, how='left', on=['adcode', 'model'], suffixes=('', '_'))[
#     'label_'].values
# y_pred = np.where(np.isnan(mean_label), y_pred, mean_label)
# eva['forecastVolum']=y_pred
# eva[['id','forecastVolum']].round().astype(int).to_csv('delay'+'CCF_sales.csv', index=False)

#队长后处理
# train_sales  = pd.read_csv('/Users/phw/chengche/Train/train_sales_data.csv')
# pre=pd.read_csv('/Users/phw/chengche/data2018.csv')
# pre['regYear']=2018
# pre['salesVolume']=pre['forecastVolum']
# pre.drop('forecastVolum',axis=1,inplace=True)
# data_all=train_sales.ix[:,['province','model','regMonth','salesVolume','regYear']].append(pre)
# data_all.reset_index(drop=True,inplace=True)
#
# data_all.sort_values(by=['province','model','regYear','regMonth'],inplace=True)
# e=series_to_supervised(data_all[['salesVolume']],27)
# data_all=pd.concat([data_all,e],axis=1)
# data_all=data_all[(data_all['regYear']==2018)&(data_all['regMonth']==4)]
# data_all.reset_index(drop=True,inplace=True)
# #计算2017年均值和三年的标准差
# mean_list=['var1(t-{})'.format(i) for i in range(4,16)]
# std_list=mean_list=['var1(t-{})'.format(i) for i in range(1,29)]+['vat1(t)']
# for i in range(len(data_all)):
#     data_all.ix[i,'mean_lag12']=np.mean(data_all.ix[i,mean_list])
#     data_all.ix[i,'std']=np.std(data_all.ix[i,std_list])
#
# print(data_all.loc[data_all['std']>=1000])
# pre_new=pd.merge(pre,data_all.ix[:,['province','model','mean_lag12','std']],how='left',on=['province','model'])
# #大于标准差阈值的用2017年的均值填充销量
# for i in range(len(pre_new)):
#     if pre_new.ix[i,'std']>1000: #阈值可以更改:
#         pre_new.ix[i,'salesVolume']=pre_new.ix[i,'mean_lag12']
#     else:
#         pass
# pre_new['forecastVolum']=pre_new['salesVolume']
# pre_new.ix[:,['id','forecastVolum']].round().astype(int).to_csv('delay_sub.csv',index=False,encoding='utf-8')

