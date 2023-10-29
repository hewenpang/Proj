
import numpy as np
import pandas as pd
import os
#off:0.580220962942851,
from tqdm import tqdm, tqdm_notebook
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, roc_auc_score
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
yanhaicitylist=['大连','锦州','秦皇岛','天津','烟台','威海','青岛','日照','连云港','上海','杭州','宁波','台州','温州','福州','厦门','泉州',
                '汕头','广州','湛江','深圳','珠海','北海']
yixiancitylist=['浙江','江苏','广东','福建','山东','辽宁','新疆','湖北','河北','吉林']
erxiancitylist=['海南','湖南','河南','山西','黑龙江','宁夏','安徽','重庆','青海','四川','西藏','陕西','云南','江西']
shengcahncity=['北京','上海','吉林','四川','广东']
def province_model(x):
    return str(x['province'])+str(x['model'])
def province_bodyType(x):
    return str(x['province'])+str(x['bodyType'])
def model_bodyType(x):
    return str(x['model'])+str(x['bodyType'])
def shift(x):
    return x.shift(1)
def issea(x):
    if x in yanhaicitylist:
        return 1
    else:
        return 0
def citytype(x):
    if x in yixiancitylist:
        return 2
    elif x in erxiancitylist:
        return 1
    else:
        return 0
def isshengchan(x):
    if x in shengcahncity:
        return 10000
    else:
        return 0
def get_st(x):
    if x<=700:
        return 1
    elif x<=1445:
        return 2
    elif x<=2816:
        return 3
    else:
        return 0

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
# popularity=pd.read_csv('/Users/phw/chengche/popularity54new.csv',index_col=0)
# data['popularity']=pd.concat([data['popularity'].iloc[0:43296],popularity['popularity']],ignore_index=True)

# newsreply=pd.read_csv('/Users/phw/chengche/newsReplyVolum54.csv',index_col=0)
# data['newsReplyVolum']=pd.concat([data['newsReplyVolum'].iloc[0:31680],newsreply['newsReplyVolum']],ignore_index=True)
# carcomment=pd.read_csv('/Users/phw/chengche/carCommentVolum54.csv',index_col=0)
# data['carCommentVolum']=pd.concat([data['carCommentVolum'].iloc[0:31680],carcomment['carCommentVolum']],ignore_index=True)

train_2016 = train_sales.loc[train_sales['regYear'] == 2016]
train_2017 = train_sales.loc[train_sales['regYear'] == 2017]
train_2016.rename(columns={'salesVolume': 'sales_2016'}, inplace=True)
train_2017.rename(columns={'salesVolume': 'sales_2017'}, inplace=True)
train_2016.drop('regYear', axis=1, inplace=True)
train = pd.merge(train_2017, train_2016, on=['province', 'adcode', 'model', 'bodyType', 'regMonth'], how='left')
province_model_1_12 = train.groupby(['model', 'province']).agg({'sales_2017': 'sum', 'sales_2016': 'sum'}).reset_index()
province_model_1_12['12_province_model'] = province_model_1_12['sales_2017'] / province_model_1_12['sales_2016']
data = pd.merge(data, province_model_1_12[['province','model','12_province_model']],'left',on=['province','model'])

data['model']=pd.Categorical(data['model']).codes
data['province']=pd.Categorical(data['province']).codes
#LabelEncoder
for i in ['bodyType', 'model']:
    data[i] = data[i].map(dict(zip(data[i].unique(), range(data[i].nunique()))))
data['mt'] = (data['regYear'] - 2016) * 12 + data['regMonth']



##特征
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
    # for col in tqdm(['label','newsReplyVolum']):
    #     # shift
    #     for i in [1,2,3,4,5,6,7,8]:
    #         stat_feat.append('shift_model_adcode_mt_{}_{}'.format(col,i))
    #         df['model_adcode_mt_{}_{}'.format(col,i)] = df['model_adcode_mt'] + i
    #         df_last = df[~df[col].isnull()].set_index('model_adcode_mt_{}_{}'.format(col,i))
    #         df['shift_model_adcode_mt_{}_{}'.format(col,i)] = df['model_adcode_mt'].map(df_last[col])
    # for col in tqdm(['label']):
    #     # 历史销量数据特征
    #     for i in [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
    #         stat_feat.append('shift_model_adcode_mt_{}_{}'.format(col, i))
    #         df['model_adcode_mt_{}_{}'.format(col, i)] = df[
    #                                                          'model_adcode_mt'] + i  # 新加一列值，等于车型*省*时间+i，寻求i个月前的值，将model_adcode_mt_作为索引
    #         df_last = df[~df[col].isnull()].set_index('model_adcode_mt_{}_{}'.format(col, i))
    #         df['shift_model_adcode_mt_{}_{}'.format(col, i)] = df['model_adcode_mt'].map(
    #             df_last[col])  # 后者索引是31000002开始，前者少i，取前面的匹配后面索引成功，就取值
    # for col in tqdm(['popularity']):
    #     # 历史销量数据特征
    #     for i in [1, 2, 3, 10, 11, 12]:  # popularity只取一部分
    #         stat_feat.append('shift_model_adcode_mt_{}_{}'.format(col, i))
    #         df['model_adcode_mt_{}_{}'.format(col, i)] = df[
    #                                                          'model_adcode_mt'] + i  # 新加一列值，等于车型*省*时间+i，寻求i个月前的值，将model_adcode_mt_作为索引
    #         df_last = df[~df[col].isnull()].set_index('model_adcode_mt_{}_{}'.format(col, i))
    #         df['shift_model_adcode_mt_{}_{}'.format(col, i)] = df['model_adcode_mt'].map(
    #             df_last[col])  # 后者索引是31000002开始，前者少i，取前面的匹配后面索引成功，就取值
    # df["increase16_4"] = (df["shift_model_adcode_mt_label_16"] - df["shift_model_adcode_mt_label_4"]) / df[
    #     "shift_model_adcode_mt_label_16"]  # 同比一年前的增长
    # mean = pd.DataFrame(df.groupby(["model", "mt"]).shift_model_adcode_mt_label_12.agg({"mean_province": "mean",
    #                                                                                     "min_province": "min", }))
    # df = pd.merge(df, mean, on=["model", "mt"], how="left")
    # mean = pd.DataFrame(df.groupby(["model", "mt"]).shift_model_adcode_mt_label_15.agg({"mean_province_15": "mean", }))
    # df = pd.merge(df, mean, on=["model", "mt"], how="left")
    # mean = pd.DataFrame(df.groupby(["model", "mt"]).shift_model_adcode_mt_label_3.agg({"mean_province_3": "mean", }))
    # df = pd.merge(df, mean, on=["model", "mt"], how="left")
    # mean = pd.DataFrame(df.groupby(["model", "mt"]).shift_model_adcode_mt_label_16.agg({"mean_province_16": "mean", }))
    # df = pd.merge(df, mean, on=["model", "mt"], how="left")
    # mean = pd.DataFrame(df.groupby(["model", "mt"]).shift_model_adcode_mt_label_4.agg({"mean_province_4": "mean", }))
    # df = pd.merge(df, mean, on=["model", "mt"], how="left")
    # # 另一种统计方式
    # mean = pd.DataFrame(df.groupby(["adcode", "mt"]).shift_model_adcode_mt_label_15.agg({"mean_Month_15": "mean"}))
    # df = pd.merge(df, mean, on=["adcode", "mt"], how="left")
    # mean = pd.DataFrame(df.groupby(["adcode", "mt"]).shift_model_adcode_mt_label_3.agg({"mean_Month_3": "mean"}))
    # df = pd.merge(df, mean, on=["adcode", "mt"], how="left")
    # mean = pd.DataFrame(df.groupby(["adcode", "mt"]).shift_model_adcode_mt_label_16.agg({"mean_Month_16": "mean"}))
    # df = pd.merge(df, mean, on=["adcode", "mt"], how="left")
    # mean = pd.DataFrame(df.groupby(["adcode", "mt"]).shift_model_adcode_mt_label_4.agg({"mean_Month_4": "mean"}))
    # df = pd.merge(df, mean, on=["adcode", "mt"], how="left")
    # # 基于统计特征的increase，强特
    # df["increase_mean_province_16_4"] = (df["mean_province_16"] - df["mean_province_4"]) / df["mean_province_16"]
    # df["increase_mean_province_15_3"] = (df["mean_province_15"] - df["mean_province_3"]) / df["mean_province_15"]
    # df["increase_mean_Month_15_3"] = (df["mean_Month_15"] - df["mean_Month_3"]) / df["mean_Month_15"]
    # df["increase_mean_Month_16_4"] = (df["mean_Month_16"] - df["mean_Month_4"]) / df["mean_Month_16"]
    # mean = pd.DataFrame(df.groupby(["adcode", "mt"]).shift_model_adcode_mt_label_12.agg({"mean_Month": "mean", }))
    # df = pd.merge(df, mean, on=["adcode", "mt"], how="left")
    # # 几个月sum
    # df["sum_1"] = df["shift_model_adcode_mt_label_11"].values + df["shift_model_adcode_mt_label_12"].values + df[
    #     "shift_model_adcode_mt_label_1"].values + df["shift_model_adcode_mt_label_2"].values
    # df["sum_2"] = df["shift_model_adcode_mt_label_12"].values + df["shift_model_adcode_mt_label_1"].values
    # df["sum_3"] = df["shift_model_adcode_mt_label_3"].values + df["shift_model_adcode_mt_label_2"].values + df[
    #     "shift_model_adcode_mt_label_1"].values
    # stat_feat_3 = ["mean_province", "min_province", "mean_Month", "sum_1", "sum_2", "sum_3", "increase16_4",
    #                "increase_mean_province_15_3", "increase_mean_Month_15_3", "increase_mean_province_16_4",
    #                "increase_mean_Month_16_4"]  # 所有统计特征
    # stat_feat.remove("shift_model_adcode_mt_label_15")  # 删掉两个特征
    # stat_feat.remove("shift_model_adcode_mt_label_16")
    # for col in tqdm(['label']):
    # # 历史销量数据特征
    #     for i in [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
    #         stat_feat.append('shift_model_adcode_mt_{}_{}'.format(col, i))
    #         df['model_adcode_mt_{}_{}'.format(col, i)] = df[
    #                                                          'model_adcode_mt'] + i  # 新加一列值，等于车型*省*时间+i，寻求i个月前的值，将model_adcode_mt_作为索引
    #         df_last = df[~df[col].isnull()].set_index('model_adcode_mt_{}_{}'.format(col, i))
    #         df['shift_model_adcode_mt_{}_{}'.format(col, i)] = df['model_adcode_mt'].map(
    #             df_last[col])  # 后者索引是31000002开始，前者少i，取前面的匹配后面索引成功，就取值
    # for col in tqdm(['popularity']):
    #     # 历史销量数据特征
    #     for i in [1, 2, 3, 10, 11, 12]:  # popularity只取一部分
    #         stat_feat.append('shift_model_adcode_mt_{}_{}'.format(col, i))
    #         df['model_adcode_mt_{}_{}'.format(col, i)] = df[
    #                                                          'model_adcode_mt'] + i  # 新加一列值，等于车型*省*时间+i，寻求i个月前的值，将model_adcode_mt_作为索引
    #         df_last = df[~df[col].isnull()].set_index('model_adcode_mt_{}_{}'.format(col, i))
    #         df['shift_model_adcode_mt_{}_{}'.format(col, i)] = df['model_adcode_mt'].map(
    #             df_last[col])  # 后者索引是31000002开始，前者少i，取前面的匹配后面索引成功，就取值
    # df["increase16_4"] = (df["shift_model_adcode_mt_label_16"] - df["shift_model_adcode_mt_label_4"]) / df[
    #     "shift_model_adcode_mt_label_16"]  # 同比一年前的增长
    # mean = pd.DataFrame(df.groupby(["model", "mt"]).shift_model_adcode_mt_label_12.agg({"mean_province": "mean",
    #                                                                                     "min_province": "min", }))
    # df = pd.merge(df, mean, on=["model", "mt"], how="left")
    # mean = pd.DataFrame(df.groupby(["model", "mt"]).shift_model_adcode_mt_label_15.agg({"mean_province_15": "mean", }))
    # df = pd.merge(df, mean, on=["model", "mt"], how="left")
    # mean = pd.DataFrame(df.groupby(["model", "mt"]).shift_model_adcode_mt_label_3.agg({"mean_province_3": "mean", }))
    # df = pd.merge(df, mean, on=["model", "mt"], how="left")
    # mean = pd.DataFrame(df.groupby(["model", "mt"]).shift_model_adcode_mt_label_16.agg({"mean_province_16": "mean", }))
    # df = pd.merge(df, mean, on=["model", "mt"], how="left")
    # mean = pd.DataFrame(df.groupby(["model", "mt"]).shift_model_adcode_mt_label_4.agg({"mean_province_4": "mean", }))
    # df = pd.merge(df, mean, on=["model", "mt"], how="left")
    # # 另一种统计方式
    # mean = pd.DataFrame(df.groupby(["adcode", "mt"]).shift_model_adcode_mt_label_15.agg({"mean_Month_15": "mean"}))
    # df = pd.merge(df, mean, on=["adcode", "mt"], how="left")
    # mean = pd.DataFrame(df.groupby(["adcode", "mt"]).shift_model_adcode_mt_label_3.agg({"mean_Month_3": "mean"}))
    # df = pd.merge(df, mean, on=["adcode", "mt"], how="left")
    # mean = pd.DataFrame(df.groupby(["adcode", "mt"]).shift_model_adcode_mt_label_16.agg({"mean_Month_16": "mean"}))
    # df = pd.merge(df, mean, on=["adcode", "mt"], how="left")
    # mean = pd.DataFrame(df.groupby(["adcode", "mt"]).shift_model_adcode_mt_label_4.agg({"mean_Month_4": "mean"}))
    # df = pd.merge(df, mean, on=["adcode", "mt"], how="left")
    # # 基于统计特征的increase，强特
    # df["increase_mean_province_16_4"] = (df["mean_province_16"] - df["mean_province_4"]) / df["mean_province_16"]
    # df["increase_mean_province_15_3"] = (df["mean_province_15"] - df["mean_province_3"]) / df["mean_province_15"]
    # df["increase_mean_Month_15_3"] = (df["mean_Month_15"] - df["mean_Month_3"]) / df["mean_Month_15"]
    # df["increase_mean_Month_16_4"] = (df["mean_Month_16"] - df["mean_Month_4"]) / df["mean_Month_16"]
    # mean = pd.DataFrame(df.groupby(["adcode", "mt"]).shift_model_adcode_mt_label_12.agg({"mean_Month": "mean", }))
    # df = pd.merge(df, mean, on=["adcode", "mt"], how="left")
    # # 几个月sum
    # df["sum_1"] = df["shift_model_adcode_mt_label_11"].values + df["shift_model_adcode_mt_label_12"].values + df[
    #     "shift_model_adcode_mt_label_1"].values + df["shift_model_adcode_mt_label_2"].values
    # df["sum_2"] = df["shift_model_adcode_mt_label_12"].values + df["shift_model_adcode_mt_label_1"].values
    # df["sum_3"] = df["shift_model_adcode_mt_label_3"].values + df["shift_model_adcode_mt_label_2"].values + df[
    #     "shift_model_adcode_mt_label_1"].values
    # stat_feat_3 = ["mean_province", "min_province", "mean_Month", "sum_1", "sum_2", "sum_3", "increase16_4",
    #                "increase_mean_province_15_3", "increase_mean_Month_15_3", "increase_mean_province_16_4",
    #                "increase_mean_Month_16_4"]  # 所有统计特征
    # stat_feat.remove("shift_model_adcode_mt_label_15")  # 删掉两个特征
    # stat_feat.remove("shift_model_adcode_mt_label_16")

    return df,stat_feat


##评价参数
def score(data, pred='pred_label', label='label', group='model'):
    data['pred_label'] = data['pred_label'].apply(lambda x: 0 if x < 0 else x).round().astype(int)
    data_agg = data.groupby('model').agg({
        pred:  list,
        label: [list, 'mean']
    }).reset_index()
    data_agg.columns = ['_'.join(col).strip() for col in data_agg.columns]
    nrmse_score = []
    # print(data_agg)
    for raw in data_agg[['{0}_list'.format(pred), '{0}_list'.format(label), '{0}_mean'.format(label)]].values:
        nrmse_score.append(
            mse(raw[0], raw[1]) ** 0.5 / raw[2]
        )
    print(1 - np.mean(nrmse_score))
    return 1 - np.mean(nrmse_score)

##模型选择
def get_model_type(train_x,train_y,valid_x,valid_y,m_type='lgb',objective='mae'):
    if m_type == 'lgb':
        model = lgb.LGBMRegressor(
                                num_leaves=31, reg_alpha=0.25, reg_lambda=0.25, objective=objective,
                                max_depth=-1, learning_rate=0.05, min_child_samples=5, random_state=2019,
                                n_estimators=2000, subsample=0.9, colsample_bytree=0.7,
                                )
        model.fit(train_x, train_y,
              eval_set=[(train_x, train_y),(valid_x, valid_y)],
              categorical_feature=cate_feat,
              early_stopping_rounds=100, verbose=100)
        # feature_important = pd.Series(model.feature_importances_, index=train_x.columns).sort_values(
        #     ascending=False)
        # plt.bar(feature_important.index, feature_important.data)
        # plt.xticks(rotation=90)
        # plt.show()
    elif m_type == 'xgb':
        model = xgb.XGBRegressor(
                                max_depth=5 , learning_rate=0.05, n_estimators=3000,
                                objective='reg:gamma', tree_method = 'hist',subsample=0.9,
                                colsample_bytree=0.7, min_child_samples=5,eval_metric ='rmse'
                                )
        model.fit(train_x, train_y,
              eval_set=[(train_x, train_y),(valid_x, valid_y)],
              early_stopping_rounds=100, verbose=100)
    return model
##模型训练
def get_train_model(df_, m, m_type='lgb',is_log=0):
    df = df_.copy()
    # 数据集划分
    st = 13
    all_idx   = (df['mt'].between(st , m-1))
    train_idx = (df['mt'].between(st , m-5))
    valid_idx = (df['mt'].between(m-4, m-4))
    test_idx  = (df['mt'].between(m  , m  ))
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


#逐步预测
for month in [25, 26, 27, 28]:
    m_type = 'lgb'
    data_df, stat_feat= get_stat_feature(data)
    print(data_df)
    # lowfeatures=['shift_model_adcode_mt_model_4','shift_model_adcode_mt_bodyType_3','shift_model_adcode_mt_model_5','shift_model_adcode_mt_bodyType_4',
    #              'shift_model_adcode_mt_model_6','shift_model_adcode_mt_model_7','shift_model_adcode_mt_model_8','shift_model_adcode_mt_bodyType_5',
    #              'shift_model_adcode_mt_bodyType_6','shift_model_adcode_mt_bodyType_7','shift_model_adcode_mt_bodyType_8']
    # for item in lowfeatures:
    #     if item in stat_feat:
    #         stat_feat.remove(item)
    num_feat =['regYear']+stat_feat+['label_model_rank'
                                     ]
    cate_feat = ['adcode', 'bodyType', 'model', 'regMonth','province']
    if m_type == 'lgb':
        for i in cate_feat:
            data_df[i] = data_df[i].astype('category')
    elif m_type == 'xgb':
        lbl = LabelEncoder()
        for i in tqdm(cate_feat):
            data_df[i] = lbl.fit_transform(data_df[i].astype(str))

    features = num_feat + cate_feat
    sub, val_pred = get_train_model(data_df, month, m_type)
    data.loc[(data.regMonth == (month - 24)) & (data.regYear == 2018), 'salesVolume'] = sub['forecastVolum'].values
    data.loc[(data.regMonth == (month - 24)) & (data.regYear == 2018), 'label'] = sub['forecastVolum'].values
sub = data.loc[(data.regMonth >= 1) & (data.regYear == 2018), ['id', 'salesVolume']]
sub.columns = ['id', 'forecastVolum']
print(sub['forecastVolum'].mean())
sub['forecastVolum']=sub['forecastVolum'].map(lambda x:np.math.ceil(x))
sub['forecastVolum']=sub['forecastVolum'].map(lambda x:x* 0.79 if x>=500 else x)
sub[['id', 'forecastVolum']].astype(int).to_csv(path+"LGB"+'CCF_sales.csv', index=False)


