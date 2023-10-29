import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def shift_n_month_features(data: pd.DataFrame, shift_n_month_list: list,
                           o_features: list):
    data['month_index'] = 12 * \
        (data.regYear - data.regYear.min()) + data.regMonth
    for shift_n_month in shift_n_month_list:
        data['shift_n_month'] = data['month_index'] - shift_n_month
        tmp = data.merge(data,
                         left_on=['shift_n_month', 'adcode', 'model'],
                         right_on=['month_index', 'adcode', 'model'],
                         how='left',
                         suffixes=('',
                                   '_shift_{}_month'.format(shift_n_month)))
        for feature in o_features:
            data[feature + '_shift_{}_month'.format(shift_n_month)] = tmp[
                feature + '_shift_{}_month'.format(shift_n_month)]
    data.drop(['month_index', 'shift_n_month'], axis=1, inplace=True)
    del tmp
    return data


# 环比
def ring_features(data, n_month):
    _data = data.copy()
    _data['ring_ratio_year'] = _data.regYear
    _data['ring_ratio_month'] = _data.regMonth - n_month
    _data.loc[_data.ring_ratio_month <= 0, 'ring_ratio_year'] -= 1
    _data.loc[_data.ring_ratio_month <= 0, 'ring_ratio_month'] = 13 - n_month

    tmp = _data.merge(
        _data,
        left_on=['ring_ratio_year', 'ring_ratio_month', 'adcode', 'model'],
        right_on=['regYear', 'regMonth', 'adcode', 'model'],
        how='left',
        suffixes=('', '_ring_ratio_feature'))
    data['ring_ratio_{}_salesVolum_increment'.format(n_month)] = (
        tmp.label -
        tmp.label_ring_ratio_feature) / tmp.label_ring_ratio_feature
    data['ring_ratio_{}_salesVolum'.format(
        n_month)] = (tmp.label) / tmp.label_ring_ratio_feature
    data['ring_diff_{}_salesVolum'.format(
        n_month)] = tmp.label - tmp.label_ring_ratio_feature

    data['ring_ratio_{}_popularity_increment'.format(n_month)] = (
        tmp.popularity -
        tmp.popularity_ring_ratio_feature) / tmp.popularity_ring_ratio_feature
    data['ring_ratio_{}_popularity'.format(n_month)] = (tmp.popularity) / \
        tmp.popularity_ring_ratio_feature
    data['ring_diff_{}_popularity'.format(
        n_month)] = tmp.popularity - tmp.popularity_ring_ratio_feature

    data['ring_ratio_{}_carCommentVolum_increment'.format(n_month)] = (
        tmp.carCommentVolum - tmp.carCommentVolum_ring_ratio_feature
    ) / tmp.carCommentVolum_ring_ratio_feature
    data['ring_ratio_{}_carCommentVolum'.format(n_month)] = (
        tmp.carCommentVolum) / tmp.carCommentVolum_ring_ratio_feature
    data['ring_diff_{}_carCommentVolum'.format(
        n_month
    )] = tmp.carCommentVolum - tmp.carCommentVolum_ring_ratio_feature

    data['ring_ratio_{}_newsReplyVolum_increment'.format(
        n_month)] = (tmp.newsReplyVolum - tmp.newsReplyVolum_ring_ratio_feature
                     ) / tmp.newsReplyVolum_ring_ratio_feature
    data['ring_ratio_{}_newsReplyVolum'.format(n_month)] = (
        tmp.newsReplyVolum) / tmp.newsReplyVolum_ring_ratio_feature
    data['ring_diff_{}_newsReplyVolum'.format(
        n_month)] = tmp.newsReplyVolum - tmp.newsReplyVolum_ring_ratio_feature
    del tmp

    return data


def whole_describe(data, features):
    data['month_index'] = 12 * \
        (data.regYear - data.regYear.min()) + data.regMonth
    for feature in features:
        data[feature + '_whole_mean'] = 0
        data[feature + '_whole_max'] = 0
        data[feature + '_whole_min'] = 0
        data[feature + '_whole_median'] = 0
        for mi in data['month_index'].unique():
            data.loc[data.month_index == mi, feature +
                     '_whole_mean'] = pd.merge(
                         left=data.loc[data.month_index ==
                                       mi, ['model', 'adcode']],
                         right=data[data.month_index < mi].groupby(
                             by=['model',
                                 'adcode'], as_index=False)[feature].mean(),
                         how='left',
                         on=['model', 'adcode'])[feature].values
            data.loc[data.month_index == mi, feature +
                     '_whole_max'] = pd.merge(
                         left=data.loc[data.month_index ==
                                       mi, ['model', 'adcode']],
                         right=data[data.month_index < mi].groupby(
                             by=['model',
                                 'adcode'], as_index=False)[feature].max(),
                         how='left',
                         on=['model', 'adcode'])[feature].values
            data.loc[data.month_index == mi, feature +
                     '_whole_min'] = pd.merge(
                         left=data.loc[data.month_index ==
                                       mi, ['model', 'adcode']],
                         right=data[data.month_index < mi].groupby(
                             by=['model',
                                 'adcode'], as_index=False)[feature].min(),
                         how='left',
                         on=['model', 'adcode'])[feature].values
            data.loc[data.month_index == mi, feature +
                     '_whole_median'] = pd.merge(
                         left=data.loc[data.month_index ==
                                       mi, ['model', 'adcode']],
                         right=data[data.month_index < mi].groupby(
                             by=['model',
                                 'adcode'], as_index=False)[feature].median(),
                         how='left',
                         on=['model', 'adcode'])[feature].values

    data.drop('month_index', axis=1, inplace=True)
    return data


def score(data, pred='pred_label', label='label', group='model'):
    data['pred_label'] = data['pred_label'].apply(
        lambda x: 0 if x < 0 else x).round().astype(int)
    data_agg = data.groupby('model').agg({
        pred: list,
        label: [list, 'mean'],
    }).reset_index()

    data_agg.columns = ['_'.join(col).strip() for col in data_agg.columns]
    nrmse_score = []
    for raw in data_agg[[
            '{0}_list'.format(pred), '{0}_list'.format(label),
            '{0}_mean'.format(label)
    ]].values:
        nrmse_score.append(mean_squared_error(raw[0], raw[1])**0.5 / raw[2])
    return 1 - np.mean(nrmse_score)


def my_rmse(estimator, X, y):
    return np.sqrt(mean_squared_error(y, estimator.predict(X)))


def get_model(model, X, y, cate_features):
    model.fit(X,
              y,
              eval_set=[(X, y)],
              eval_names=['train'],
              eval_metric='mae',
              categorical_feature=cate_features,
              verbose=False)
    return model


def make_features(data, base_features):
    _data = data.copy()
    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        # 全国车型销量比
        _data['model_dvd_global_shift_{}'.format(i)] = _data.groupby(['regYear', 'regMonth', 'model'])[
            'label_shift_{}_month'.format(
                i)].transform('sum') / \
            _data.groupby(['regYear', 'regMonth'])[
            'label_shift_{}_month'.format(i)].transform('sum')
        # 全国车身类型销量比
        _data['bodyType_dvd_global_shift_{}'.format(i)] = _data.groupby(['regYear', 'regMonth', 'bodyType'])[
            'label_shift_{}_month'.format(
                i)].transform('sum') / \
            _data.groupby(['regYear', 'regMonth'])[
            'label_shift_{}_month'.format(i)].transform('sum')
        # 全国同车身类型销量占比
        _data['model_dvd_bodyType_shift_{}'.format(i)] = _data.groupby(['regYear', 'regMonth', 'model'])[
            'label_shift_{}_month'.format(
                i)].transform('sum') / \
            _data.groupby(['regYear', 'regMonth', 'bodyType'])[
            'label_shift_{}_month'.format(i)].transform('sum')
        # 各省销量比
        _data['adcode_dvd_global_shift_{}'.format(i)] = _data.groupby(['regYear', 'regMonth', 'adcode'])[
            'label_shift_{}_month'.format(
                i)].transform('sum') / \
            _data.groupby(['regYear', 'regMonth'])[
            'label_shift_{}_month'.format(i)].transform('sum')
        # 省内车型销量比
        _data['adcode&model_dvd_adcode_shift_{}'.format(i)] = \
            _data.groupby(['regYear', 'regMonth', 'adcode', 'model'])['label_shift_{}_month'.format(
                i)].transform('sum') / _data.groupby(['regYear', 'regMonth', 'adcode'])[
                'label_shift_{}_month'.format(i)].transform(
                'sum')
        # 省内车身类型销量比
        _data['adcode&bodyType_dvd_adcode_shift_{}'.format(i)] = \
            _data.groupby(['regYear', 'regMonth', 'adcode', 'bodyType'])['label_shift_{}_month'.format(
                i)].transform('sum') / _data.groupby(['regYear', 'regMonth', 'adcode'])[
                'label_shift_{}_month'.format(i)].transform(
                'sum')
        # 省内同车身类型销量比
        _data['adcode&model_dvd_adcode&bodyType_shift_{}'.format(i)] = \
            _data.groupby(['regYear', 'regMonth', 'adcode', 'model'])['label_shift_{}_month'.format(
                i)].transform('sum') / _data.groupby(['regYear', 'regMonth', 'adcode', 'bodyType'])[
                'label_shift_{}_month'.format(i)].transform(
                'sum')
        # 不明意义。。
        _data['adcode&model_dvd_bodyType_shift_{}'.format(i)] = \
            _data.groupby(['regYear', 'regMonth', 'adcode', 'model'])['label_shift_{}_month'.format(
                i)].transform('sum') / _data.groupby(['regYear', 'regMonth', 'bodyType'])[
                'label_shift_{}_month'.format(i)].transform(
                'sum')


        _data['adcode_dvd_bodyType_shift_{}'.format(i)] = _data.groupby(['regYear', 'regMonth', 'adcode'])[
            'label_shift_{}_month'.format(
                i)].transform('sum') / \
            _data.groupby(['regYear', 'regMonth', 'bodyType'])[
            'label_shift_{}_month'.format(i)].transform('sum')

        _data['adcode&model_dvd_global_shift_{}'.format(i)] = \
            _data.groupby(['regYear', 'regMonth', 'adcode', 'model'])['label_shift_{}_month'.format(
                i)].transform('sum') / _data.groupby(['regYear', 'regMonth'])[
                'label_shift_{}_month'.format(i)].transform(
                'sum')

        for f in ['label', 'popularity']:
            # 环比
            _data['ring_ratio_{}_shift_{}'.format(
                f, i)] = _data['label_shift_{}_month'.format(i)] / _data[
                    'label_shift_{}_month'.format(i + 1)]
            # 环差
            _data['ring_diff_{}_shift_{}'.format(
                f, i)] = _data['label_shift_{}_month'.format(i)] - _data[
                    'label_shift_{}_month'.format(i + 1)]
            # 环增幅比
            _data['ring_ratio_{}_increment_shift_{}'.format(
                f, i)] = (_data['label_shift_{}_month'.format(i)] -
                          _data['label_shift_{}_month'.format(i + 1)]
                          ) / _data['label_shift_{}_month'.format(i + 1)]

    # 12月交叉特征 column mean
    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        for c in ['label']:
            _data['{}_mean_in_adcode&bodyType_shift_{}'.format(
                c, i)] = _data.groupby(
                    by=['regYear', 'regMonth', 'adcode', 'bodyType'])[
                        '{}_shift_{}_month'.format(c, i)].transform('mean')
            for g in ['adcode', 'model', 'bodyType']:
                _data['{}_mean_in_{}'.format(c, g)] = _data.groupby(
                    by=['regYear', 'regMonth', g])['{}_shift_{}_month'.format(
                        c, i)].transform('mean')

    _data.drop(['regYear'] +
               [f for f in _data.columns if f.__contains__('shift_13_month')],
               axis=1,
               inplace=True)
    return _data


if __name__ == '__main__':
    data = pd.read_csv('../data/data.csv')
    data.drop('province', axis=1, inplace=True)
    # 年月编码
    day_map = {
        1: 31,
        2: 28,
        3: 31,
        4: 30,
        5: 31,
        6: 30,
        7: 31,
        8: 31,
        9: 30,
        10: 31,
        11: 30,
        12: 31
    }
    data['dayCount'] = data['regMonth'].map(day_map)
    data.loc[(data.regMonth == 2) & (data.regYear == 2016), 'dayCount'] = 29

    data['isSf'] = 0
    data.loc[(data.regMonth == 2) & (data.regYear == 2016), 'isSf'] = 1
    data.loc[(data.regMonth == 1) & (data.regYear == 2017), 'isSf'] = 1
    data.loc[(data.regMonth == 2) & (data.regYear == 2018), 'isSf'] = 1

    data['month_x'] = np.sin(data.regMonth * np.pi / 6)
    data['month_y'] = np.cos(data.regMonth * np.pi / 6)
    time_features = [
        'regYear', 'regMonth', 'month_x', 'month_y', 'dayCount', 'isSf'
    ]
    # 类别特征
    cate_features = ['adcode', 'bodyType', 'model']
    data[cate_features] = data[cate_features].astype('category')

    # 类别特征，按年份 target mean encoding
    for cate_feature in cate_features:
        data[cate_feature + '_mean_encoding'] = data[cate_feature].map(
            data.groupby(by=cate_feature)['label'].mean()).astype('float64')

    mean_encoding_feature = [
        f for f in data.columns if f.__contains__('_mean_encoding')
    ]

    # model、 adcode 整体描述
    data = whole_describe(data, ['label', 'popularity'])

    shift_features = [
        'label',
        'popularity',
    ]
    data = shift_n_month_features(data,
                                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                                  shift_features)

    # 类别特征+类别mean encoding特征+shift特征
    features = cate_features + time_features + mean_encoding_feature + \
        [f for f in data.columns if f.__contains__('shift')] + \
        [f for f in data.columns if f.__contains__('whole')]

    train_idx = (data.regYear == 2017) & (data.regMonth.between(1, 8))
    valid_idx = (data.regYear == 2017) & (data.regMonth.between(9, 12))
    test_idx = data.regYear == 2018

    train_x = data[train_idx][features]
    train_y = data[train_idx]['label']

    valid_x = data[valid_idx][features]
    valid_y = data[valid_idx]['label']

    test_x = data[test_idx][features]

    reg = LGBMRegressor(
        n_estimators=9999,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=128,
        subsample=0.8,
        subsample_freq=3,
        colsample_bytree=0.8,
        reg_alpha=1,
        reg_lambda=1,
        min_child_samples=5,
        random_state=2019,
        objective='mse',
        silent=True,
    )

    reg.fit(make_features(train_x, base_features=shift_features),
            train_y,
            eval_set=[(make_features(train_x,
                                     base_features=shift_features), train_y),
                      (make_features(valid_x,
                                     base_features=shift_features), valid_y)],
            eval_names=['train', 'val'],
            eval_metric='mae',
            early_stopping_rounds=25,
            categorical_feature=cate_features,
            verbose=20)
    reg.n_estimators = reg.best_iteration_ + 25

    tmp_x = train_x
    tmp_y = train_y
    for regMonth in tqdm(valid_x.regMonth.unique()):
        for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            if regMonth - i >= valid_x.regMonth.min():
                valid_x.loc[valid_x.regMonth == regMonth, [
                    f for f in features
                    if f.__contains__('_shift_{}_month'.format(i))
                ]] = None

                valid_x.loc[valid_x.regMonth == regMonth,
                            'label_shift_{}_month'.format(i)] = reg.predict(
                                make_features(
                                    valid_x.loc[valid_x.regMonth == regMonth -
                                                i, :],
                                    base_features=shift_features))

        # if regMonth <= valid_x.regMonth.min(): continue
        tmp_x = tmp_x.append(
            valid_x.loc[valid_x.regMonth == regMonth, features],
            ignore_index=True)
        tmp_y = tmp_y.append(pd.Series(
            reg.predict(
                make_features(
                    valid_x.loc[valid_x.regMonth == regMonth, features],
                    base_features=shift_features))),
                             ignore_index=True)

        reg = get_model(reg, make_features(tmp_x,
                                           base_features=shift_features),
                        tmp_y, cate_features)

    y_valid = reg.predict(make_features(valid_x, base_features=shift_features))
    data['pred_label'] = 0
    data.loc[valid_idx, 'pred_label'] = y_valid
    print('valid data score: ' + str(score(data.loc[valid_idx, :])))

    reg.fit(make_features(data[data.regYear == 2017][features],
                          base_features=shift_features),
            data[data.regYear == 2017]['label'],
            eval_set=[(make_features(data[data.regYear == 2017][features],
                                     base_features=shift_features),
                       data[data.regYear == 2017]['label'])],
            eval_names=['train'],
            eval_metric='mae',
            categorical_feature=cate_features,
            verbose=20)

    tmp_x = data[data.regYear == 2017][features]
    tmp_y = data[data.regYear == 2017]['label']
    for regMonth in tqdm(test_x.regMonth.unique()):
        for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            if regMonth - i >= test_x.regMonth.min():
                test_x.loc[test_x.regMonth == regMonth, 'label_shift_{}_month'.
                           format(i)] = reg.predict(
                               make_features(
                                   test_x.loc[test_x.regMonth == regMonth -
                                              i, :],
                                   base_features=shift_features))

        # if regMonth <= test_x.regMonth.min(): continue
        tmp_x = tmp_x.append(test_x.loc[test_x.regMonth == regMonth, features],
                             ignore_index=True)
        tmp_y = tmp_y.append(pd.Series(
            reg.predict(
                make_features(
                    test_x.loc[test_x.regMonth == regMonth, features],
                    base_features=shift_features))),
                             ignore_index=True)

        reg = get_model(reg, make_features(tmp_x,
                                           base_features=shift_features),
                        tmp_y, cate_features)

    y_pred = reg.predict(make_features(test_x, base_features=shift_features))

    # submit = pd.read_csv('../data/evaluation_public.csv')
    # submit['forecastVolum'] = y_pred
    # submit['forecastVolum'][submit['forecastVolum'] < 0] = 0
    # from datetime import datetime
    # time_str = datetime.now().strftime(format='%Y-%m-%d %H-%M-%S ')
    # submit[['id', 'forecastVolum']].round().astype(int).to_csv(
    #     '../submit/submit_{0}.csv'.format(time_str), encoding='utf8', index=False)
