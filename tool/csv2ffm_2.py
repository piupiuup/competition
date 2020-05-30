#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Bo Song on 2018/4/26


import xlearn as xl
from kdd2019.feat.feat1 import *

data_path = 'C:/Users/cui/Desktop/python/kdd2019/data/eval/'

# 加权平均F1
def f1_score(y_true,y_pred,w_average=False,slient=True):
    y_trues = np.unique(y_true)
    score_list = []
    weight = [sum(y_true == i) for i in np.unique(y_true)] if w_average else np.ones(len(np.unique(y_true)))
    for i,label in enumerate(y_trues):
        a = sum((y_true==y_pred) & (y_true==label))
        b = sum(y_true == label)
        c = sum(y_pred == label)
        score_list.append(2*a/(b+c))
        if not slient:
            print('种类{}的真实个数：{}，    预测个数：{}，    正确个数：{}，    最后得分：{}'.format(i,b,c,a,2*a/(b+c)))
    return np.average(score_list,weights=weight)

refit = False
if not refit:
    data_path = 'C:/Users/cui/Desktop/python/kdd2019/data/eval/'
    pro = pd.read_csv(data_path + 'profiles.csv')
    test_plan = pd.read_csv(data_path + 'test_plans.csv')
    test_que = pd.read_csv(data_path + 'test_queries.csv')
    train_click = pd.read_csv(data_path + 'train_clicks.csv')
    train_plan = pd.read_csv(data_path + 'train_plans.csv')
    train_que = pd.read_csv(data_path + 'train_queries.csv')

    for que in [train_que,test_que]:
        que['start_lon'] = que['o'].apply(lambda x: float(x.split(',')[0]))
        que['start_lat'] = que['o'].apply(lambda x: float(x.split(',')[1]))
        que['end_lon'] = que['d'].apply(lambda x: float(x.split(',')[0]))
        que['end_lat'] = que['d'].apply(lambda x: float(x.split(',')[1]))
        que['pid'].fillna(-1,inplace=True)
        que.drop(['o','d'],axis=1,inplace=True)
    print('开始生成特征...')
    data_feat = make_feat(train_que,train_plan,train_click,test_que,test_plan, pro,data_key='eval_data')
else :
    print('开始读取特征...')
    data_feat = make_feat(data_key='eval_data')
train_feat = data_feat[data_feat['sid'].isin(train_que['sid'].values)].copy()
test_feat = data_feat[data_feat['sid'].isin(test_que['sid'].values)].copy()

test_click = pd.read_csv(data_path + 'test_clicks.csv')
test_click_temp = test_plan[['sid']].merge(test_click[['sid','click_mode']],on='sid', how='left')
test_click_temp['click_mode'].fillna(0,inplace=True)
test_click_temp['label'] = 1
test_feat = test_feat.drop(['label'],axis=1).merge(test_click_temp, on=['sid', 'click_mode'], how='left')
test_feat['label'].fillna(0,inplace=True)



class FFMFormat:
    def __init__(self,vector_feat,one_hot_feat,continus_feat):
        self.field_index_ = None
        self.feature_index_ = None
        self.vector_feat=vector_feat
        self.one_hot_feat=one_hot_feat
        self.continus_feat=continus_feat


    def get_params(self):
        pass

    def set_params(self, **parameters):
        pass

    def fit(self, df, y=None):
        self.field_index_ = {col: i for i, col in enumerate(df.columns)}
        self.feature_index_ = dict()
        last_idx = 0
        for col in df.columns:
            if col in self.one_hot_feat:
                print(col)
                df[col]=df[col].astype('int')
                vals = np.unique(df[col])
                for val in vals:
                    if val==-1: continue
                    name = '{}_{}'.format(col, val)
                    if name not in self.feature_index_:
                        self.feature_index_[name] = last_idx
                        last_idx += 1
            elif col in self.vector_feat:
                print(col)
                vals=[]
                for data in df[col].apply(str):
                    if data!="-1":
                        for word in data.strip().split(' '):
                            vals.append(word)
                vals = np.unique(vals)
                for val in vals:
                    if val=="-1": continue
                    name = '{}_{}'.format(col, val)
                    if name not in self.feature_index_:
                        self.feature_index_[name] = last_idx
                        last_idx += 1
            self.feature_index_[col] = last_idx
            last_idx += 1
        return self

    def fit_transform(self, df, y=None):
        self.fit(df, y)
        return self.transform(df)

    def transform_row_(self, row):
        ffm = []

        for col, val in row.loc[row != 0].to_dict().items():
            if col in self.one_hot_feat:
                name = '{}_{}'.format(col, val)
                if name in self.feature_index_:
                    ffm.append('{}:{}:1'.format(self.field_index_[col], self.feature_index_[name]))
                # ffm.append('{}:{}:{}'.format(self.field_index_[col], self.feature_index_[col], 1))
            elif col in self.vector_feat:
                for word in str(val).split(' '):
                    name = '{}_{}'.format(col, word)
                    if name in self.feature_index_:
                        ffm.append('{}:{}:1'.format(self.field_index_[col], self.feature_index_[name]))
            elif col in self.continus_feat:
                if val!=-1:
                    ffm.append('{}:{}:{}'.format(self.field_index_[col], self.feature_index_[col], val))
        return ' '.join(ffm)

    def transform(self, df):
        # val=[]
        # for k,v in self.feature_index_.items():
        #     val.append(v)
        # val.sort()
        # print(val)
        # print(self.field_index_)
        # print(self.feature_index_)
        return pd.Series({idx: self.transform_row_(row) for idx, row in df.iterrows()})

    def dump_ffm(self, data, y, data_path):
        ffm_out = open(data_path, 'w')
        for (i, line) in enumerate(data.values):
            ffm_out.write(str(y[i]) + ' ' + line)
        ffm_out.close()

print('生成ffm格式的数据...')

vector_feature = [ 'click_mode','c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10']
one_hot_feature = []
continus_feature = ['distance', 'eta', 'price', 'plan_rk', 'sid_plan_cnt', 'sid_count', 'dayofweek',
         'dayofyear', 'hour', 'start_lon', 'start_lat', 'end_lon', 'end_lat', 'speed', 'price_eta', 'distance_price']
tr = FFMFormat(vector_feature,one_hot_feature,continus_feature)
data_ffm = tr.fit(data_feat)
train_ffm = tr.transform(train_feat)
test_ffm = tr.transform(test_feat)
tr.dump_ffm(train_ffm,train_feat['label'].values,data_path=data_path+'train_ffm.csv')
tr.dump_ffm(test_ffm,test_feat['label'].values,data_path=data_path+'test_ffm.csv')

print('读取ffm格式的数据...')
ffm_model = xl.create_ffm()
ffm_model.setTrain(data_path+'train_ffm1.csv')
# ffm_model.setTest(data_path+'test_ffm.csv')
ffm_model.setValidate(data_path+'test_ffm1.csv')
print('开始ffm训练...')
ffm_model.setQuiet()
param = {'task':'reg', 'lr':0.2, 'lambda':0.002,'epoch':1}
ffm_model.cv(param)
ffm_model.fit(param, "./model.out")

