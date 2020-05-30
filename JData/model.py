# -*-coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
import matplotlib.pylab as plt
from sklearn.metrics import roc_auc_score
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
import random
from sklearn.cross_validation import KFold

#模型
def predict(train, test, alg = XGBClassifier( learning_rate=0.05, max_delta_step=0, max_depth=5,min_child_weight=4,
                          n_estimators=300,scale_pos_weight=21, nthread=-1, subsample=0.6)):

    train = train.fillna(-1)
    test = test.fillna(-1)
    target = 'label'
    #IDcol = ['user_id','sku_id']
    predictors = [x for x in train.columns if x not in ['user_id','sku_id','label']]

    #alg1 = GradientBoostingRegressor(loss='ls')
    alg.fit(train[predictors], train[target])
    result = alg.predict_proba(test[predictors])
    if 'sku_id' in train.columns:
        result = pd.DataFrame({ 'user_id':test['user_id'].values, 'sku_id':test['sku_id'].values, 'label':result[:, 1]})
    else:
        result = pd.DataFrame(
            {'user_id': test['user_id'].values, 'label': result[:, 1]})
    result.sort_values('label',ascending=False,inplace=True)

    return result


#F11公式
def grade_F11(y_true,y_pred,silent=0):
    label_pred = y_pred['user_id'].copy().drop_duplicates()
    label_true = y_true['user_id'].copy().drop_duplicates()

    a = float(sum(label_pred.isin(label_true.values)))  # 正确的个数
    m = len(label_pred)                                 # 预测的个数
    n = len(label_true)                                 # 实际的个数
    score = 6 * a / (5 * m + n)
    if silent != 1:
        print ('label正确的个数为：' + str(a))
        print ('label实际的个数为：' + str(n) + '，  label召回率为：' + str(a / n))
        print ('label预测的个数为：' + str(m) + '，  label准确率为：' + str(a / (1*m)))
        print ('F11得分：' + str(score))
    return score


# F12公式
def grade_F12(y_true, y_pred, silent=0):

    a = (y_pred[['user_id','sku_id']].merge(y_true[['user_id', 'sku_id']], on=['user_id', 'sku_id'])).shape[0]   # 正确的个数
    m = len(y_pred)                                  # 预测的个数
    n = len(y_true)                                  # 实际的个数
    score = 5 * a / (2 * m + 3 * n)
    if silent != 1:
        print ('pred正确的个数为：' + str(a))
        print ('pred实际的个数为：' + str(n) + '，  pred召回率为：' + str(a / n))
        print ('pred预测的个数为：' + str(m) + '，  pred准确率为：' + str(a / (1*m)))
        print ('predF12得分：' + str(score))
    return score


#测评函数
def grade(y_true,y_pred,silent=0):
    y_pred = y_pred[['user_id','sku_id']].drop_duplicates()
    y_true = y_true[['user_id', 'sku_id']].drop_duplicates()


    #计算F11
    F11 = grade_F11(y_true, y_pred, silent)
    F12 = grade_F12(y_true, y_pred, silent)
    score = 0.4*F11 + 0.6*F12
    if silent!=1:
        print ('最终得分：' + str(score))
    return score


#提交去重
def select(data,n=600):
    data.sort_values('label', ascending=False, inplace=True)
    result = data.drop_duplicates('user_id')
    result = result.head(n)[['user_id','sku_id']]
    result = result.astype(int)
    print ('筛选出'+str(result.shape[0])+'个ID')
    return result


#auc评分
def grade_auc( y_true, y_pred):
    return roc_auc_score(y_true, y_pred)


#测试
def check(train,test,y_true,n=600,alg = XGBClassifier( learning_rate=0.05, max_delta_step=0, max_depth=5,min_child_weight=4,
                          n_estimators=300, scale_pos_weight=21, nthread=-1, subsample=0.8),predict=predict):
    y_pred = predict(train,test,alg)
    auc_score = roc_auc_score(test['label'], y_pred['label'])
    print('AUC得分：   ' + str(auc_score) + '\t')
    y_pred = select(y_pred, n)
    grade(y_true, y_pred, 0)

def roc_auc_score_head(y_true, y_pred, n=10000):
    data = pd.DataFrame({'true':y_true.values,'pred':y_pred.values})
    data = data.sort_values('pred',ascending=False).head(n)
    return roc_auc_score(data['true'].values, data['pred'].values)


#用户测评
def user_check(train,test,y_true,n=600,alg = XGBClassifier( learning_rate =0.06,
                        n_estimators=140, max_depth=4, min_child_weight=1, gamma=0,
                        subsample=0.9, colsample_bytree=0.8, scale_pos_weight=21,
                                                           seed=27),predict=predict):
    y_pred = predict(train,test,alg)
    auc_score = roc_auc_score(test['label'], y_pred['label'])
    auc_score10000 = roc_auc_score_head(test['label'], y_pred['label'])
    print('用户AUC得分：   ' + str(auc_score) + '\t')
    print('前10000用户AUC得分：   ' + str(auc_score10000) + '\t')

    grade_F11(y_true, y_pred.sort_values('label',ascending=False).head(n))

#商品测评
def product_check(data,  alg = XGBClassifier( learning_rate =0.06,
                        n_estimators=110, max_depth=3, min_child_weight=1, gamma=0,
                        subsample=1, colsample_bytree=0.8, scale_pos_weight=5,
                                                           seed=27),predict=predict):
    kf = KFold(len(data), n_folds=5, shuffle=True, random_state=520)
    result = None
    for i, (train_index, test_index) in enumerate(kf):
        train = data.iloc[train_index]
        test = data.iloc[test_index]
        y_pred = predict(train,test,alg)
        y_pred['y_true'] = test['label'].values
        if result is None:
            result = y_pred
        else:
            result = pd.concat([result,y_pred])
    auc_score = roc_auc_score(result['y_true'], result['label'])
    print('商品AUC得分：   ' + str(auc_score) + '\t')
    y_true = train[train['label']==1]
    y_pred = result.sort_values('label',ascending=True).drop_duplicates('user_id',keep='last')
    y_pred = pd.merge(y_pred[['user_id','sku_id']],y_true,on=['user_id','sku_id'],how='left')
    accuracy = y_pred['label'].sum()/y_pred.shape[0]
    print('商品准确率：   ' + str(accuracy) + '\t')


#特征重要性
def get_feat_imp(train,ID=['user_id','sku_id'],target='label'):

    ID_and_target = ID.copy()
    ID_and_target.append(target)
    predictors = [x for x in train.columns if x not in ID_and_target]
    model = XGBClassifier( learning_rate =0.06,n_estimators=140, max_depth=4,
                           min_child_weight=1, gamma=0,subsample=0.9, colsample_bytree=0.8,
                           scale_pos_weight=1, seed=27)
    model.fit(train[predictors],train[target])
    feat_imp = pd.Series(model.feature_importances_,index=predictors).sort_values(ascending=False)
    feat_imp.head(20).plot(kind='bar', title='Feature Importances',figsize=(15,5))
    plt.ylabel('Feature Importance Score')
    plt.xticks(rotation=45)

    return feat_imp

# 输入线上的分计算正确个数
def get_n_right(F11_score,F12_score,n):
    m = 1223
    r_F11 = round(F11_score * (5 * n + m) / 6)
    r_F12 = round(F12_score * (2 * n + 3 * m) / 5)
    print('A榜个数：%d' % m)
    print('F11正确个数：%d' % r_F11)
    print('F12正确个数：%d' % r_F12)

# 重新筛选样本
def get_final_user_id():
    dump_path = r'F:\cache\final_user_id.pkl'
    if os.path.exists(dump_path):
        user_list = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_cate8('2016-01-31', '2016-04-16')
        actions['date'] = actions['time'].apply(lambda x:x[:10])
        # 用户第一次浏览的日期
        user_first_date = actions.drop_duplicates('user_id')
        user_first_date.rename(columns={'date':'user_first_date'},inplace=True)
        # 用户第一次购买的日期
        user_buy_date = actions[actions['type']==4].drop_duplicates('user_id')
        user_buy_date.rename(columns={'date': 'user_buy_date'}, inplace=True)
        user_buy_date['label'] = 1
        user_first_date = pd.merge(user_first_date,user_buy_date[['user_id','label']],on='user_id',how='left')
        user_first_date['label'].fillna(0,inplace=True)
        n_user_buy = user_buy_date.groupby('user_buy_date',as_index=False)['user_id'].count()
        n_user_buy.columns = [['user_buy_date','n_user_buy']]
        n_user_buy['n_user_look_exception'] = n_user_buy['n_user_buy']**0.5*10
        n_user_look_exception = dict(zip(n_user_buy['user_buy_date'],n_user_buy['n_user_look_exception']))
        user_list = []
        for date in n_user_buy['user_buy_date'].values:
            users = user_first_date[user_first_date['user_first_date']==date]
            users_buy = users[users['label']==1]
            users_not_buy = users[users['label']!=1]
            n_positive = int(n_user_look_exception[date]) - users_buy.shape[0]
            user_buy_sub = list(users_buy['user_id'].values)
            user_list.extend(user_buy_sub)

            user_not_buy_sub = list(users_not_buy['user_id'].values)
            random.shuffle(user_not_buy_sub)
            user_list.extend(user_not_buy_sub[:n_positive])

        pickle.dump(user_list, open(dump_path, 'wb+'))
    return user_list

# 获取购买过的用户，选随机选择1000人
def get_output(x_output, n_add=1000,n=700):
    data = x_output.copy()
    if 'sku_id' not in data:
        data['sku_id'] = data['user_id'].values[300]
    if data.shape[0] > 5000:
        data = data.head(700)
    dump_path = r'F:\cache\labels_2016-01-31_2016-04-16.pkl'
    if os.path.exists(dump_path):
        actions = pickle.load(open(dump_path, 'rb+'))
    else:
        actions = get_cate8('2016-01-31', '2016-04-16')
        actions = actions[actions['type'] == 4]
        actions = actions.drop_duplicates('user_id')
        actions = actions[['user_id','sku_id']]
        pickle.dump(actions, open(dump_path, 'wb+'))

    addition = list(np.random.permutation(actions.shape[0]))
    addition = actions.iloc[addition[:n_add]]
    y_output = pd.concat([addition,data])[['user_id','sku_id']]

    return y_output

# 计算提交的个数
def truth(F11, F12):
    for i in range(100,2000):
        x = ((1223+i*5)*F11)/6.
        z = ((1223*3+i*2)*F12)/5.
        if (abs(abs(x*100%100-50) - 50) < 1) and (abs(abs(z*100%100-50) - 50) < 1) :
            print ("Submission: " + str(i) + '\n' + \
                "True user: " + str(int(round(x))) + '\t\t' + \
                "True sku: " + str(int(round(z))))

# 获取对应的商品id
def get_sku_id(data):
    user = data.copy()
    user_sku = pd.read_csv(r'C:\Users\csw\Documents\Tencent Files\278044960\FileRecv\detail_2016_05_04_2.csv')
    user_sku['user_id'] = user_sku['user_id'].astype(np.int)
    user_sku.sort_values('label',ascending=False,inplace=True)
    user_sku.drop_duplicates('user_id',keep='first',inplace=True)
    user = pd.merge(user[['user_id']],user_sku[['user_id','sku_id']],on='user_id',how='left')

    return user

# 输入个数，计算得分
def get_score(n_f11, n_f12, n, m):
    f11 = 6 * n_f11 / (5 * n + m)
    f12 = 5 * n_f12 / (2 * n + 3 * m)
    score = 0.4*f11 + 0.6*f12
    print('F11得分为：  ' + str(f11))
    print('F12得分为：  ' + str(f12))
    print('得分为：  ' + str(score))
    return None

# 输入f11，f12得分，计算每多对一个的收益
def truth(F11, F12):
    m = 1224
    for i in range(100, 2000):
        x = ((m + i * 5) * F11) / 6.
        z = ((m * 3 + i * 2) * F12) / 5.
        if (abs(abs(x * 100 % 100 - 50) - 50) < 1) and (abs(abs(z * 100 % 100 - 50) - 50) < 1):
            print("Submission： " + str(i))
            print("True user： " + str(int(round(x))),end='  ')
            print("True sku： " + str(int(round(z))))
            score = get_score(x, z, i, m)
            score_gradient_negative = get_score(x, z, i + 1, m) - score
            score_gradient_F11 = get_score(x + 1, z, i + 1, m) - score
            score_gradient_F12 = get_score(x + 1, z + 1, i + 1, m) - score
            print('多一个错误收益为：     ' + str(score_gradient_negative))
            print('多对一个F11收益为：    ' + str(score_gradient_F11))
            print('多对一个F12收益为：    ' + str(score_gradient_F12))
            print('F12与F11的收益比为：   ' + str(score_gradient_F12/score_gradient_F11))

# 计算b榜人数
def get_n_a(F11,F12,F13,F14,F15,F16):
    for m in range(100, 2000):
        flag1 = False
        flag2 = False
        flag3 = False
        for i in range(500, 1000):
            x = ((m + i * 5) * F11) / 6.
            z = ((m * 3 + i * 2) * F12) / 5.
            F11_pred = 6 * int(round(x)) / (5 * i + m)
            F12_pred = 5 * int(round(z)) / (2 * i + 3 * m)
            if (abs((F11_pred-F11)*100000) < 0.5) and (abs((F12_pred-F12)*100000)  < 0.5):
                flag1 = True
        for i in range(500, 1000):
            x = ((m + i * 5) * F13) / 6.
            z = ((m * 3 + i * 2) * F14) / 5.
            F11_pred = 6 * int(round(x)) / (5 * i + m)
            F12_pred = 5 * int(round(z)) / (2 * i + 3 * m)
            if (abs((F11_pred-F13)*100000)  < 0.5) and (abs((F12_pred-F14)*100000)  < 0.5):
                flag2 = True
        for i in range(500, 1000):
            x = ((m + i * 5) * F15) / 6.
            z = ((m * 3 + i * 2) * F16) / 5.
            F11_pred = 6 * int(round(x)) / (5 * i + m)
            F12_pred = 5 * int(round(z)) / (2 * i + 3 * m)
            if (abs((F11_pred-F15)*100000)  < 0.5) and (abs((F12_pred-F16)*100000)  < 0.5):
                flag3 = True

        if (flag1 & flag2 & flag3):
            print('榜上总数：' + str(m))

# 获取sku_id
def get_sku(user):
    result = user.copy()
    ui = pd.read_csv(r'C:\Users\csw\Desktop\ui.csv')
    ui.sort_values('label',ascending=False,inplace=True)
    ui.drop_duplicates('user_id',keep='first',inplace=True)
    result = pd.merge(result,ui[['user_id','sku_id']],on='user_id',how='left')
    return result
