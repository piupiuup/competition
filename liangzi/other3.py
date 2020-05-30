import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from datetime import datetime
from sklearn.metrics import roc_auc_score
import re

import warnings
import gc
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from utils import *


path = 'F:\\Python\\Game_data\\CCF\\rrisk\\concat_data\\'
name =['train','test','1entbase','2alter','3branch','4invest','5right','6project',
       '7lawsuit','8breakfaith','9recruit','10qualification']


def get_num(x):
    try:
        num = re.search('\d+', x).group()
        return float(num)
    except:
        return 0


def HY_ETYPE_order(df):
    for feat in ['HY', 'ETYPE_copy']:
        #     order = df[df.TARGET!=-1].groupby(feat).mean()['TARGET'].sort_values().index
        #     order_dict = {j:i+1 for i, j in enumerate(order)}
        order_dict = df[df.TARGET.notnull()].groupby(feat).mean()['TARGET'].sort_values().to_dict()
        df['{}_order'.format(feat)] = df[feat].map(order_dict)
    return df


def rmb(x):
    if (str(x).startswith('人民币')):
        return 1

    elif (str(x).startswith('美元')):
        return 2

    elif (str(x).startswith('港币')):
        return 0

    else:
        return 1


def handle_alter_num(x, hk_rate=0.845, us_rate=6.6):
    if isinstance(x, str):
        if '港币' in x:
            return hk_rate * get_num(x)
        elif '美元' in x:
            return us_rate * get_num(x)
        else:
            return get_num(x)
    else:
        return x


def handle_alter_num_c(x, y, hk_rate=0.845, us_rate=6.6):
    if isinstance(y, int):
        if y == 0:
            return hk_rate * x
        elif y == 2:
            return us_rate * x
        else:
            return x
    else:
        return x

# (218264, 3)
tr = pd.read_csv(path+name[0]+'.csv', date_parser='ENDDATE')
test = pd.read_csv(path+name[1]+'.csv')

# (436798, 12)
# ZCZB 5null  all in test
# HY  8null  4 in test, 4 in tr
En = pd.read_csv(path+name[2]+'.csv')
En[['MPNUM','INUM','ENUM', 'FINZB', 'FSTINUM', 'TZINUM','IENUM']] = En[['MPNUM','INUM','ENUM', 'FINZB', 'FSTINUM', 'TZINUM','IENUM']].fillna(0)
En['ZCZB'] = En['ZCZB'].fillna(0)


# (398238, 5)
ch = pd.read_csv(path+name[3]+'.csv')
ch.ix[37600,'ALTAF'] = 300.9171*6.6
ch.ix[61332,'ALTBE'] = 100*0.845
ch.ix[61333,'ALTBE'] = 100*0.845
ch.ix[[37600,61332,61333],'f1'] = 1
ch["ALTBE"]= list(map(lambda x,y: handle_alter_num_c(x,y),ch['ALTBE'],ch['f1']))
ch["ALTAF"]= list(map(lambda x,y: handle_alter_num_c(x,y),ch['ALTAF'],ch['f1']))


# (164150, 5)
bra = pd.read_csv(path+name[4]+'.csv')
# (84189, 6)
tz = pd.read_csv(path+name[5]+'.csv')
# (831157, 5)
right = pd.read_csv(path+name[6]+'.csv')
# (49601, 4)
proj = pd.read_csv(path+name[7]+'.csv')
# (40102, 4)
law = pd.read_csv(path+name[8]+'.csv')
#(4388, 4)
shixin = pd.read_csv(path+name[9]+'.csv')
# (1376943, 5)
recruit = pd.read_csv(path+name[10]+'.csv')
recruit['PNUM'] = recruit['PNUM'].replace('若干','1').apply(get_num)

# (6693, 4)
qualif = pd.read_csv(path+name[11]+'.csv')


all = pd.concat([tr,test])
# all.fillna(-1,inplace=True)

def ch_feature():
    df = ch.copy()

    # 每年变更事项次数统计
    df['ALT_year'] = df['ALTDATE'].apply(lambda x: str(x)[0:4])
    temp0 = pd.crosstab(df.EID, df.ALT_year, margins=True).reset_index()
    temp0 = temp0.drop(temp0.index[-1])

    # 对ALTERNO 进行每种变更类型次数统计
    zj = df[['EID', 'ALTERNO']]
    temp3 = pd.crosstab(zj.EID, zj.ALTERNO, margins=False).reset_index()

    # 企业最后一次变更后的类型是什么，资本是多少
    alter_last = df.sort_values("ALTDATE")
    alter_last = alter_last.drop_duplicates("EID", keep="last")
    alter_last['ch_ALTDATE_year'] = alter_last['ALTDATE'].apply(lambda x: x.split('-')[0]).astype(int)
    alter_last['ch_ALTDATE_mon'] = alter_last['ALTDATE'].apply(lambda x: x.split('-')[1]).astype(int)
    alter_last_1 = alter_last[["EID", "ALTERNO", 'ch_ALTDATE_year', 'ch_ALTDATE_mon']]
    temp4 = pd.get_dummies(alter_last_1, columns=['ALTERNO'], prefix='ALTERNO')

    fin = temp0.merge(temp3, how='left', on='EID').merge(temp4, how='left', on='EID')
    return fin


# 加入fzjh数据
def bra_feature():
    df = bra.copy()

    # 分支机构总数量，分支机构本省与外省的数量
    temp0 = pd.crosstab([df.EID], df.IFHOME, margins=True).reset_index()
    temp0.columns = ['EID', 'fzjg_ifhome_0', 'fzjg_ifhome_1', 'fzjg_ifhome_all']
    temp0 = temp0.drop(temp0.index[-1])

    # 分支机构仍然正常营业的数量，不正常的数量以及总量
    df['normal'] = df['B_ENDYEAR'].isnull().astype(int)
    temp1 = pd.crosstab([df.EID], df.normal, margins=True).reset_index()
    temp1.columns = ['EID', 'fzjg_normal_0', 'fzjg_normal_1', 'fzjg_normal_all']
    temp1 = temp1.drop(temp1.index[-1])

    # piu piu

    df['branch_active_year'] = df['B_ENDYEAR'] - df['B_REYEAR']
    feat = df.groupby('EID').agg({'B_ENDYEAR': {'n_branch': 'size',
                                                'n_end_branch': 'count',
                                                'last_end_branch': 'max',
                                                'first_end_branch': 'max',
                                                'median_end_branch': 'median'},
                                  'B_REYEAR': {'last_start_branch': 'max',
                                               'first_start_branch': 'min'},
                                  'branch_active_year': {'branch_active_year': 'mean'}})
    feat.columns = feat.columns.droplevel(0)
    feat['EID'] = feat.index
    feat['n_active_branch'] = feat['n_branch'] - feat['n_end_branch']
    feat['active_branch_rate'] = feat['n_active_branch'] / (feat['n_branch'] + 0.1)

    fin = temp0.merge(temp1, on='EID', how='left').merge(feat, on='EID', how='left')
    return fin


# 加入tz 数据
def tz_feature():
    df = tz.copy()

    temp0 = wu_count(df, ['EID'], 'BTEID', 'tz_BTEID_count')

    # 投资机构仍然正常营业的数量，不正常的数量以及总量,比例
    df['normal'] = df['BTENDYEAR'].isnull().astype(int)
    temp2 = pd.crosstab([df.EID], df.normal, margins=True).reset_index()
    temp2.columns = ['EID', 'tz_normal_0', 'tz_normal_1', 'tz_normal_all']
    temp2 = temp2.drop(temp2.index[-1])
    temp2['tz_normal1_ratio'] = temp2['tz_normal_1'] / (temp2['tz_normal_all'] + 0.1)
    temp2['tz_normal0_ratio'] = temp2['tz_normal_0'] / (temp2['tz_normal_all'] + 0.1)
    temp2.drop(['tz_normal_all'], axis=1, inplace=True)

    # piu piu
    mean_btbl = df.groupby('EID', as_index=False)['BTBL'].agg({'mean_btbl': 'mean'})
    n_home_invest = df.groupby('EID', as_index=False)['IFHOME'].agg({'n_home_invest': 'sum'})

    id_label_dict = dict(zip(tr['EID'].values, tr['TARGET'].values))
    df['btlabel'] = df['BTEID'].map(id_label_dict)
    df['idlabel'] = df['EID'].map(id_label_dict)
    n_negitive_invest = df.groupby('EID', as_index=False)['btlabel'].agg({'n_negitive_invest': 'sum'})
    n_negitive_invested = df.groupby('BTEID', as_index=False)['idlabel'].agg({'n_negitive_invested': 'sum'})
    n_negitive_invested.rename(columns={'BTEID': 'EID'}, inplace=True)

    fin = temp0.merge(temp2, on='EID', how='left').merge(mean_btbl, on='EID', how='left').merge(n_home_invest, on='EID',
                                                                                                how='left' \
                                                                                                ).merge(
        n_negitive_invest, on='EID', how='left').merge(n_negitive_invested, on='EID', how='left')

    return fin


# 加入权利数据
def right_feature():
    df = right.copy()

    temp00 = wu_nunique(df, ['EID'], 'TYPECODE', 'zhuanli_TYPECODE_nunique')

    # 企业申请的专利利总数量
    temp0 = wu_count(df, ['EID'], 'TYPECODE', 'zhuanli_TYPECODE_count')

    # 专利类型几种
    temp1 = wu_nunique(df, ['EID'], 'ASKDATE', 'zhuanli_ASKDATE_nunique')

    # 近五年2010-2015每年的专利申请数量和它的增长率
    df['ASKDATE_year'] = df['ASKDATE'].apply(lambda x: str(x)[0:4]).astype(int)
    data = df[df.ASKDATE_year >= 2010]
    temp3 = pd.crosstab([data.EID], data.ASKDATE_year, margins=False).reset_index()
    temp3.columns = ['EID', 'zhuanli_ASKDATE_2010', 'zhuanli_ASKDATE_2011', 'zhuanli_ASKDATE_2012',
                     'zhuanli_ASKDATE_2013', 'zhuanli_ASKDATE_2014',
                     'zhuanli_ASKDATE_2015']

    # piu piu
    df['FBDATE'] = df['FBDATE'].apply(lambda x: x if x is np.nan else (2020 - int(x[:4])) * 12 - int(x[-2:]))
    df['weight'] = df['ASKDATE'].apply(lambda x: 0.5 ** (2015 - int(x[:4])))
    df['ASKDATE'] = df['ASKDATE'].apply(lambda x: x if x is np.nan else (2020 - int(x[:4])) * 12 - int(x[-2:]))

    n_right2 = wu_sum(df, ['EID'], 'weight', 'piu_weight_sum')
    n_right3 = wu_count(df, ['EID'], 'FBDATE', 'piu_FBDATE_count')
    righttype = pd.get_dummies(df['RIGHTTYPE'], prefix='piu_RIGHTTYPE')
    righttype = pd.concat([df['EID'], righttype], axis=1)
    righttype = righttype.groupby(['EID'], as_index=False).sum()
    last_fbdate = df.groupby('EID', as_index=False)['FBDATE'].agg({'last_fbdate': 'min'})
    last_askdate = df.groupby('EID', as_index=False)['ASKDATE'].agg({'last_askdate': 'min'})

    fin = temp0.merge(temp1, on='EID', how='left').merge(temp3, on='EID', how='left').merge(n_right2, on='EID',
                                                                                            how='left').merge( \
        n_right3, on='EID', how='left').merge(righttype, on='EID', how='left').merge(last_fbdate, on='EID',
                                                                                     how='left').merge( \
        last_askdate, on='EID', how='left')
    fin = fin.merge(temp00, on=['EID'], how='left')
    return fin


# 加入项目数据

def proj_feature():
    df = proj.copy()
    df["DJDATE"] = pd.to_datetime(df["DJDATE"])
    df["DJDATE_year"] = df["DJDATE"].dt.year
    df["DJDATE_month"] = df["DJDATE"].dt.month
    df["DJDATE_year"] = df["DJDATE_year"] + df["DJDATE_month"] / 12.0

    temp1 = wu_count(df, ['EID'], 'TYPECODE', 'xiangmu_TYPECODE_count')
    temp2 = wu_nunique(df, ['EID'], 'TYPECODE', 'xiangmu_TYPECODE_nunique')

    fin = temp1.merge(temp2, on=['EID'], how='left')
    return fin


# 加入案件数据
def law_feature():
    df = law.copy()

    # 企业anjian总数量
    temp0 = wu_count(df, ['EID'], 'TYPECODE', 'anjian_TYPECODE_count')

    # 总得罚金
    temp1 = wu_sum(df, ['EID'], 'LAWAMOUNT', 'anjian_LAWAMOUNT_sum')

    # 最后一次
    lawsuit_last = df.sort_values("LAWDATE")
    lawsuit_last = lawsuit_last.drop_duplicates("EID", keep="last")
    lawsuit_last = lawsuit_last[["EID", "LAWDATE", "LAWAMOUNT"]]
    lawsuit_last['law_LAWDATE_year'] = lawsuit_last['LAWDATE'].apply(lambda x: str(x)[:4]).astype(int)
    lawsuit_last['law_LAWDATE_mon'] = lawsuit_last['LAWDATE'].apply(lambda x: str(x)[5:7]).astype(int)
    temp2 = lawsuit_last[['EID', 'law_LAWDATE_year', 'law_LAWDATE_mon']]

    fin = temp0.merge(temp1, on='EID', how='left').merge(temp2, on='EID', how='left')
    return fin


# 加入失信数据
def shixin_feature():
    df = shixin.copy()
    df['FBDATE'] = df['FBDATE'].str.replace('年', '-')
    df['FBDATE'] = df['FBDATE'].str.replace('月', '')
    df['shixin_FBDATE_year'] = df['FBDATE'].apply(lambda x: str(x)[:4]).astype(int)
    df['shixin_FBDATE_mon'] = df['FBDATE'].apply(lambda x: str(x)[5:7]).astype(int)

    # 企业shixin总数量
    temp0 = wu_count(df, ['EID'], 'TYPECODE', 'shixin_TYPECODE_count')
    temp1 = wu_nunique(df, ['EID'], 'TYPECODE', 'shixin_TYPECODE_nunique')

    # 总共还没结束失信结束数量,和比例
    df['normal'] = df['SXENDDATE'].isnull().astype(int)
    temp2 = pd.crosstab([df.EID], df.normal, margins=True).reset_index()
    temp2.columns = ['EID', 'shixin_normal_0', 'shixin_normal_1', 'shixin_normal_all']
    temp2 = temp2.drop(temp2.index[-1])
    temp2['shixin_normal0_ratio'] = temp2['shixin_normal_0'] / (temp2['shixin_normal_all'])
    temp2['shixin_normal1_ratio'] = temp2['shixin_normal_1'] / (temp2['shixin_normal_all'])

    fin = temp0.merge(temp2, on='EID', how='left').merge(temp1, on='EID', how='left')
    return fin


# 加入招聘数据
def recruit_feature():
    df = recruit.copy()

    # 企业zhaopin员工总数量
    temp0 = wu_sum(df, ['EID'], 'PNUM', 'zhaopin_PNUM_sum')

    # 企业zhaoppin网站的数量
    temp1 = wu_nunique(df, ['EID'], 'WZCODE', 'zhaopin_WZCODE_nunique')

    # 最后一次
    temp2 = df.sort_values(['RECDATE']).drop_duplicates('EID', keep='last')
    temp2 = temp2[['EID', 'PNUM', 'RECDATE']]
    temp2['zhaopin_RECDATE_maxyear'] = temp2['RECDATE'].apply(lambda x: str(x)[0:4]).astype(int)
    temp2['zhaopin_RECDATE_maxmon'] = temp2['RECDATE'].apply(lambda x: str(x)[5:7]).astype(int)
    temp2['zhaopin_2017_dist'] = datetime(2017, 8, 1) - pd.to_datetime(temp2['RECDATE'])
    temp2['zhaopin_2017_dist'] = temp2['zhaopin_2017_dist'].apply(lambda x: str(x)[:3]).astype(int)
    temp2 = temp2[['EID', 'zhaopin_RECDATE_maxyear', 'zhaopin_RECDATE_maxmon', 'zhaopin_2017_dist']]

    fin = temp0.merge(temp1, on='EID', how='left').merge(temp2, on='EID', how='left')
    return fin


def qualification_feat(df, df_features):
    df_features = df_features.drop_duplicates('EID', keep='last')
    df = df.merge(df_features[['EID', 'ADDTYPE']], on='EID', how='left')
    return df


def qualif_feature():
    df = qualif.copy()
    temp0 = df.groupby('EID').size().reset_index().fillna(-1)
    temp0.columns = ['EID', 'qualif_size']

    df['BEGINDATE'] = df['BEGINDATE'].apply(
        lambda x: datetime(int(re.findall('[0-9]+', str(x))[0]), int(re.findall('[0-9]+', str(x))[1]), 1))
    df['EXPIRYDATE'] = df['EXPIRYDATE'].apply(
        lambda x: datetime(int(re.findall('[0-9]+', str(x))[0]), int(re.findall('[0-9]+', str(x))[1]),
                           1) if x is not np.nan else pd.NaT)

    fin = temp0
    fin = qualification_feat(fin, df)
    fin = pd.get_dummies(fin, columns=['ADDTYPE'])
    return fin

def merge_static(df):
    for col in ['HY']:
        for num_col in['ZCZB','MPNUM','INUM','FINZB','FSTINUM','TZINUM','ENUM']:
            for method in [np.mean,np.median,np.max,np.min,np.std,np.sum]:
                temp1 = df.groupby([col])[num_col].apply(method).reset_index()
                coltwo =col+'_'+num_col+'('+method.__name__+')'
                temp1.columns=[col,coltwo]
                df = df.merge(temp1,on=[col],how='left')
                df['dist_'+coltwo+'_rank'] = (df[num_col]-df[coltwo]).rank()
    return df


Endf = En.copy()
# Endf['HY'] = Endf['HY'].fillna(0)
col = ['MPNUM','INUM','ENUM','FINZB','FSTINUM','TZINUM']
Endf[['ETYPE_MPNUM_rank','ETYPE_INUM_rank','ETYPE_ENUM_rank','ETYPE_FINZB_rank','ETYPE_FSTINUM_rank','ETYPE_TZINUM_rank']]=Endf.groupby(['ETYPE'])[col].rank()

Endf['finzb2'] = Endf['FINZB'] / (Endf['ZCZB'] + 0.1)
Endf['mpnum2'] = Endf['MPNUM'] / (Endf['ZCZB'] + 0.1)
Endf['inum2'] = Endf['INUM'] / (Endf['ZCZB'] + 0.1)
Endf['fstinum2'] = Endf['FSTINUM'] / (Endf['ZCZB'] + 0.1)
Endf['tzinum2'] = Endf['TZINUM'] / (Endf['ZCZB'] + 0.1)
Endf['sumnum'] = Endf[['MPNUM','INUM','FSTINUM','TZINUM']].sum(axis=1)


col = ['ZCZB','MPNUM','INUM','ENUM','FINZB','FSTINUM','TZINUM']
Endf[['HY_ZCZB_rank','HY_MPNUM_rank','HY_INUM_rank','HY_ENUM_rank','HY_FINZB_rank','HY_FSTINUM_rank','HY_TZINUM_rank']]=Endf.groupby(['HY'])[col].rank()


# Endf = merge_nunique(Endf,['HY'],'ETYPE','fuck_a')
# Endf = merge_nunique(Endf,['HY'],'MPNUM','fuck_b')
# Endf = merge_nunique(Endf,['HY'],'INUM','fuck_c')
# Endf = merge_nunique(Endf,['HY'],'ENUM','fuck_d')
# Endf = merge_nunique(Endf,['HY'],'FSTINUM','fuck_e')
# Endf = merge_nunique(Endf,['HY'],'TZINUM','fuck_f')


Endf['ETYPE_copy'] = Endf['ETYPE'].copy()
Endf = pd.get_dummies(Endf,columns=['ETYPE'],prefix='ETYPE')


# Endf = merge_static(Endf)
# Endf = pd.get_dummies(Endf,columns=['PROV'],prefix='PROV')
# Endf = pd.get_dummies(Endf,columns=['HY'],prefix='HY')


chdf = ch_feature()
bradf = bra_feature()
tzdf = tz_feature()
rightdf = right_feature()
projdf = proj_feature()
lawdf = law_feature()
shixindf = shixin_feature()
recruitdf = recruit_feature()

df_all= all.merge(Endf,on=['EID'],how='left')

df_all = df_all.merge(chdf,on=['EID'],how='left').merge(bradf,on=['EID'],how='left').merge(\
            tzdf,on=['EID'],how='left').merge(rightdf,on=['EID'],how='left').merge(projdf,on=['EID'],how='left').merge(\
            lawdf,on=['EID'],how='left').merge(shixindf,on=['EID'],how='left').merge(recruitdf,on=['EID'],how='left')
# .merge(qualifdf,on=['EID'],how='left')


df_all.shape

df_all['EID_num'] = df_all['EID'].apply(get_num)
df_all['EID_num_fu'] = list(map(lambda x, y: -x if y == 11 else x, df_all['EID_num'], df_all['PROV']))
# eid_all = df_all.pop('EID')


na_col = ['2013', '2014', '2015', 'All', '01', '02', '03', '04', '05', '10', '12', '13', '14', '27', '99', 'A_015',
          'ch_ALTDATE_year',
          'ch_ALTDATE_mon', 'ALTERNO_01', 'ALTERNO_02', 'ALTERNO_03', 'ALTERNO_04', 'ALTERNO_05', 'ALTERNO_10',
          'ALTERNO_12', 'ALTERNO_13',
          'ALTERNO_14', 'ALTERNO_27', 'ALTERNO_99', 'ALTERNO_A_015',
          'fzjg_ifhome_0', 'fzjg_ifhome_1', 'fzjg_ifhome_all', 'fzjg_normal_0', 'fzjg_normal_1', 'fzjg_normal_all',
          'tz_BTEID_count', 'tz_normal_0', 'tz_normal_1', 'tz_normal1_ratio', 'tz_normal0_ratio',
          'zhuanli_TYPECODE_nunique', 'zhuanli_ASKDATE_nunique', 'zhuanli_ASKDATE_2010', 'zhuanli_ASKDATE_2011',
          'zhuanli_ASKDATE_2012',
          'zhuanli_ASKDATE_2013', 'zhuanli_ASKDATE_2014', 'zhuanli_ASKDATE_2015',
          'xiangmu_TYPECODE_nunique',
          'anjian_TYPECODE_count', 'anjian_LAWAMOUNT_sum', 'law_LAWDATE_year', 'law_LAWDATE_mon',
          'shixin_TYPECODE_nunique', 'shixin_normal_0', 'shixin_normal_1', 'shixin_normal_all', 'shixin_normal0_ratio',
          'shixin_normal1_ratio',
          'zhaopin_PNUM_sum', 'zhaopin_WZCODE_nunique', 'zhaopin_RECDATE_maxyear', 'zhaopin_RECDATE_maxmon',
          'zhaopin_2017_dist',
          'piu_weight_sum', 'last_fbdate', 'last_askdate', 'piu_FBDATE_count', 'piu_RIGHTTYPE_11', 'piu_RIGHTTYPE_12',
          'piu_RIGHTTYPE_20',
          'piu_RIGHTTYPE_30', 'piu_RIGHTTYPE_40', 'piu_RIGHTTYPE_50', 'piu_RIGHTTYPE_60',
          'mean_btbl', 'n_home_invest', 'n_negitive_invest', 'n_negitive_invested',
          'first_end_branch', 'n_branch', 'median_end_branch', 'n_end_branch', 'last_end_branch', 'first_start_branch',
          'last_start_branch',
          'branch_active_year', 'n_active_branch', 'active_branch_rate',
          'zhuanli_TYPECODE_count', 'xiangmu_TYPECODE_count', 'shixin_TYPECODE_count'
          ]

# 'HY_order','ETYPE_copy_order'
df_all[na_col] = df_all[na_col].fillna(0)


x_train = df_all[:len(tr)].drop(['EID','ENDDATE','EID_num','n_branch','n_end_branch','n_active_branch','ETYPE_copy'],axis=1)
x_test = df_all[len(tr):].drop(['TARGET','ENDDATE','EID_num','n_branch','n_end_branch','n_active_branch','ETYPE_copy'],axis=1)


"""
# x_train去除16.0, 91.0, 94.0HY
index_HY = list(x_train[x_train.HY.isin([16.0, 91.0, 94.0])].index)
index_0 = list(x_train[x_train['ZCZB']<=0].index)+list(x_train[x_train['ZCZB']>5*1e7].index)+index_HY

x_train = x_train.drop(index_HY)
x_train.index = range(len(x_train))
y = y.drop(index_HY)
y.index = range(len(y))

"""

# x_train.fillna(0,inplace=True)
# x_test.fillna(0,inplace=True)
print(df_all.shape,x_train.shape,x_test.shape)
print('\n')

col = ['zhuanli_TYPECODE_nunique', 'xiangmu_TYPECODE_nunique', 'shixin_TYPECODE_nunique', 'IENUM']
x_train.drop(col, axis=1, inplace=True)
x_test.drop(col, axis=1, inplace=True)
print(x_train.shape)

# Train and predict
train_11_all = x_train
train_12_all = x_train
for i in range(3):
    train_11_all = train_11_all.append(x_train[x_train.PROV == 11]).reset_index(drop=True)
    train_12_all = train_12_all.append(x_train[x_train.PROV == 12]).reset_index(drop=True)

y_train_11_all = train_11_all.pop('TARGET')
y_train_12_all = train_12_all.pop('TARGET')

test_11 = x_test[(x_test.PROV == 11)]
EID_test_11 = test_11.pop('EID')

test_12 = x_test[(x_test.PROV == 12)]
EID_test_12 = test_12.pop('EID')


# -----------------------------------lgb试特征-------------------------

lgb_test_11 = test_11
lgb_test_12 = test_12

params = dict()
params['learning_rate'] = 0.02
# params['boosting_type'] = 'dart'
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'auc'
params['feature_fraction'] = 0.8            # feature_fraction
params['bagging_fraction'] = 0.8            # sub_row
params['bagging_freq'] = 10
params['num_leaves'] = 2**6
params['verbose']=1
params['max_bin'] = 10
params['min_data_in_leaf'] = 50           # min_data_in_leaf
params['min_sum_hessian_in_leaf'] = 0.05     # min_sum_hessian_in_leaf
params['lambda_l2']=10

sub = pd.DataFrame(columns=['EID', 'FORTARGET', 'PROB'])
sub['EID'] = EID_test_11.values
sub.fillna(0, inplace=True)


def stratified(lgbm=False, xgbm=False):
    kfold = 5
    cv_scores = []
    cv_iter = []
    sss = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=2017)
    for i, (train_index, test_index) in enumerate(sss.split(train_11_all, y_train_11_all)):
        print('[Fold %d]' % (i + 1))
        X_train, X_valid = train_11_all.ix[train_index], train_11_all.ix[test_index]
        y_train, y_valid = y_train_11_all[train_index], y_train_11_all[test_index]
        print("TRAIN:", Counter(y_train), "TEST:", Counter(y_valid))

        if lgbm:
            print('start lgb--------------------------------')
            d_train = lgb.Dataset(X_train, y_train)
            d_valid = lgb.Dataset(X_valid, y_valid)
            model = lgb.train(params, d_train, num_boost_round=758, valid_sets=[d_valid], verbose_eval=False)
            pre = model.predict(X_valid).reshape((X_valid.shape[0], 1))

            X_train.ix[test_index] = pre
            p_test = model.predict(lgb_test_11)
            sub['PROB'] += p_test / kfold
            cv_iter.append(roc_auc_score(y_valid, pre))
            print(cv_scores, cv_iter, '\n')

        if xgbm:
            print('start xgb')
            d_train = xgb.DMatrix(X_train, y_train)
            d_valid = xgb.DMatrix(X_valid, y_valid)
            model = xgb.train(params, d_train, num_boost_round=2000, evals=[(d_valid, 'eval')], verbose_eval=False,
                              early_stopping_rounds=50, show_stdv=True)
            #             p_test = model.predict(d_test,ntree_limit=model.best_iteration)
            #             sub['PROB'] += p_test/kfold
            cv_scores.append(model.best_iteration)
            cv_iter.append(roc_auc_score(y_valid, pre))
            print(cv_scores, cv_iter, '\n')

    print(cv_scores)
    print(cv_iter)
    print(np.mean(cv_scores), np.mean(cv_iter))
    print(np.std(cv_iter))


stratified(lgbm = True,xgbm = False)
# 757.8 [801, 577, 710, 678, 1023]