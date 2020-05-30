import pandas as pd
import numpy as np
import matplotlib
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def intx(x):
    return int(x)


entbase=pd.read_csv('1entbase.csv')
entbase = entbase.drop_duplicates()
alter=pd.read_csv('2alter.csv')
alter = alter.drop_duplicates()
branch=pd.read_csv('3branch.csv')
branch = branch.drop_duplicates()
invest=pd.read_csv('4invest.csv')
invest = invest.drop_duplicates()
right=pd.read_csv('5right.csv')
right = right.drop_duplicates()
project=pd.read_csv('6project.csv')
project = project.drop_duplicates()
lawsuit=pd.read_csv('7lawsuit.csv')
lawsuit = lawsuit.drop_duplicates()
breakfaith=pd.read_csv('8breakfaith.csv')
breakfaith = breakfaith.drop_duplicates()
recruit=pd.read_csv('9recruit.csv')
recruit = recruit.drop_duplicates()
train=pd.read_csv('train.csv')
predict=pd.read_csv('evaluation_public.csv')
#特征：企业年龄
entbase['RGYEAR']=entbase['RGYEAR'].map(intx)
entbase['start_old']=2015-entbase['RGYEAR']

#特征：企业变更次数alter_num
alter_num=alter.groupby('EID',as_index=False)['EID'].agg({'alter_num':'count'})

#特征：企业分支个数
branch_num=branch.groupby('EID',as_index=False)['TYPECODE'].agg({'branch_num':'count'})
#倒闭企业个数
branchend_num=branch[branch['B_ENDYEAR'].notnull()].groupby('EID',as_index=False)['TYPECODE'].agg({'branchend_num':'count'})


#特征：投资企业个数
invest_num=invest.groupby('EID',as_index=False)['BTEID'].agg({'invest_num':'count'})
#投资倒闭数
investend_num=invest[invest['BTENDYEAR'].notnull()].groupby('EID',as_index=False)['BTEID'].agg({'investend_num':'count'})

#特征：权利数目
right_num=right.groupby('EID',as_index=False)['TYPECODE'].agg({'right_num':'count'})
right_fy_num=right[right['FBDATE'].notnull()].groupby('EID',as_index=False)['TYPECODE'].agg({'right_fy_num':'count'})
right_type_num=right.groupby('EID',as_index=False)['RIGHTTYPE'].agg({'right_TYPE_num':'count'})
#特征：项目数
project_num=project.groupby('EID',as_index=False)['TYPECODE'].agg({'project_num':'count'})

#特征：案件数目
lawsuit_num_max_sum_median=lawsuit.groupby('EID',as_index=False)['LAWAMOUNT'].agg({'lawsuit_num':'count','lawsuit_max':'max','lawsuit_sum':'sum','lawsuit_median':'median'})

#HY onehot后进行pca降维
entbaseid = entbase[['EID']]
hy_dummies = pd.get_dummies(entbase['HY'])

pca = PCA(n_components=15)
hy_compressed = pca.fit_transform(hy_dummies)
hy_compressed_df = pd.DataFrame(hy_compressed,
             columns=list(['hy'+str(x) for x in range(1,16)]))
hy_dummies = entbaseid.join(hy_compressed_df)

etype_dummies=pd.get_dummies(entbase.ETYPE,prefix='etype')
etype_dummies_eid = entbase[['EID']]
etype_dummies=pd.concat([etype_dummies_eid,etype_dummies],axis=1)

#特征：失信数目
breakfaith_num=breakfaith.groupby('EID',as_index=False)['TYPECODE'].agg({'breakfaith_num':'count'})

#特征：招聘数目
recruit_num=recruit.groupby('EID',as_index=False)['WZCODE'].agg({'recruit_num':'count'})
recruit_people=recruit.groupby('EID',as_index=False)['RECRNUM'].agg({'recruit_people':'sum'})


train=pd.merge(train,entbase,how='left',on='EID')
#train=pd.merge(train,alter_num,how='left',on='EID')
train=pd.merge(train,etype_dummies,how='left',on='EID')
#train=pd.merge(train,branch_num,how='left',on='EID')
#train=pd.merge(train,branchend_num,how='left',on='EID')
#train=pd.merge(train,invest_num,how='left',on='EID')
#train=pd.merge(train,investend_num,how='left',on='EID')
#train=pd.merge(train,right_num,how='left',on='EID')
#train=pd.merge(train,project_num,how='left',on='EID')
train=pd.merge(train,lawsuit_num_max_sum_median,how='left',on='EID')
#train=pd.merge(train,breakfaith_num,how='left',on='EID')
train=pd.merge(train,right_type_num,how='left',on='EID')
train=pd.merge(train,recruit_people,how='left',on='EID')
train=pd.merge(train,hy_dummies,how='left',on='EID')

predict=pd.merge(predict,entbase,how='left',on='EID')
#predict=pd.merge(predict,alter_num,how='left',on='EID')
predict=pd.merge(predict,etype_dummies,how='left',on='EID')
#predict=pd.merge(predict,branch_num,how='left',on='EID')
#predict=pd.merge(predict,branchend_num,how='left',on='EID')
#predict=pd.merge(predict,invest_num,how='left',on='EID')
#predict=pd.merge(predict,investend_num,how='left',on='EID')
#predict=pd.merge(predict,right_num,how='left',on='EID')
#predict=pd.merge(predict,project_num,how='left',on='EID')
predict=pd.merge(predict,lawsuit_num_max_sum_median,how='left',on='EID')
#predict=pd.merge(predict,breakfaith_num,how='left',on='EID')
predict=pd.merge(predict,right_type_num,how='left',on='EID')
predict=pd.merge(predict,recruit_people,how='left',on='EID')
predict=pd.merge(predict,hy_dummies,how='left',on='EID')


#------------------------------------------------------------------------------------------------------------------------------------------
#这部分代码主要是统计2015年各张表的总数


#2015年更改的次数
alter['alter_year'] = alter['ALTDATE'].map(lambda x:int(x.split('-')[0]))
alter_2015 = alter[alter['alter_year'] == 2015]
alter_2015['alter_2015'] = 1
alter_2015 = alter_2015.groupby(['EID']).agg('sum').reset_index()
alter_2015 = alter_2015[['EID','alter_2015']]

#2015年分支成立的次数
branch_2015 = branch[branch['B_REYEAR'] == 2015]
branch_2015['branch_2015'] = 1
branch_2015 = branch_2015.groupby(['EID']).agg('sum').reset_index()
branch_2015 = branch_2015[['EID','branch_2015']]

#2015年分支关闭的次数
branch_c_2015 = branch[branch['B_ENDYEAR'] == 2015]
branch_c_2015['branch_c_2015'] = 1
branch_c_2015 = branch_c_2015.groupby(['EID']).agg('sum').reset_index()
branch_c_2015 = branch_c_2015[['EID','branch_c_2015']]

#2015年分支成立并且在省外的次数
branch_no_2015 = branch[(branch['B_REYEAR'] == 2015) & (branch['IFHOME'] == 0)]
branch_no_2015['branch_no_2015'] = 1
branch_no_2015 = branch_no_2015.groupby(['EID']).agg('sum').reset_index()
branch_no_2015 = branch_no_2015[['EID','branch_no_2015']]

#2015年分支倒闭并且在省外的次数
branch_c_no_2015 = branch[(branch['B_ENDYEAR'] == 2015) & (branch['IFHOME'] == 0)]
branch_c_no_2015['branch_c_no_2015'] = 1
branch_c_no_2015 = branch_c_no_2015.groupby(['EID']).agg('sum').reset_index()
branch_c_no_2015 = branch_c_no_2015[['EID','branch_c_no_2015']]

#2015年投资成立并且在省外的次数
invest_no_2015 = invest[(invest['BTYEAR'] == 2015) & (branch['IFHOME'] == 0)]
invest_no_2015['invest_no_2015'] = 1
invest_no_2015 = invest_no_2015.groupby(['EID']).agg('sum').reset_index()
invest_no_2015 = invest_no_2015[['EID','invest_no_2015']]

#2015年投资倒闭并且在省外的次数
invest_c_no_2015 = invest[(invest['BTENDYEAR'] == 2015) & (branch['IFHOME'] == 0)]
invest_c_no_2015['invest_c_no_2015'] = 1
invest_c_no_2015 = invest_c_no_2015.groupby(['EID']).agg('sum').reset_index()
invest_c_no_2015 = invest_c_no_2015[['EID','invest_c_no_2015']]

#2015年投资成立的次数
invest_2015 = invest[invest['BTYEAR'] == 2015]
invest_2015['invest_2015'] = 1
invest_2015 = invest_2015.groupby(['EID']).agg('sum').reset_index()
invest_2015 = invest_2015[['EID','invest_2015']]

#2015年投资倒闭的次数
invest_c_2015 = invest[invest['BTENDYEAR'] == 2015]
invest_c_2015['invest_c_2015'] = 1
invest_c_2015 = invest_c_2015.groupby(['EID']).agg('sum').reset_index()
invest_c_2015 = invest_c_2015[['EID','invest_c_2015']]


#2015年权利申请的次数
right['right_year'] = right['ASKDATE'].map(lambda x:int(x.split('-')[0]))
right_2015 = right[right['right_year'] == 2015]
right_2015['right_2015'] = 1
right_2015 = right_2015.groupby(['EID']).agg('sum').reset_index()
right_2015 = right_2015[['EID','right_2015']]

#2015年权利赋予的次数
right1 = right[right['FBDATE'].notnull()]
right1['right_fy_year'] = right1['FBDATE'].map(lambda x:int(x.split('-')[0]))
right_fy_2015 = right1[right1['right_fy_year'] == 2015]
right_fy_2015['right_fy_2015'] = 1
right_fy_2015 = right_fy_2015.groupby(['EID']).agg('sum').reset_index()
right_fy_2015 = right_fy_2015[['EID','right_fy_2015']]

#2015年项目的次数
project['project_year'] = project['DJDATE'].map(lambda x:int(x.split('-')[0]))
project_2015 = project[project['project_year'] == 2015]
project_2015['project_2015'] = 1
project_2015 = project_2015.groupby(['EID']).agg('sum').reset_index()
project_2015 = project_2015[['EID','project_2015']]

#2015年项目并且在省外的次数
project_no_2015 = project[(project['project_year'] == 2015) & (branch['IFHOME'] == 0)]
project_no_2015['project_no_2015'] = 1
project_no_2015 = project_no_2015.groupby(['EID']).agg('sum').reset_index()
project_no_2015 = project_no_2015[['EID','project_no_2015']]


#2015年案件执行的次数
lawsuit['lawsuit_year'] = lawsuit['LAWDATE'].map(lambda x:int(x.split('-')[0]))
lawsuit_2015 = lawsuit[lawsuit['lawsuit_year'] == 2015]
lawsuit_2015['lawsuit_2015'] = 1
lawsuit_2015 = lawsuit_2015.groupby(['EID']).agg('sum').reset_index()
lawsuit_2015 = lawsuit_2015[['EID','lawsuit_2015']]

#2015年失信的次数
breakfaith['breakfaith_year'] = breakfaith['FBDATE'].map(lambda x:int(x.split('/')[0]))
breakfaith_2015 = breakfaith[breakfaith['breakfaith_year'] == 2015]
breakfaith_2015['breakfaith_2015'] = 1
breakfaith_2015 = breakfaith_2015.groupby(['EID']).agg('sum').reset_index()
breakfaith_2015 = breakfaith_2015[['EID','breakfaith_2015']]

#2015年各种更改事项的次数
alterno_num2015=alter[alter['alter_year'] == 2015]
alterno=alterno_num2015['ALTERNO'].drop_duplicates().tolist()
alterno_dummies2015=pd.get_dummies(alterno_num2015.ALTERNO,prefix='2015_ALTERNO')
alterno_dummies2015=pd.concat([alterno_num2015,alterno_dummies2015],axis=1)
alterno_dummies2015=alterno_dummies2015.drop(['ALTERNO','ALTDATE','ALTBE','ALTAF','alter_year'],axis=1)
alterno_dummies2015=alterno_dummies2015.groupby('EID',as_index=False).sum()


train=pd.merge(train,alter_2015,how='left',on=['EID'])
train=pd.merge(train,branch_2015,how='left',on='EID')
train=pd.merge(train,branch_c_2015,how='left',on=['EID'])
train=pd.merge(train,invest_2015,how='left',on='EID')
train=pd.merge(train,right_2015,how='left',on=['EID'])
train=pd.merge(train,right_fy_2015,how='left',on='EID')
train=pd.merge(train,lawsuit_2015,how='left',on=['EID'])
train=pd.merge(train,breakfaith_2015,how='left',on='EID')
train=pd.merge(train,alterno_dummies2015,how='left',on='EID')
train=pd.merge(train,invest_c_2015,how='left',on='EID')
train=pd.merge(train,branch_no_2015,how='left',on=['EID'])
train=pd.merge(train,branch_c_no_2015,how='left',on='EID')
train=pd.merge(train,invest_no_2015,how='left',on='EID')
train=pd.merge(train,invest_c_no_2015,how='left',on='EID')
train=pd.merge(train,project_2015,how='left',on='EID')
train=pd.merge(train,project_no_2015,how='left',on='EID')

predict=pd.merge(predict,alter_2015,how='left',on=['EID'])
predict=pd.merge(predict,branch_2015,how='left',on='EID')
predict=pd.merge(predict,branch_c_2015,how='left',on=['EID'])
predict=pd.merge(predict,invest_2015,how='left',on='EID')
predict=pd.merge(predict,right_2015,how='left',on=['EID'])
predict=pd.merge(predict,right_fy_2015,how='left',on='EID')
predict=pd.merge(predict,lawsuit_2015,how='left',on=['EID'])
predict=pd.merge(predict,breakfaith_2015,how='left',on='EID')
predict=pd.merge(predict,alterno_dummies2015,how='left',on='EID')
predict=pd.merge(predict,invest_c_2015,how='left',on='EID')
predict=pd.merge(predict,branch_no_2015,how='left',on=['EID'])
predict=pd.merge(predict,branch_c_no_2015,how='left',on='EID')
predict=pd.merge(predict,invest_no_2015,how='left',on='EID')
predict=pd.merge(predict,invest_c_no_2015,how='left',on='EID')
predict=pd.merge(predict,project_2015,how='left',on='EID')
predict=pd.merge(predict,project_no_2015,how='left',on='EID')

entbase=pd.read_csv('1entbase.csv')
entbase = entbase.drop_duplicates()
alter=pd.read_csv('2alter.csv')
alter = alter.drop_duplicates()
branch=pd.read_csv('3branch.csv')
branch = branch.drop_duplicates()
invest=pd.read_csv('4invest.csv')
invest = invest.drop_duplicates()
right=pd.read_csv('5right.csv')
right = right.drop_duplicates()
project=pd.read_csv('6project.csv')
project = project.drop_duplicates()
lawsuit=pd.read_csv('7lawsuit.csv')
lawsuit = lawsuit.drop_duplicates()
breakfaith=pd.read_csv('8breakfaith.csv')
breakfaith = breakfaith.drop_duplicates()
recruit=pd.read_csv('9recruit.csv')
recruit = recruit.drop_duplicates()

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#这部分代码是用指数衰减的方法统计所有表的次数，用到exp（-x），x是时间间隔，还有就是统计这个数目在HY中的排序


import math
#更改表的统计
alter_event = alter[['EID','ALTDATE']]
alter_event['ALTDATE_y'] = alter_event['ALTDATE'].map(lambda x:int(x.split('-')[0]))
alter_event['ALTDATE_m'] = alter_event['ALTDATE'].map(lambda x:int(x.split('-')[1]))
alter_event['jiange'] = (2015 - alter_event['ALTDATE_y'])*12 + 8 - alter_event['ALTDATE_m']
alter_event['alter_event_weights'] = alter_event['jiange'].map(lambda x: math.exp(-0.083*x))
alter_all_event = alter_event.groupby(['EID']).agg('sum').reset_index()
alter_all_event = alter_all_event[['EID','alter_event_weights']]

entbase_hy = entbase[['EID','HY']]
alter_all_event = pd.merge(alter_all_event,entbase_hy,on='EID',how='left')
alter_all_event['alter_hy_paiming'] = alter_all_event['alter_event_weights'].groupby(alter_all_event['HY']).rank(ascending=False)
alter_all_event = alter_all_event.drop(['HY'],axis=1)

#分支的统计
branch_num = branch[['EID','B_REYEAR']]
branch_num['jiange'] = (2015 - branch_num['B_REYEAR'])
branch_num['branch_num_weights'] = branch_num['jiange'].map(lambda x: math.exp(-0.1*x))
branch_num = branch_num.groupby(['EID']).agg('sum').reset_index()
branch_num = branch_num[['EID','branch_num_weights']]

entbase_hy = entbase[['EID','HY']]
branch_num = pd.merge(branch_num,entbase_hy,on='EID',how='left')
branch_num['branch_num_hy_paiming'] = branch_num['branch_num_weights'].groupby(branch_num['HY']).rank(ascending=False)
branch_num = branch_num.drop(['HY'],axis=1)

#投资的统计
invest_in_num = invest[['EID','BTYEAR']]
invest_in_num['jiange'] = (2015 - invest_in_num['BTYEAR'])
invest_in_num['invest_in_num_weights'] = invest_in_num['jiange'].map(lambda x: math.exp(-0.1*x))
invest_in_num = invest_in_num.groupby(['EID']).agg('sum').reset_index()
invest_in_num = invest_in_num[['EID','invest_in_num_weights']]

entbase_hy = entbase[['EID','HY']]
invest_in_num = pd.merge(invest_in_num,entbase_hy,on='EID',how='left')
invest_in_num['invest_in_num_hy_paiming'] = invest_in_num['invest_in_num_weights'].groupby(invest_in_num['HY']).rank(ascending=False)
invest_in_num = invest_in_num.drop(['HY'],axis=1)

#权利的统计
right_shengqing = right[['EID','ASKDATE']]
right_shengqing['ASKDATE_y'] = right_shengqing['ASKDATE'].map(lambda x:int(x.split('-')[0]))
right_shengqing['ASKDATE_m'] = right_shengqing['ASKDATE'].map(lambda x:int(x.split('-')[1]))
right_shengqing['jiange'] = (2015 - right_shengqing['ASKDATE_y'])*12 + 8 - right_shengqing['ASKDATE_m']
right_shengqing['right_shengqing_weights'] = right_shengqing['jiange'].map(lambda x: math.exp(-0.083*x))
right_shengqing = right_shengqing.groupby(['EID']).agg('sum').reset_index()
right_shengqing = right_shengqing[['EID','right_shengqing_weights']]

entbase_hy = entbase[['EID','HY']]
right_shengqing = pd.merge(right_shengqing,entbase_hy,on='EID',how='left')
right_shengqing['right_shengqing_hy_paiming'] = right_shengqing['right_shengqing_weights'].groupby(right_shengqing['HY']).rank(ascending=False)
right_shengqing = right_shengqing.drop(['HY'],axis=1)

#项目的统计
project_num = project[['EID','DJDATE']]
project_num['DJDATE_y'] = project_num['DJDATE'].map(lambda x:int(x.split('-')[0]))
project_num['DJDATE_m'] = project_num['DJDATE'].map(lambda x:int(x.split('-')[1]))
project_num['jiange'] = (2015 - project_num['DJDATE_y'])*12 + 8 - project_num['DJDATE_m']
project_num['project_num_weights'] = project_num['jiange'].map(lambda x: math.exp(-0.083*x))
project_num = project_num.groupby(['EID']).agg('sum').reset_index()
project_num = project_num[['EID','project_num_weights']]

entbase_hy = entbase[['EID','HY']]
project_num = pd.merge(project_num,entbase_hy,on='EID',how='left')
project_num['project_num_hy_paiming'] = project_num['project_num_weights'].groupby(project_num['HY']).rank(ascending=False)
project_num = project_num.drop(['HY','ETYPE'],axis=1)

#案件的统计
lawsuit_num = lawsuit[['EID','LAWDATE']]
lawsuit_num['LAWDATE_y'] = lawsuit_num['LAWDATE'].map(lambda x:int(x.split('-')[0]))
lawsuit_num['LAWDATE_m'] = lawsuit_num['LAWDATE'].map(lambda x:int(x.split('-')[1]))
lawsuit_num['jiange'] = (2015 - lawsuit_num['LAWDATE_y'])*12 + 8 - lawsuit_num['LAWDATE_m']
lawsuit_num['lawsuit_num_weights'] = lawsuit_num['jiange'].map(lambda x: math.exp(-0.083*x))
lawsuit_num = lawsuit_num.groupby(['EID']).agg('sum').reset_index()
lawsuit_num = lawsuit_num[['EID','lawsuit_num_weights']]

entbase_hy = entbase[['EID','HY']]
lawsuit_num = pd.merge(lawsuit_num,entbase_hy,on='EID',how='left')
lawsuit_num['lawsuit_num_hy_paiming'] = lawsuit_num['lawsuit_num_weights'].groupby(lawsuit_num['HY']).rank(ascending=False)
lawsuit_num = lawsuit_num.drop(['HY'],axis=1)

#失信的统计
breakfaith_num = breakfaith[['EID','FBDATE']]
breakfaith_num['FBDATE_y'] = breakfaith_num['FBDATE'].map(lambda x:int(x.split('/')[0]))
breakfaith_num['FBDATE_m'] = breakfaith_num['FBDATE'].map(lambda x:int(x.split('/')[1]))
breakfaith_num['jiange'] = (2015 - breakfaith_num['FBDATE_y'])*12 + 8 - breakfaith_num['FBDATE_m']
breakfaith_num['breakfaith_num_weights'] = breakfaith_num['jiange'].map(lambda x: math.exp(-0.083*x))
breakfaith_num = breakfaith_num.groupby(['EID']).agg('sum').reset_index()
breakfaith_num = breakfaith_num[['EID','breakfaith_num_weights']]

entbase_hy = entbase[['EID','HY']]
breakfaith_num = pd.merge(breakfaith_num,entbase_hy,on='EID',how='left')
breakfaith_num['breakfaith_num_hy_paiming'] = breakfaith_num['breakfaith_num_weights'].groupby(breakfaith_num['HY']).rank(ascending=False)
breakfaith_num = breakfaith_num.drop(['HY'],axis=1)

#招聘信息的统计
recruit_num = recruit[['EID','RECDATE']]
recruit_num['RECDATE_y'] = recruit_num['RECDATE'].map(lambda x:int(x.split('-')[0]))
recruit_num['RECDATE_m'] = recruit_num['RECDATE'].map(lambda x:int(x.split('-')[1]))
recruit_num['jiange'] = (2015 - recruit_num['RECDATE_y'])*12 + 8 - recruit_num['RECDATE_m']
recruit_num['recruit_num_weights'] = recruit_num['jiange'].map(lambda x: math.exp(-0.083*x))
recruit_num = recruit_num.groupby(['EID']).agg('sum').reset_index()
recruit_num = recruit_num[['EID','recruit_num_weights']]

entbase_hy = entbase[['EID','HY']]
recruit_num = pd.merge(recruit_num,entbase_hy,on='EID',how='left')
recruit_num['recruit_num_hy_paiming'] = recruit_num['recruit_num_weights'].groupby(recruit_num['HY']).rank(ascending=False)
recruit_num = recruit_num.drop(['HY'],axis=1)

#权利各种类型的统计
righttype_list=right['RIGHTTYPE'].drop_duplicates().tolist()
righttype_dummies=pd.get_dummies(right.RIGHTTYPE,prefix='RIGHTTYPE')
right_jiange = right[['EID','ASKDATE']]
right_jiange['ASKDATE_y'] = right_jiange['ASKDATE'].map(lambda x:int(x.split('-')[0]))
right_jiange['ASKDATE_m'] = right_jiange['ASKDATE'].map(lambda x:int(x.split('-')[1]))
right_jiange['jiange'] = (2015 - right_jiange['ASKDATE_y'])*12 + 8 - right_jiange['ASKDATE_m']
right_jiange = right_jiange[['jiange']]
righttype_dummies=pd.concat([righttype_dummies,right_jiange],axis=1)
righttype_dummies['RIGHTTYPE_11'] = righttype_dummies['RIGHTTYPE_11'] * righttype_dummies['jiange'].map(lambda x: math.exp(-0.083*x))
righttype_dummies['RIGHTTYPE_12'] = righttype_dummies['RIGHTTYPE_12'] * righttype_dummies['jiange'].map(lambda x: math.exp(-0.083*x))
righttype_dummies['RIGHTTYPE_20'] = righttype_dummies['RIGHTTYPE_20'] * righttype_dummies['jiange'].map(lambda x: math.exp(-0.083*x))
righttype_dummies['RIGHTTYPE_30'] = righttype_dummies['RIGHTTYPE_30'] * righttype_dummies['jiange'].map(lambda x: math.exp(-0.083*x))
righttype_dummies['RIGHTTYPE_40'] = righttype_dummies['RIGHTTYPE_40'] * righttype_dummies['jiange'].map(lambda x: math.exp(-0.083*x))
righttype_dummies['RIGHTTYPE_50'] = righttype_dummies['RIGHTTYPE_50'] * righttype_dummies['jiange'].map(lambda x: math.exp(-0.083*x))
righttype_dummies['RIGHTTYPE_60'] = righttype_dummies['RIGHTTYPE_60'] * righttype_dummies['jiange'].map(lambda x: math.exp(-0.083*x))
righttype_dummies = righttype_dummies.drop(['jiange'],axis=1)
righttype_dummies=pd.concat([right,righttype_dummies],axis=1)
righttype_dummies=righttype_dummies.drop(['RIGHTTYPE','TYPECODE','ASKDATE','FBDATE'],axis=1)
righttype_dummies=righttype_dummies.groupby('EID',as_index=False).sum()
righttype_dummies.columns = ['EID','r11','r12','r20','r30','r40','r50','r60']

#时间更改的统计
alterno=alter['ALTERNO'].drop_duplicates().tolist()
alterno_dummies=pd.get_dummies(alter.ALTERNO,prefix='ALTERNO')
alter_jiange = alter[['EID','ALTDATE']]
alter_jiange['ALTDATE_y'] = alter_jiange['ALTDATE'].map(lambda x:int(x.split('-')[0]))
alter_jiange['ALTDATE_m'] = alter_jiange['ALTDATE'].map(lambda x:int(x.split('-')[1]))
alter_jiange['jiange'] = (2015 - alter_jiange['ALTDATE_y'])*12 + 8 - alter_jiange['ALTDATE_m']
alter_jiange = alter_jiange[['jiange']]
alterno_dummies=pd.concat([alterno_dummies,alter_jiange],axis=1)
alterno_dummies['ALTERNO_01'] = alterno_dummies['ALTERNO_01'] * alterno_dummies['jiange'].map(lambda x: math.exp(-0.083*x))
alterno_dummies['ALTERNO_02'] = alterno_dummies['ALTERNO_02'] * alterno_dummies['jiange'].map(lambda x: math.exp(-0.083*x))
alterno_dummies['ALTERNO_03'] = alterno_dummies['ALTERNO_03'] * alterno_dummies['jiange'].map(lambda x: math.exp(-0.083*x))
alterno_dummies['ALTERNO_04'] = alterno_dummies['ALTERNO_04'] * alterno_dummies['jiange'].map(lambda x: math.exp(-0.083*x))
alterno_dummies['ALTERNO_05'] = alterno_dummies['ALTERNO_05'] * alterno_dummies['jiange'].map(lambda x: math.exp(-0.083*x))
alterno_dummies['ALTERNO_10'] = alterno_dummies['ALTERNO_10'] * alterno_dummies['jiange'].map(lambda x: math.exp(-0.083*x))
alterno_dummies['ALTERNO_12'] = alterno_dummies['ALTERNO_12'] * alterno_dummies['jiange'].map(lambda x: math.exp(-0.083*x))
alterno_dummies['ALTERNO_13'] = alterno_dummies['ALTERNO_13'] * alterno_dummies['jiange'].map(lambda x: math.exp(-0.083*x))
alterno_dummies['ALTERNO_14'] = alterno_dummies['ALTERNO_14'] * alterno_dummies['jiange'].map(lambda x: math.exp(-0.083*x))
alterno_dummies['ALTERNO_27'] = alterno_dummies['ALTERNO_27'] * alterno_dummies['jiange'].map(lambda x: math.exp(-0.083*x))
alterno_dummies['ALTERNO_99'] = alterno_dummies['ALTERNO_99'] * alterno_dummies['jiange'].map(lambda x: math.exp(-0.083*x))
alterno_dummies['ALTERNO_A_015'] = alterno_dummies['ALTERNO_A_015'] * alterno_dummies['jiange'].map(lambda x: math.exp(-0.083*x))
alterno_dummies = alterno_dummies.drop(['jiange'],axis=1)
alterno_dummies=pd.concat([alter,alterno_dummies],axis=1)
alterno_dummies=alterno_dummies.drop(['ALTERNO','ALTDATE','ALTBE','ALTAF'],axis=1)
alterno_dummies=alterno_dummies.groupby('EID',as_index=False).sum()
alterno_dummies.columns = ['EID','a1','a2','a3','a4','a5','a10','a12','a13','a14','a27','a99','a15']

#分支关闭的统计
branchend_num = branch[['EID','B_ENDYEAR']]
branchend_num = branchend_num[branchend_num['B_ENDYEAR'].notnull()]
branchend_num['jiange'] = (2015 - branchend_num['B_ENDYEAR'])
branchend_num['branchend_num_weights'] = branchend_num['jiange'].map(lambda x: math.exp(-0.1*x))
branchend_num = branchend_num.groupby(['EID']).agg('sum').reset_index()
branchend_num = branchend_num[['EID','branchend_num_weights']]

entbase_hy = entbase[['EID','HY']]
branchend_num = pd.merge(branchend_num,entbase_hy,on='EID',how='left')
branchend_num['branchend_num_hy_paiming'] = branchend_num['branchend_num_weights'].groupby(branchend_num['HY']).rank(ascending=False)
branchend_num = branchend_num.drop(['HY'],axis=1)

#投资关闭的统计
investend_num = invest[['EID','BTENDYEAR']]
investend_num = investend_num[investend_num['BTENDYEAR'].notnull()]
investend_num['jiange'] = (2015 - investend_num['BTENDYEAR'])
investend_num['investend_num_weights'] = investend_num['jiange'].map(lambda x: math.exp(-0.1*x))
investend_num = investend_num.groupby(['EID']).agg('sum').reset_index()
investend_num = investend_num[['EID','investend_num_weights']]

entbase_hy = entbase[['EID','HY']]
investend_num = pd.merge(investend_num,entbase_hy,on='EID',how='left')
investend_num['investend_num_hy_paiming'] = investend_num['investend_num_weights'].groupby(investend_num['HY']).rank(ascending=False)
investend_num = investend_num.drop(['HY'],axis=1)

#分支在省内的统计
branch_inhome = branch[branch['IFHOME']==1]
branch_inhome = branch_inhome[['EID','B_REYEAR']]
branch_inhome['jiange'] = (2015 - branch_inhome['B_REYEAR'])
branch_inhome['branch_inhome_weights'] = branch_inhome['jiange'].map(lambda x: math.exp(-0.1*x))
branch_inhome = branch_inhome.groupby(['EID']).agg('sum').reset_index()
branch_inhome = branch_inhome[['EID','branch_inhome_weights']]

entbase_hy = entbase[['EID','HY']]
branch_inhome = pd.merge(branch_inhome,entbase_hy,on='EID',how='left')
branch_inhome['branch_inhome_hy_paiming'] = branch_inhome['branch_inhome_weights'].groupby(branch_inhome['HY']).rank(ascending=False)
branch_inhome = branch_inhome.drop(['HY'],axis=1)

#分支在省外的统计
branch_nohome = branch[branch['IFHOME']==0]
branch_nohome = branch_nohome[['EID','B_REYEAR']]
branch_nohome['jiange'] = (2015 - branch_nohome['B_REYEAR'])
branch_nohome['branch_nohome_weights'] = branch_nohome['jiange'].map(lambda x: math.exp(-0.1*x))
branch_nohome = branch_nohome.groupby(['EID']).agg('sum').reset_index()
branch_nohome = branch_nohome[['EID','branch_nohome_weights']]

entbase_hy = entbase[['EID','HY']]
branch_nohome = pd.merge(branch_nohome,entbase_hy,on='EID',how='left')
branch_nohome['branch_nohome_hy_paiming'] = branch_nohome['branch_nohome_weights'].groupby(branch_nohome['HY']).rank(ascending=False)
branch_nohome = branch_nohome.drop(['HY'],axis=1)

#项目在省内的统计
project_inhome = project[project['IFHOME']==1]
project_inhome = project_inhome[['EID','DJDATE']]
project_inhome['DJDATE_y'] = project_inhome['DJDATE'].map(lambda x:int(x.split('-')[0]))
project_inhome['DJDATE_m'] = project_inhome['DJDATE'].map(lambda x:int(x.split('-')[1]))
project_inhome['jiange'] = (2015 - project_inhome['DJDATE_y'])*12 + 8 - project_inhome['DJDATE_m']
project_inhome['project_inhome_weights'] = project_inhome['jiange'].map(lambda x: math.exp(-0.083*x))
project_inhome = project_inhome.groupby(['EID']).agg('sum').reset_index()
project_inhome = project_inhome[['EID','project_inhome_weights']]

entbase_hy = entbase[['EID','HY']]
project_inhome = pd.merge(project_inhome,entbase_hy,on='EID',how='left')
project_inhome['project_inhome_hy_paiming'] = project_inhome['project_inhome_weights'].groupby(project_inhome['HY']).rank(ascending=False)
project_inhome = project_inhome.drop(['HY'],axis=1)

#项目在省外的统计
project_nohome = project[project['IFHOME']==0]
project_nohome = project_nohome[['EID','DJDATE']]
project_nohome['DJDATE_y'] = project_nohome['DJDATE'].map(lambda x:int(x.split('-')[0]))
project_nohome['DJDATE_m'] = project_nohome['DJDATE'].map(lambda x:int(x.split('-')[1]))
project_nohome['jiange'] = (2015 - project_nohome['DJDATE_y'])*12 + 8 - project_nohome['DJDATE_m']
project_nohome['project_nohome_weights'] = project_nohome['jiange'].map(lambda x: math.exp(-0.083*x))
project_nohome = project_nohome.groupby(['EID']).agg('sum').reset_index()
project_nohome = project_nohome[['EID','project_nohome_weights']]

entbase_hy = entbase[['EID','HY']]
project_nohome = pd.merge(project_nohome,entbase_hy,on='EID',how='left')
project_nohome['project_nohome_hy_paiming'] = project_nohome['project_nohome_weights'].groupby(project_nohome['HY']).rank(ascending=False)
project_nohome = project_nohome.drop(['HY'],axis=1)

#投资在省内的统计
invest_inhome = invest[invest['IFHOME']==1]
invest_inhome = invest_inhome[['EID','BTYEAR']]
invest_inhome['jiange'] = (2015 - invest_inhome['BTYEAR'])
invest_inhome['invest_inhome_weights'] = invest_inhome['jiange'].map(lambda x: math.exp(-0.1*x))
invest_inhome = invest_inhome.groupby(['EID']).agg('sum').reset_index()
invest_inhome = invest_inhome[['EID','invest_inhome_weights']]

entbase_hy = entbase[['EID','HY']]
invest_inhome = pd.merge(invest_inhome,entbase_hy,on='EID',how='left')
invest_inhome['invest_inhome_hy_paiming'] = invest_inhome['invest_inhome_weights'].groupby(invest_inhome['HY']).rank(ascending=False)
invest_inhome = invest_inhome.drop(['HY'],axis=1)

#投资在省外的统计
invest_nohome = invest[invest['IFHOME']==0]
invest_nohome = invest_nohome[['EID','BTYEAR']]
invest_nohome['jiange'] = (2015 - invest_nohome['BTYEAR'])
invest_nohome['invest_nohome_weights'] = invest_nohome['jiange'].map(lambda x: math.exp(-0.1*x))
invest_nohome = invest_nohome.groupby(['EID']).agg('sum').reset_index()
invest_nohome = invest_nohome[['EID','invest_nohome_weights']]

entbase_hy = entbase[['EID','HY']]
invest_nohome = pd.merge(invest_nohome,entbase_hy,on='EID',how='left')
invest_nohome['invest_nohome_hy_paiming'] = invest_nohome['invest_nohome_weights'].groupby(invest_nohome['HY']).rank(ascending=False)
invest_nohome = invest_nohome.drop(['HY'],axis=1)


train=pd.merge(train,alter_all_event,how='left',on='EID')
train=pd.merge(train,branch_num,how='left',on='EID')
train=pd.merge(train,invest_in_num,how='left',on='EID')
train=pd.merge(train,right_shengqing,how='left',on='EID')
train=pd.merge(train,project_num,how='left',on='EID')
train=pd.merge(train,lawsuit_num,how='left',on='EID')
train=pd.merge(train,breakfaith_num,how='left',on='EID')
train=pd.merge(train,recruit_num,how='left',on='EID')
train=pd.merge(train,righttype_dummies,how='left',on='EID')
train=pd.merge(train,alterno_dummies,how='left',on='EID')
train=pd.merge(train,branchend_num,how='left',on='EID')
train=pd.merge(train,investend_num,how='left',on='EID')
train=pd.merge(train,branch_inhome,how='left',on='EID')
train=pd.merge(train,branch_nohome,how='left',on='EID')
train=pd.merge(train,project_inhome,how='left',on='EID')
train=pd.merge(train,project_nohome,how='left',on='EID')
train=pd.merge(train,invest_inhome,how='left',on='EID')
train=pd.merge(train,invest_nohome,how='left',on='EID')

predict=pd.merge(predict,alter_all_event,how='left',on='EID')
predict=pd.merge(predict,branch_num,how='left',on='EID')
predict=pd.merge(predict,invest_in_num,how='left',on='EID')
predict=pd.merge(predict,right_shengqing,how='left',on='EID')
predict=pd.merge(predict,project_num,how='left',on='EID')
predict=pd.merge(predict,lawsuit_num,how='left',on='EID')
predict=pd.merge(predict,breakfaith_num,how='left',on='EID')
predict=pd.merge(predict,recruit_num,how='left',on='EID')
predict=pd.merge(predict,righttype_dummies,how='left',on='EID')
predict=pd.merge(predict,alterno_dummies,how='left',on='EID')
predict=pd.merge(predict,branchend_num,how='left',on='EID')
predict=pd.merge(predict,investend_num,how='left',on='EID')
predict=pd.merge(predict,branch_inhome,how='left',on='EID')
predict=pd.merge(predict,branch_nohome,how='left',on='EID')
predict=pd.merge(predict,project_inhome,how='left',on='EID')
predict=pd.merge(predict,project_nohome,how='left',on='EID')
predict=pd.merge(predict,invest_inhome,how='left',on='EID')
predict=pd.merge(predict,invest_nohome,how='left',on='EID')


#-----------------------------------------------------------------------------------------------------------------------------------------------------
#这部分代码是针对表2更改表中5和27类别的处理

alter=pd.read_csv('2alter.csv')
alter = alter.drop_duplicates()
alter_event = alter[['EID','ALTERNO','ALTBE','ALTAF','ALTDATE']]

alter_event['ALTBE'] = alter_event['ALTBE'].replace('null万元',0)
alter_event = alter_event.fillna(0)

alter_event['ALTBE'] = alter_event['ALTBE'].map(lambda x :str(x)+'*')
alter_event['ALTBE'] = alter_event['ALTBE'].map(lambda x :str(x).replace('万','0000*'))
alter_event['ALTBE'] = alter_event['ALTBE'].map(lambda x : float(x.split('*')[0]))

alter_event['ALTAF'] = alter_event['ALTAF'].map(lambda x :str(x)+'*')
alter_event['ALTAF'] = alter_event['ALTAF'].map(lambda x :str(x).replace('万','0000*'))
alter_event['ALTAF'] = alter_event['ALTAF'].map(lambda x : float(x.split('*')[0]))

alter_event_05 = alter_event[alter_event['ALTERNO'] == '05']
alter_event_05['ALT_cha_05'] = alter_event_05['ALTAF'] - alter_event_05['ALTBE']
alter_event_05 = alter_event_05.groupby('EID',as_index=False)['ALT_cha_05'].agg({'ALT_cha_05sum':'sum','ALT_max_05':'max','ALT_min_05':'min','ALT_median_05':'median'})
alter_event_05 = alter_event_05[['EID','ALT_cha_05sum','ALT_max_05','ALT_min_05','ALT_median_05']]
alter_event_05 = alter_event_05.replace(0,np.nan)

alter_event_27 = alter_event[alter_event['ALTERNO'] == '27']
alter_event_27['ALT_cha_27'] = alter_event_27['ALTAF'] - alter_event_27['ALTBE']
alter_event_27 = alter_event_27.groupby('EID',as_index=False)['ALT_cha_27'].agg({'ALT_cha_27sum':'sum','ALT_max_27':'max','ALT_min_27':'min','ALT_median_27':'median'})
alter_event_27 = alter_event_27[['EID','ALT_cha_27sum','ALT_max_27','ALT_min_27','ALT_median_27']]
alter_event_27 = alter_event_27.replace(0,np.nan)

train=pd.merge(train,alter_event_05,how='left',on='EID')
train=pd.merge(train,alter_event_27,how='left',on='EID')

predict=pd.merge(predict,alter_event_05,how='left',on='EID')
predict=pd.merge(predict,alter_event_27,how='left',on='EID')


#--------------------------------------------------------------------------------------------------------------
#对投资其他企业中位数的统计
invest=pd.read_csv('4invest.csv')
invest = invest.drop_duplicates()
invest_num_max_sum_median=invest.groupby('EID',as_index=False)['BTBL'].agg({'btbl_median':'median'})

train=pd.merge(train,invest_num_max_sum_median,how='left',on='EID')
predict=pd.merge(predict,invest_num_max_sum_median,how='left',on='EID')

train=train.fillna(0)
predict=predict.fillna(0)


#---------------------------------------------------------------------------------------------------------------------
#这部分代码是求各张和2015年的最小时间间隔

entbase=pd.read_csv('1entbase.csv')
entbase = entbase.drop_duplicates()
alter=pd.read_csv('2alter.csv')
alter = alter.drop_duplicates()
branch=pd.read_csv('3branch.csv')
branch = branch.drop_duplicates()
invest=pd.read_csv('4invest.csv')
invest = invest.drop_duplicates()
right=pd.read_csv('5right.csv')
right = right.drop_duplicates()
project=pd.read_csv('6project.csv')
project = project.drop_duplicates()
lawsuit=pd.read_csv('7lawsuit.csv')
lawsuit = lawsuit.drop_duplicates()
breakfaith=pd.read_csv('8breakfaith.csv')
breakfaith = breakfaith.drop_duplicates()
recruit=pd.read_csv('9recruit.csv')
recruit = recruit.drop_duplicates()

#项目的最小时间间隔
project_jiange = project[['EID','DJDATE']]
project_jiange['DJDATE_y'] = project_jiange['DJDATE'].map(lambda x:int(x.split('-')[0]))
project_jiange['DJDATE_m'] = project_jiange['DJDATE'].map(lambda x:int(x.split('-')[1]))
project_jiange['jiange'] = (2015 - project_jiange['DJDATE_y'])*12 + 8 - project_jiange['DJDATE_m']
project_jiange_all = project_jiange.groupby('EID',as_index=False)['jiange'].agg({'project_jiange_min':'min'})
#更改的最小时间间隔
alter_jiange = alter[['EID','ALTDATE']]
alter_jiange['ALTDATE_y'] = alter_jiange['ALTDATE'].map(lambda x:int(x.split('-')[0]))
alter_jiange['ALTDATE_m'] = alter_jiange['ALTDATE'].map(lambda x:int(x.split('-')[1]))
alter_jiange['jiange'] = (2015 - alter_jiange['ALTDATE_y'])*12 + 8 - alter_jiange['ALTDATE_m']
alter_jiange_all = alter_jiange.groupby('EID',as_index=False)['jiange'].agg({'alter_jiange_min':'min'})
#分支的最小时间间隔
branch_jiange = branch[['EID','B_REYEAR']]
branch_jiange['jiange'] = (2015 - branch_jiange['B_REYEAR'])
branch_jiange_all = branch_jiange.groupby('EID',as_index=False)['jiange'].agg({'branch_jiange_min':'min'})
#分支关闭的最小时间间隔
branch_jiange_c = branch[['EID','B_ENDYEAR']]
branch_jiange_c['jiange'] = (2015 - branch_jiange_c['B_ENDYEAR'])
branch_jiange_c_all = branch_jiange_c.groupby('EID',as_index=False)['jiange'].agg({'branch_jiange_c_min':'min'})
#投资的最小时间间隔
invest_in_jiange = invest[['EID','BTYEAR']]
invest_in_jiange['jiange'] = (2015 - invest_in_jiange['BTYEAR'])
invest_in_jiange_all = invest_in_jiange.groupby('EID',as_index=False)['jiange'].agg({'invest_in_jiange_min':'min'})
#投资关闭的最小时间间隔
invest_c_jiange = invest[['EID','BTENDYEAR']]
invest_c_jiange['jiange'] = (2015 - invest_c_jiange['BTENDYEAR'])
invest_c_jiange_all = invest_c_jiange.groupby('EID',as_index=False)['jiange'].agg({'invest_c_jiange_min':'min'})
#权利申请的最小时间间隔
right_shengqing_jiange = right[['EID','ASKDATE']]
right_shengqing_jiange['ASKDATE_y'] = right_shengqing_jiange['ASKDATE'].map(lambda x:int(x.split('-')[0]))
right_shengqing_jiange['ASKDATE_m'] = right_shengqing_jiange['ASKDATE'].map(lambda x:int(x.split('-')[1]))
right_shengqing_jiange['jiange'] = (2015 - right_shengqing_jiange['ASKDATE_y'])*12 + 8 - right_shengqing_jiange['ASKDATE_m']
right_shengqing_jiange_all = right_shengqing_jiange.groupby('EID',as_index=False)['jiange'].agg({'right_shengqing_jiange_min':'min'})
#权利赋予的最小时间间隔
right_fy_jiange = right[['EID','FBDATE']]
right_fy_jiange = right_fy_jiange[right_fy_jiange['FBDATE'].notnull()]
right_fy_jiange['FBDATE_y'] = right_fy_jiange['FBDATE'].map(lambda x:int(x.split('-')[0]))
right_fy_jiange['FBDATE_m'] = right_fy_jiange['FBDATE'].map(lambda x:int(x.split('-')[1]))
right_fy_jiange['jiange'] = (2015 - right_fy_jiange['FBDATE_y'])*12 + 8 - right_fy_jiange['FBDATE_m']
right_fy_jiange_all = right_fy_jiange.groupby('EID',as_index=False)['jiange'].agg({'right_fy_jiange_min':'min'})
#案件的最小时间间隔
lawsuit_jiange = lawsuit[['EID','LAWDATE']]
lawsuit_jiange['LAWDATE_y'] = lawsuit_jiange['LAWDATE'].map(lambda x:int(x.split('-')[0]))
lawsuit_jiange['LAWDATE_m'] = lawsuit_jiange['LAWDATE'].map(lambda x:int(x.split('-')[1]))
lawsuit_jiange['jiange'] = (2015 - lawsuit_jiange['LAWDATE_y'])*12 + 8 - lawsuit_jiange['LAWDATE_m']
lawsuit_jiange_all = lawsuit_jiange.groupby('EID',as_index=False)['jiange'].agg({'lawsuit_jiange_min':'min'})
#失信的最小时间间隔
breakfaith_jiange = breakfaith[['EID','FBDATE']]
breakfaith_jiange['FBDATE_y'] = breakfaith_jiange['FBDATE'].map(lambda x:int(x.split('/')[0]))
breakfaith_jiange['FBDATE_m'] = breakfaith_jiange['FBDATE'].map(lambda x:int(x.split('/')[1]))
breakfaith_jiange['jiange'] = (2015 - breakfaith_jiange['FBDATE_y'])*12 + 8 - breakfaith_jiange['FBDATE_m']
breakfaith_jiange_all = breakfaith_jiange.groupby('EID',as_index=False)['jiange'].agg({'breakfaith_jiange_min':'min'})
#失信结束的最小时间间隔
breakfaith_o_jiange = breakfaith[['EID','SXENDDATE']]
breakfaith_o_jiange = breakfaith_o_jiange[breakfaith_o_jiange['SXENDDATE'].notnull()]
breakfaith_o_jiange['SXENDDATE_y'] = breakfaith_o_jiange['SXENDDATE'].map(lambda x:int(x.split('/')[0]))
breakfaith_o_jiange['SXENDDATE_m'] = breakfaith_o_jiange['SXENDDATE'].map(lambda x:int(x.split('/')[1]))
breakfaith_o_jiange['jiange'] = (2015 - breakfaith_o_jiange['SXENDDATE_y'])*12 + 8 - breakfaith_o_jiange['SXENDDATE_m']
breakfaith_o_jiange = breakfaith_o_jiange.groupby('EID',as_index=False)['jiange'].agg({'breakfaith_o_jiange_min':'min'})
#招聘的最小时间间隔
recruit_jiange = recruit[['EID','RECDATE']]
recruit_jiange['RECDATE_y'] = recruit_jiange['RECDATE'].map(lambda x:int(x.split('-')[0]))
recruit_jiange['RECDATE_m'] = recruit_jiange['RECDATE'].map(lambda x:int(x.split('-')[1]))
recruit_jiange['jiange'] = (2015 - recruit_jiange['RECDATE_y'])*12 + 8 - recruit_jiange['RECDATE_m']
recruit_jiange_all = recruit_jiange.groupby('EID',as_index=False)['jiange'].agg({'recruit_jiange_min':'min'})


train=pd.merge(train,project_jiange_all,how='left',on='EID')
train=pd.merge(train,alter_jiange_all,how='left',on='EID')
train=pd.merge(train,branch_jiange_all,how='left',on='EID')
train=pd.merge(train,invest_in_jiange_all,how='left',on='EID')
train=pd.merge(train,right_shengqing_jiange_all,how='left',on='EID')
train=pd.merge(train,lawsuit_jiange_all,how='left',on='EID')
train=pd.merge(train,breakfaith_jiange_all,how='left',on='EID')
train=pd.merge(train,recruit_jiange_all,how='left',on='EID')
train=pd.merge(train,branch_jiange_c_all,how='left',on='EID')
train=pd.merge(train,invest_c_jiange_all,how='left',on='EID')
train=pd.merge(train,right_fy_jiange_all,how='left',on='EID')
train=pd.merge(train,breakfaith_o_jiange,how='left',on='EID')

predict=pd.merge(predict,project_jiange_all,how='left',on='EID')
predict=pd.merge(predict,alter_jiange_all,how='left',on='EID')
predict=pd.merge(predict,branch_jiange_all,how='left',on='EID')
predict=pd.merge(predict,invest_in_jiange_all,how='left',on='EID')
predict=pd.merge(predict,right_shengqing_jiange_all,how='left',on='EID')
predict=pd.merge(predict,lawsuit_jiange_all,how='left',on='EID')
predict=pd.merge(predict,breakfaith_jiange_all,how='left',on='EID')
predict=pd.merge(predict,recruit_jiange_all,how='left',on='EID')
predict=pd.merge(predict,branch_jiange_c_all,how='left',on='EID')
predict=pd.merge(predict,invest_c_jiange_all,how='left',on='EID')
predict=pd.merge(predict,right_fy_jiange_all,how='left',on='EID')
predict=pd.merge(predict,breakfaith_o_jiange,how='left',on='EID')

#----------------------------------------------------------------------------------------------------------------------------------------
#这部分是针对表2，更改表的各种类型进行统计最小时间间隔
alter=pd.read_csv('2alter.csv')
alter = alter.drop_duplicates()
alter_jiange01 = alter[alter['ALTERNO'] == '01']
alter_jiange01 = alter_jiange01[['EID','ALTDATE']]
alter_jiange01['ALTDATE_y'] = alter_jiange01['ALTDATE'].map(lambda x:int(x.split('-')[0]))
alter_jiange01['ALTDATE_m'] = alter_jiange01['ALTDATE'].map(lambda x:int(x.split('-')[1]))
alter_jiange01['jiange'] = (2015 - alter_jiange01['ALTDATE_y'])*12 + 8 - alter_jiange01['ALTDATE_m']
alter_jiange01 = alter_jiange01.groupby('EID',as_index=False)['jiange'].agg({'alter_jiange01_min':'min'})

alter_jiange02 = alter[alter['ALTERNO'] == '02']
alter_jiange02 = alter_jiange02[['EID','ALTDATE']]
alter_jiange02['ALTDATE_y'] = alter_jiange02['ALTDATE'].map(lambda x:int(x.split('-')[0]))
alter_jiange02['ALTDATE_m'] = alter_jiange02['ALTDATE'].map(lambda x:int(x.split('-')[1]))
alter_jiange02['jiange'] = (2015 - alter_jiange02['ALTDATE_y'])*12 + 8 - alter_jiange02['ALTDATE_m']
alter_jiange02 = alter_jiange02.groupby('EID',as_index=False)['jiange'].agg({'alter_jiange02_min':'min'})

alter_jiange03 = alter[alter['ALTERNO'] == '03']
alter_jiange03 = alter_jiange03[['EID','ALTDATE']]
alter_jiange03['ALTDATE_y'] = alter_jiange03['ALTDATE'].map(lambda x:int(x.split('-')[0]))
alter_jiange03['ALTDATE_m'] = alter_jiange03['ALTDATE'].map(lambda x:int(x.split('-')[1]))
alter_jiange03['jiange'] = (2015 - alter_jiange03['ALTDATE_y'])*12 + 8 - alter_jiange03['ALTDATE_m']
alter_jiange03 = alter_jiange03.groupby('EID',as_index=False)['jiange'].agg({'alter_jiange03_min':'min'})

alter_jiange04 = alter[alter['ALTERNO'] == '04']
alter_jiange04 = alter_jiange04[['EID','ALTDATE']]
alter_jiange04['ALTDATE_y'] = alter_jiange04['ALTDATE'].map(lambda x:int(x.split('-')[0]))
alter_jiange04['ALTDATE_m'] = alter_jiange04['ALTDATE'].map(lambda x:int(x.split('-')[1]))
alter_jiange04['jiange'] = (2015 - alter_jiange04['ALTDATE_y'])*12 + 8 - alter_jiange04['ALTDATE_m']
alter_jiange04 = alter_jiange04.groupby('EID',as_index=False)['jiange'].agg({'alter_jiange04_min':'min'})

alter_jiange05 = alter[alter['ALTERNO'] == '05']
alter_jiange05 = alter_jiange05[['EID','ALTDATE']]
alter_jiange05['ALTDATE_y'] = alter_jiange05['ALTDATE'].map(lambda x:int(x.split('-')[0]))
alter_jiange05['ALTDATE_m'] = alter_jiange05['ALTDATE'].map(lambda x:int(x.split('-')[1]))
alter_jiange05['jiange'] = (2015 - alter_jiange05['ALTDATE_y'])*12 + 8 - alter_jiange05['ALTDATE_m']
alter_jiange05 = alter_jiange05.groupby('EID',as_index=False)['jiange'].agg({'alter_jiange05_min':'min'})

alter_jiange10 = alter[alter['ALTERNO'] == '10']
alter_jiange10 = alter_jiange10[['EID','ALTDATE']]
alter_jiange10['ALTDATE_y'] = alter_jiange10['ALTDATE'].map(lambda x:int(x.split('-')[0]))
alter_jiange10['ALTDATE_m'] = alter_jiange10['ALTDATE'].map(lambda x:int(x.split('-')[1]))
alter_jiange10['jiange'] = (2015 - alter_jiange10['ALTDATE_y'])*12 + 8 - alter_jiange10['ALTDATE_m']
alter_jiange10 = alter_jiange10.groupby('EID',as_index=False)['jiange'].agg({'alter_jiange10_min':'min'})

alter_jiange12 = alter[alter['ALTERNO'] == '12']
alter_jiange12 = alter_jiange12[['EID','ALTDATE']]
alter_jiange12['ALTDATE_y'] = alter_jiange12['ALTDATE'].map(lambda x:int(x.split('-')[0]))
alter_jiange12['ALTDATE_m'] = alter_jiange12['ALTDATE'].map(lambda x:int(x.split('-')[1]))
alter_jiange12['jiange'] = (2015 - alter_jiange12['ALTDATE_y'])*12 + 8 - alter_jiange12['ALTDATE_m']
alter_jiange12 = alter_jiange12.groupby('EID',as_index=False)['jiange'].agg({'alter_jiange12_min':'min'})

alter_jiange13 = alter[alter['ALTERNO'] == '13']
alter_jiange13 = alter_jiange13[['EID','ALTDATE']]
alter_jiange13['ALTDATE_y'] = alter_jiange13['ALTDATE'].map(lambda x:int(x.split('-')[0]))
alter_jiange13['ALTDATE_m'] = alter_jiange13['ALTDATE'].map(lambda x:int(x.split('-')[1]))
alter_jiange13['jiange'] = (2015 - alter_jiange13['ALTDATE_y'])*12 + 8 - alter_jiange13['ALTDATE_m']
alter_jiange13 = alter_jiange13.groupby('EID',as_index=False)['jiange'].agg({'alter_jiange13_min':'min'})

alter_jiange14 = alter[alter['ALTERNO'] == '14']
alter_jiange14 = alter_jiange14[['EID','ALTDATE']]
alter_jiange14['ALTDATE_y'] = alter_jiange14['ALTDATE'].map(lambda x:int(x.split('-')[0]))
alter_jiange14['ALTDATE_m'] = alter_jiange14['ALTDATE'].map(lambda x:int(x.split('-')[1]))
alter_jiange14['jiange'] = (2015 - alter_jiange14['ALTDATE_y'])*12 + 8 - alter_jiange14['ALTDATE_m']
alter_jiange14 = alter_jiange14.groupby('EID',as_index=False)['jiange'].agg({'alter_jiange14_min':'min'})

alter_jiange27 = alter[alter['ALTERNO'] == '27']
alter_jiange27 = alter_jiange27[['EID','ALTDATE']]
alter_jiange27['ALTDATE_y'] = alter_jiange27['ALTDATE'].map(lambda x:int(x.split('-')[0]))
alter_jiange27['ALTDATE_m'] = alter_jiange27['ALTDATE'].map(lambda x:int(x.split('-')[1]))
alter_jiange27['jiange'] = (2015 - alter_jiange27['ALTDATE_y'])*12 + 8 - alter_jiange27['ALTDATE_m']
alter_jiange27 = alter_jiange27.groupby('EID',as_index=False)['jiange'].agg({'alter_jiange27_min':'min'})

alter_jiange99 = alter[alter['ALTERNO'] == '99']
alter_jiange99 = alter_jiange99[['EID','ALTDATE']]
alter_jiange99['ALTDATE_y'] = alter_jiange99['ALTDATE'].map(lambda x:int(x.split('-')[0]))
alter_jiange99['ALTDATE_m'] = alter_jiange99['ALTDATE'].map(lambda x:int(x.split('-')[1]))
alter_jiange99['jiange'] = (2015 - alter_jiange99['ALTDATE_y'])*12 + 8 - alter_jiange99['ALTDATE_m']
alter_jiange99 = alter_jiange99.groupby('EID',as_index=False)['jiange'].agg({'alter_jiange99_min':'min'})

alter_jiangeA_015 = alter[alter['ALTERNO'] == 'A_015']
alter_jiangeA_015 = alter_jiangeA_015[['EID','ALTDATE']]
alter_jiangeA_015['ALTDATE_y'] = alter_jiangeA_015['ALTDATE'].map(lambda x:int(x.split('-')[0]))
alter_jiangeA_015['ALTDATE_m'] = alter_jiangeA_015['ALTDATE'].map(lambda x:int(x.split('-')[1]))
alter_jiangeA_015['jiange'] = (2015 - alter_jiangeA_015['ALTDATE_y'])*12 + 8 - alter_jiangeA_015['ALTDATE_m']
alter_jiangeA_015 = alter_jiangeA_015.groupby('EID',as_index=False)['jiange'].agg({'alter_jiangeA_015_min':'min'})

train=pd.merge(train,alter_jiange01,how='left',on='EID')
train=pd.merge(train,alter_jiange02,how='left',on='EID')
train=pd.merge(train,alter_jiange03,how='left',on='EID')
train=pd.merge(train,alter_jiange04,how='left',on='EID')
train=pd.merge(train,alter_jiange05,how='left',on='EID')
train=pd.merge(train,alter_jiange10,how='left',on='EID')
train=pd.merge(train,alter_jiange12,how='left',on='EID')
train=pd.merge(train,alter_jiange13,how='left',on='EID')
train=pd.merge(train,alter_jiange14,how='left',on='EID')
train=pd.merge(train,alter_jiange27,how='left',on='EID')
train=pd.merge(train,alter_jiange99,how='left',on='EID')
train=pd.merge(train,alter_jiangeA_015,how='left',on='EID')

predict=pd.merge(predict,alter_jiange01,how='left',on='EID')
predict=pd.merge(predict,alter_jiange02,how='left',on='EID')
predict=pd.merge(predict,alter_jiange03,how='left',on='EID')
predict=pd.merge(predict,alter_jiange04,how='left',on='EID')
predict=pd.merge(predict,alter_jiange05,how='left',on='EID')
predict=pd.merge(predict,alter_jiange10,how='left',on='EID')
predict=pd.merge(predict,alter_jiange12,how='left',on='EID')
predict=pd.merge(predict,alter_jiange13,how='left',on='EID')
predict=pd.merge(predict,alter_jiange14,how='left',on='EID')
predict=pd.merge(predict,alter_jiange27,how='left',on='EID')
predict=pd.merge(predict,alter_jiange99,how='left',on='EID')
predict=pd.merge(predict,alter_jiangeA_015,how='left',on='EID')


#----------------------------------------------------------------------------------------------------------------------------------------
#这部分是交叉特征

#将年分成几个时间段
train_1lenbase = pd.read_csv("1entbase.csv")
train_1lenbase_1 = train_1lenbase[train_1lenbase['RGYEAR'] < 2001]
train_1lenbase_1['year_type'] = 0
train_1lenbase_2 = train_1lenbase[ (2005 > train_1lenbase['RGYEAR']) & (train_1lenbase['RGYEAR']>= 2001 ) ]
train_1lenbase_2['year_type'] = 1
train_1lenbase_3 = train_1lenbase[ (2010 >= train_1lenbase['RGYEAR']) & (train_1lenbase['RGYEAR']>= 2005 ) ]
train_1lenbase_3['year_type'] = 2
train_1lenbase_4 = train_1lenbase[ (2015 >= train_1lenbase['RGYEAR']) & (train_1lenbase['RGYEAR']>= 2011 ) ]
train_1lenbase_4['year_type'] = 3
train_1lenbase = pd.concat([train_1lenbase_1,train_1lenbase_2,train_1lenbase_3,train_1lenbase_4],axis=0)

#统计注册资本在同一类型、同一行业、同一时间段中的占比
zczb_eda = train_1lenbase[['ZCZB','year_type','HY','ETYPE']]
zczb_eda = zczb_eda.groupby(['year_type','HY','ETYPE']).agg('mean').reset_index()
zczb_eda.columns = ['year_type','HY','ETYPE','avg_ZCZB']
zczb_avg = train_1lenbase[['EID','ZCZB','year_type','HY','ETYPE']]
zczb_avg = pd.merge(zczb_avg,zczb_eda,on=['year_type','HY','ETYPE'],how='left')
zczb_avg['hy_year_avgzbzc'] = zczb_avg['ZCZB'] / zczb_avg['avg_ZCZB']
zczb_avg = zczb_avg[['EID','year_type','hy_year_avgzbzc']]

#时间段进行onthot编码
year_type_dummies=pd.get_dummies(zczb_avg.year_type,prefix='year_type')
year_type_dummies=pd.concat([zczb_avg[['EID']],year_type_dummies],axis=1)

train=pd.merge(train,year_type_dummies,how='left',on='EID')
predict=pd.merge(predict,year_type_dummies,how='left',on='EID')

train=pd.merge(train,zczb_avg,how='left',on='EID')
predict=pd.merge(predict,zczb_avg,how='left',on='EID')

#统计MPNUM、INUM、FINZB、FSTINUM、TZINUM在同一类型、同一行业、同一时间段中的占比
mpnum_eda = train_1lenbase[['MPNUM','year_type','HY','ETYPE']]
mpnum_eda = mpnum_eda[mpnum_eda['MPNUM'].notnull()]
mpnum_eda = mpnum_eda.groupby(['year_type','HY','ETYPE']).agg('mean').reset_index()
mpnum_eda.columns = ['year_type','HY','ETYPE','avg_mpnum']
mpnum_avg = train_1lenbase[['EID','MPNUM','year_type','HY','ETYPE']]
mpnum_avg = mpnum_avg[mpnum_avg['MPNUM'].notnull()]
mpnum_avg = pd.merge(mpnum_avg,mpnum_eda,on=['year_type','HY','ETYPE'],how='left')
mpnum_avg['hy_year_avgmpnum'] = mpnum_avg['MPNUM'] / mpnum_avg['avg_mpnum']
mpnum_avg = mpnum_avg[['EID','hy_year_avgmpnum']]

INUM_eda = train_1lenbase[['INUM','year_type','HY','ETYPE']]
INUM_eda = INUM_eda[INUM_eda['INUM'].notnull()]
INUM_eda = INUM_eda.groupby(['year_type','HY','ETYPE']).agg('mean').reset_index()
INUM_eda.columns = ['year_type','HY','ETYPE','avg_INUM']
INUM_avg = train_1lenbase[['EID','INUM','year_type','HY','ETYPE']]
INUM_avg = INUM_avg[INUM_avg['INUM'].notnull()]
INUM_avg = pd.merge(INUM_avg,INUM_eda,on=['year_type','HY','ETYPE'],how='left')
INUM_avg['hy_year_avgINUM'] = INUM_avg['INUM'] / INUM_avg['avg_INUM']
INUM_avg = INUM_avg[['EID','hy_year_avgINUM']]

FINZB_eda = train_1lenbase[['FINZB','year_type','HY','ETYPE']]
FINZB_eda = FINZB_eda[FINZB_eda['FINZB'].notnull()]
FINZB_eda = FINZB_eda.groupby(['year_type','HY','ETYPE']).agg('mean').reset_index()
FINZB_eda.columns = ['year_type','HY','ETYPE','avg_FINZB']
FINZB_avg = train_1lenbase[['EID','FINZB','year_type','HY','ETYPE']]
FINZB_avg = FINZB_avg[FINZB_avg['FINZB'].notnull()]
FINZB_avg = pd.merge(FINZB_avg,FINZB_eda,on=['year_type','HY','ETYPE'],how='left')
FINZB_avg['hy_year_avgFINZB'] = FINZB_avg['FINZB'] / FINZB_avg['avg_FINZB']
FINZB_avg = FINZB_avg[['EID','hy_year_avgFINZB']]

FSTINUM_eda = train_1lenbase[['FSTINUM','year_type','HY','ETYPE']]
FSTINUM_eda = FSTINUM_eda[FSTINUM_eda['FSTINUM'].notnull()]
FSTINUM_eda = FSTINUM_eda.groupby(['year_type','HY','ETYPE']).agg('mean').reset_index()
FSTINUM_eda.columns = ['year_type','HY','ETYPE','avg_FSTINUM']
FSTINUM_avg = train_1lenbase[['EID','FSTINUM','year_type','HY','ETYPE']]
FSTINUM_avg = FSTINUM_avg[FSTINUM_avg['FSTINUM'].notnull()]
FSTINUM_avg = pd.merge(FSTINUM_avg,FSTINUM_eda,on=['year_type','HY','ETYPE'],how='left')
FSTINUM_avg['hy_year_avgFSTINUM'] = FSTINUM_avg['FSTINUM'] / FSTINUM_avg['avg_FSTINUM']
FSTINUM_avg = FSTINUM_avg[['EID','hy_year_avgFSTINUM']]

TZINUM_eda = train_1lenbase[['TZINUM','year_type','HY','ETYPE']]
TZINUM_eda = TZINUM_eda[TZINUM_eda['TZINUM'].notnull()]
TZINUM_eda = TZINUM_eda.groupby(['year_type','HY','ETYPE']).agg('mean').reset_index()
TZINUM_eda.columns = ['year_type','HY','ETYPE','avg_TZINUM']
TZINUM_avg = train_1lenbase[['EID','TZINUM','year_type','HY','ETYPE']]
TZINUM_avg = TZINUM_avg[TZINUM_avg['TZINUM'].notnull()]
TZINUM_avg = pd.merge(TZINUM_avg,TZINUM_eda,on=['year_type','HY','ETYPE'],how='left')
TZINUM_avg['hy_year_avgTZINUM'] = TZINUM_avg['TZINUM'] / TZINUM_avg['avg_TZINUM']
TZINUM_avg = TZINUM_avg[['EID','hy_year_avgTZINUM']]

train=pd.merge(train,mpnum_avg,how='left',on=['EID'])
train=pd.merge(train,INUM_avg,how='left',on=['EID'])
train=pd.merge(train,FINZB_avg,how='left',on=['EID'])
train=pd.merge(train,FSTINUM_avg,how='left',on=['EID'])
train=pd.merge(train,TZINUM_avg,how='left',on=['EID'])

predict=pd.merge(predict,mpnum_avg,how='left',on=['EID'])
predict=pd.merge(predict,INUM_avg,how='left',on=['EID'])
predict=pd.merge(predict,FINZB_avg,how='left',on=['EID'])
predict=pd.merge(predict,FSTINUM_avg,how='left',on=['EID'])
predict=pd.merge(predict,TZINUM_avg,how='left',on=['EID'])

train.to_csv('train_114_nozero3.csv',index=False)
predict.to_csv('test_114_nozero3.csv',index=False)



