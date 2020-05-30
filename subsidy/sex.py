import numpy as np
import pandas as pd

#读取数据
card_train = pd.read_csv(r'C:\Users\CSW\Desktop\python\subsidy\input\train\card_train.txt',header=None,names=['studentid','消费类别','消费地点','消费方式','消费时间','消费金额','剩余金额'])
card_train['消费地点'] = card_train['消费地点'].apply(lambda x:x if x is np.nan else x.replace('地点','loc'))

#数据准备
card_train_shower = card_train[card_train['消费方式']=='淋浴'][['studentid','消费地点']]
card_train_shower['studentid'] = card_train_shower['studentid'].apply(lambda x: x if x%2==0 else x-1)
card_train_shower.drop_duplicates(inplace=True)
card_train_shower = card_train_shower[card_train_shower['消费地点']!='loc6']

student_sex = pd.DataFrame({'studentid':list(set(card_train_shower['studentid'])),'male':np.nan,'female':np.nan})
student_sex = student_sex.set_index('studentid').T
shower_sex = pd.DataFrame({'消费地点':list(set(card_train_shower['消费地点'])),'male':np.nan,'female':np.nan})
shower_sex = shower_sex.set_index('消费地点').T
student_sex[30482] = 1,0
student_sex[22008] = 0,1

#开始循环
#改变淋浴id的性别
def chang_shower(data, student_sex):
    return pd.DataFrame(dict(data.groupby('消费地点')['studentid'].agg(lambda x : list(student_sex[list(x)].mean(axis=1)))),index=['male','female']).round()

#改变学生id的性别
def chang_student(data, shower_sex):
    return pd.DataFrame(dict(data.groupby('studentid')['消费地点'].agg(lambda x : list(shower_sex[list(x)].mean(axis=1)))),index=['male','female'])

for i in xrange(10):
    shower_sex = chang_shower(card_train_shower,student_sex)
    student_sex = chang_student(card_train_shower, shower_sex)


student_sex = student_sex.T.reset_index()
student_sex = student_sex.round()
student_sex.rename(columns={'index':'studentid'},inplace=True)
student_sex2 = student_sex.copy()
student_sex2['studentid'] = student_sex2['studentid']+1
student_sex3 = pd.concat([student_sex,student_sex2])
train = pd.merge(train_temp[['studentid','subsidy','college']],student_sex3,on='studentid',how='left')