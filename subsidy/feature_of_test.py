# -*-coding:utf-8 -*-
import pandas as pd
import numpy as np
import os
import datetime,time

'''
特征挖掘
'''

#清洗重复的数据,输入文件路径对本路径下所有的csv文件进行去重
def drop_duplicates(url):
    dirlist = os.dirlist(url)
    for names in dirlist:
        file = url+names
        print '读取'+names+',开始去重……'
        data = pd.read_csv(file,header=None)
        data.drop_duplicates(inplace=True)
        data.to_csv(url,header=False,index=False)

# 添加性别特征(未完待续)
def get_sex():
    def chang_student(li, data):  # 根据学生性别，修改淋浴性别属性
        data_temp = data[data['studentid'].isin(li)]
        data = data[data['studentid'].isin(li) == False]
        location_list = []
        for row in data_temp.itertuples(index=False):
            studentid, locationid = row
            tuple_of_sex = student_sex[studentid]
            tuple_of_sex = tuple_of_sex / float(sum(tuple_of_sex))
            location_sex[locationid] += tuple_of_sex
            location_list.append(locationid)
        return location_list, data

    def chang_location(li, data):  # 根据地点性别，修改学生性别属性
        data_temp = data[data['消费地点'].isin(li)]
        data = data[data['消费地点'].isin(li) == False]
        student_list = []
        for row in data_temp.itertuples(index=False):
            studentid, locationid = row
            tuple_of_sex = location_sex[locationid]
            tuple_of_sex = tuple_of_sex / float(sum(tuple_of_sex))
            student_sex[studentid] += tuple_of_sex
            student_list.append(studentid)
        return student_list, data

    student_sex = pd.DataFrame({'studentid': test['studentid'], 'male': 0, 'female': 0})
    student_sex.set_index('studentid', inplace=True)
    student_sex = student_sex.T
    location_sex = pd.DataFrame(
        {'消费地点': card_test[card_test['消费方式'] == '淋浴']['消费地点'].unique(), 'male': 0, 'female': 0})
    location_sex.set_index('消费地点', inplace=True)
    location_sex = location_sex.T
    data = card_test[card_test['消费方式'] == '淋浴'][['studentid', '消费地点']]
    data.drop_duplicates(inplace=True)
    data = data[data['消费地点'] != '地点6']  # 地点6  出现频率过高删除
    student_sex[31281] += (0, 1)  # 选择31281  学生id 作为起始id
    student_list = [31281]
    len_of_data = len(data)
    while (True):
        location_list, data = chang_student(student_list, data)
        if len_of_data == len(data): break
        len_of_data = len(data)
        student_list, data = chang_location(location_list, data)
        if len_of_data == len(data): break
        len_of_data = len(data)



#读取文件
#选取学号，和助学经作为merge的标签
test = pd.read_csv(r'C:\Users\CSW\Desktop\python\subsidy\input\test\studentID_test.txt',header=None,names=['studentid'])
test.set_index('studentid',inplace=True)
#读取card表提取信息
card_test = pd.read_csv(r'C:\Users\CSW\Desktop\python\subsidy\input\test\card_test.txt',header=None,names=['studentid','消费类别','消费地点','消费方式','消费时间','消费金额','剩余金额'])
card_test['消费时间'] = card_test['消费时间'].apply(lambda x: datetime.datetime.strptime(x,'%Y/%m/%d %H:%M:%S'))
card_test['day'] = card_test['消费时间'].apply(lambda x: x.date())
#读取院系和成绩信息
score_test = pd.read_csv(r'C:\Users\CSW\Desktop\python\subsidy\input\test\score_test.txt',header=None,names=['studentid','college','rank'])
score_test.set_index('studentid',inplace=True)
#读取寝室门禁文件
dorm_test = pd.read_csv(r'C:\Users\CSW\Desktop\python\subsidy\input\test\dorm_test.txt',header=None,names=['studentid','time','IO'])
dorm_test['time'] = dorm_test['time'].apply(lambda x: datetime.datetime.strptime(x,'%Y/%m/%d %H:%M:%S'))
dorm_test['day'] = dorm_test['time'].apply(lambda x: x.date())
#读取图书馆图书借阅记录
borrow_test = pd.read_csv(r'C:\Users\CSW\Desktop\python\subsidy\input\test\borrow_test.txt',header=None,names=['studentid','time','book_name','bookid'])


#选取年份信息作为特征，13~14为0,14~15为1
test['year'] = test.index % 2
#寝室门禁进出次数
test['count_of_dorm'] = dorm_test.groupby('studentid').agg('size')
#寝室门禁记录的天数(默认在校天数)
test['day_of_dorm'] = dorm_test.groupby(['studentid'])['day'].agg(lambda x:len(x.unique()))
#平均每天进出次数
test['count_dorm_perday'] = test['count_of_dorm']/test['day_of_dorm']

#添加院系信息
test['college'] = score_test['college']
#添加院系人数（后期可删除）
count_of_student = score_test.groupby('college')['rank'].max()+1
score_test['count_of_student'] = score_test.apply(lambda x:count_of_student[int(x['college'])],axis=1)
test['count_of_student'] = score_test['count_of_student']
#添加排名（后期可删除）
test['rank'] = score_test['rank']
#排名归一化
test['rank_of_relative'] = test['rank']/test['count_of_student']

#每个同学卡消费总额
test['sum_of_consume'] = card_test[card_test['消费类别']=='POS消费'].groupby('studentid')['消费金额'].agg('sum')
#每个同学卡消费平均值
test['mean_of_consume'] = card_test[card_test['消费类别']=='POS消费'].groupby('studentid')['消费金额'].agg('mean')
#每个同学卡消费最大值
test['var_of_consume'] = card_test[card_test['消费类别']=='POS消费'].groupby('studentid')['消费金额'].agg('var')
#每个同学卡消费次数
test['count_of_consume'] = card_test[card_test['消费类别']=='POS消费'].groupby('studentid')['消费金额'].agg('size')
#每天消费总额
test['consume_of_day'] = card_test[card_test['消费类别']=='POS消费'].groupby(['studentid'])['day'].agg(lambda x:len(x.unique()))
#在校时校园卡使用频率
day_of_dorm = pd.DataFrame(dorm_test.groupby(['studentid'])['day'].agg(lambda x:set(x)))
day_of_consume = pd.DataFrame(card_test.groupby(['studentid'])['day'].agg(lambda x:set(x)))
dorm_and_consume = pd.merge(day_of_dorm,day_of_consume,left_index=True,right_index=True)
test['percent_of_day'] = dorm_and_consume.apply(lambda x:len(x[0].intersection(x[1])),axis=1)/day_of_dorm['day'].apply(len)

#每个同学吃饭消费总额
test['sum_of_meal'] = card_test[card_test['消费方式']=='食堂'].groupby('studentid')['消费金额'].agg('sum')
#每个同学吃饭消费的次数(对于时间间隔小于两个小时的合并为一次)
card_test_temp = list(card_test[card_test['消费方式']=='食堂'].values)
data = []
row_temp = card_test_temp.pop(0)
for row in card_test_temp:
    if np.abs((row[4]-row_temp[4]).total_seconds())<7200:
        row[5] = row_temp[5]+row[5]
    else:
        data.append(row_temp)
    row_temp = row
data.append(row_temp)
data = pd.DataFrame(data,columns=card_test.columns)
test['count_of_meal'] = data.groupby('studentid').size()
#每个同学的平均一次的饭钱
test['mean_of_meal'] = test['sum_of_meal']/test['count_of_meal']
#每个同学平均每顿饭的点餐次数
test['count_of_permeal'] = card_test[card_test['消费方式']=='食堂'].groupby('studentid').size()/test['count_of_meal']
#每位同学吃饭所占校园卡消费的总量
test['percent_of_meal'] = test['sum_of_meal']/test['sum_of_consume']
#每个同学吃饭的天数
test['day_of_meal'] = card_test[card_test['消费方式']=='食堂'].groupby('studentid')['day'].agg('nunique')
#在校吃饭天数占在校天数的比例
test['percent_of_atSchool'] = test['day_of_meal']/test['day_of_dorm']
#平均每天在校吃饭次数
test['count_meal_perday'] = test['count_of_meal']/test['day_of_meal']
#每天吃饭金额的标准差
temp  = pd.DataFrame(card_test[card_test['消费方式']=='食堂'].groupby(['studentid','day'])['消费金额'].sum())
temp.reset_index(inplace=True)
test['std_meal_perday'] = temp.groupby('studentid')['消费金额'].std()

#每个同学卡余额关于时间的平均值
card_test_temp = card_test.copy()
mean_of_overage = {}
for name,student in card_test_temp.groupby('studentid'):
    student = list(student.values)
    row_temp = student.pop(0)
    sum_of_seconds = 0
    sum_of_overage = 0
    for row in student:
        seconds = abs((row[4]-row_temp[4]).total_seconds())
        sum_of_seconds = sum_of_seconds + seconds
        sum_of_overage = sum_of_overage + seconds*row_temp[6]
        row_temp = row
    mean = sum_of_overage/sum_of_seconds if sum_of_seconds!=0 else None
    mean_of_overage[name] = mean
test['mean_of_overage'] = pd.Series(mean_of_overage)


#卡充值次数
card_test_temp = card_test[card_test['消费类别'].isin(['圈存转账','卡充值'])]
test['count_of_storage'] = card_test_temp.groupby('studentid')['消费类别'].agg('size')
#卡平均充值金额
test['mean_of_storage'] = card_test_temp.groupby('studentid')['消费金额'].agg('mean')
#卡最大充值金额
test['max_of_storage'] = card_test_temp.groupby('studentid')['消费金额'].agg('max')
#充值时卡上的余额的平均值
test['mean_of_over_storage'] = card_test_temp.groupby('studentid')['剩余金额'].agg('mean')-card_test_temp.groupby('studentid')['消费金额'].agg('mean')

#每个同学开水消费总额
test['sum_of_water'] = card_test[card_test['消费方式']=='开水'].groupby('studentid')['消费金额'].agg('sum')
#每个同学开水消费总次数
test['count_of_meal'] = card_test[card_test['消费方式']=='开水'].groupby('studentid')['消费金额'].agg('count')
#每个同学开水消费平均值
test['mean_of_water'] = card_test[card_test['消费方式']=='开水'].groupby('studentid')['消费金额'].agg('mean')
#每个同学开水消费总额所占花销比例
test['percent_of_water'] = test['sum_of_water']/test['sum_of_consume']

#每个同学淋浴消费总额
test['sum_of_bath'] = card_test[card_test['消费方式']=='淋浴'].groupby('studentid')['消费金额'].agg('sum')
#每个同学淋浴消费总次数
test['count_of_bath'] = card_test[card_test['消费方式']=='淋浴'].groupby('studentid')['消费金额'].agg('count')
#每个同学淋浴消费平均值
test['mean_of_bath'] = card_test[card_test['消费方式']=='淋浴'].groupby('studentid')['消费金额'].agg('median')
#每个同学淋浴消费总额所占花销比例
test['percent_of_bath'] = test['sum_of_bath']/test['sum_of_consume']

#每个同学校车消费总额
test['sum_of_bus'] = card_test[card_test['消费方式']=='校车'].groupby('studentid')['消费金额'].agg('sum')
#每个同学校车消费总次数
test['count_of_bus'] = card_test[card_test['消费方式']=='校车'].groupby('studentid')['消费金额'].agg('count')
#每个同学校车消费平均值
test['mean_of_bus'] = card_test[card_test['消费方式']=='校车'].groupby('studentid')['消费金额'].agg('mean')

#每个同学超市消费总额
test['sum_of_shopping'] = card_test[card_test['消费方式']=='超市'].groupby('studentid')['消费金额'].agg('sum')
#每个同学超市消费总次数
test['count_of_shopping'] = card_test[card_test['消费方式']=='超市'].groupby('studentid')['消费金额'].agg('count')
#每个同学超市消费平均值
test['mean_of_shopping'] = card_test[card_test['消费方式']=='超市'].groupby('studentid')['消费金额'].agg('mean')
#每个同学超市消费总额所占花销比例
test['percent_of_shopping'] = test['sum_of_shopping']/test['sum_of_consume']

#每个同学洗衣房消费总额
test['sum_of_wash'] = card_test[card_test['消费方式']=='洗衣房'].groupby('studentid')['消费金额'].agg('sum')
#每个同学洗衣房消费总次数
test['count_of_wash'] = card_test[card_test['消费方式']=='洗衣房'].groupby('studentid')['消费金额'].agg('count')
#每个同学洗衣房消费平均值
test['mean_of_wash'] = card_test[card_test['消费方式']=='洗衣房'].groupby('studentid')['消费金额'].agg('mean')
#每个同学洗衣房消费总额所占花销比例
test['percent_of_wash'] = test['sum_of_wash']/test['sum_of_consume']

#每个同学图书馆消费总额
test['sum_of_library'] = card_test[card_test['消费方式']=='图书馆'].groupby('studentid')['消费金额'].agg('sum')
#每个同学图书馆消费总次数
test['count_of_library'] = card_test[card_test['消费方式']=='图书馆'].groupby('studentid')['消费金额'].agg('count')
#每个同学图书馆消费平均值
test['mean_of_library'] = card_test[card_test['消费方式']=='图书馆'].groupby('studentid')['消费金额'].agg('mean')
#每个同学图书馆消费总额所占花销比例
test['percent_of_library'] = test['sum_of_library']/test['sum_of_consume']

#每个同学文印中心消费总额
test['sum_of_print'] = card_test[card_test['消费方式']=='文印中心'].groupby('studentid')['消费金额'].agg('sum')
#每个同学文印中心消费总次数
test['count_of_print'] = card_test[card_test['消费方式']=='文印中心'].groupby('studentid')['消费金额'].agg('count')
#每个同学文印中心消费平均值
test['mean_of_print'] = card_test[card_test['消费方式']=='文印中心'].groupby('studentid')['消费金额'].agg('mean')
#每个同学文印中心消费总额所占花销比例
test['percent_of_print'] = test['sum_of_print']/test['sum_of_consume']

#每个同学教务处消费总额
test['sum_of_Trianing'] = card_test[card_test['消费方式']=='教务处'].groupby('studentid')['消费金额'].agg('sum')
#每个同学教务处消费总次数
test['count_of_Trianing'] = card_test[card_test['消费方式']=='教务处'].groupby('studentid')['消费金额'].agg('count')

#每个同学医院消费总额
test['sum_of_hospital'] = card_test[card_test['消费方式']=='医院'].groupby('studentid')['消费金额'].agg('sum')
#每个同学医院消费总次数
test['count_of_hospital'] = card_test[card_test['消费方式']=='医院'].groupby('studentid')['消费金额'].agg('count')

#每个同学其他消费总额
test['sum_of_other'] = card_test[card_test['消费方式']=='其他'].groupby('studentid')['消费金额'].agg('sum')
#每个同学其他消费总次数
test['count_of_other'] = card_test[card_test['消费方式']=='其他'].groupby('studentid')['消费金额'].agg('count')

#对比两年的变化
test_temp1 = test[test.index%2==1]
test_temp2 = test[test.index%2==0]
test_temp1.index = test_temp1.index-1
test_temp2.index = test_temp2.index+1
test_temp = pd.concat([test_temp1,test_temp2])
test_temp3 = (test+test_temp)/2
#计算第二年校园卡消费比第一年增加量
test['differenct_of_consume'] = test_temp['sum_of_consume']-test['sum_of_consume']
#计算第二年校园卡消费比第一年增加量所占百分比
test['percent_differenct_consume'] = test['differenct_of_consume'] /(test_temp3['sum_of_consume']+1)
#计算第二年校园卡食堂消费比第一年增加量
test['differenct_of_meal'] = test_temp['sum_of_meal']-test['sum_of_meal']
#计算第二年校园卡食堂消费比第一年增加量所占百分比
test['percent_differenct_meal'] = test['differenct_of_meal'] /(test_temp3['sum_of_meal']+1)
#计算第二年校园卡非食堂消费比第一年增加量
test['differenct_of_nomeal'] = test['differenct_of_consume']-test['differenct_of_meal']
#计算第二年校园卡非消费比第一年增加量所占百分比
test['percent_differenct_noconsume'] = test['differenct_of_nomeal'] /(test_temp3['sum_of_consume']-test_temp3['sum_of_meal']+1)

test.reset_index(inplace=True)




















