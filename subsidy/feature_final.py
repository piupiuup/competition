# -*-coding:utf-8 -*-
import pandas as pd
import numpy as np
import os
import datetime,time

'''
由于我们的特征提取代码比较长，看起来可能会不太方便，所以在开头部分我们进行一些简单的介绍：
1、由于本次比赛数据中有不少重复错误的行，所以在提取特征之前我们已经进行了去重的处理
2、这个代码由于涉及大量的时间运算，所以运行起来会比较慢，我i7的cpu大概需要一个小时吧。。。
3、代码中有几个部分可能不太容易阅读或者理解：
    （1）考虑到学生在食堂的每次消费可能不是通过一次刷开完成的，我对每顿饭的定义是:前后两个
        小时相邻的食堂消费我都个合并成一次了。
    （2）在对开水、淋浴、洗衣房等消费类型进行特征统计时，
        我们发现了一些异常的消费地点（金额异常或者消费次数异常），对其进行了删除。
    （3）中间有部分我们对13~14年的消费，和14~15年的消费进行了对比，在这里是我发现相邻两个学号的
        可能是同一个学生连续两年的消费信息
4、我们的特征经过多次的添加，修改中间可能会会出现一些很没有用，或者重复的特征。我们之后对特征
进行过按照分类模型的importance和人工进行过筛选（主要是前者）。
5、本代码由我们队伍共同合作完成，丁明（QQ：智浪淘沙，我们爱称的 浪教授）同学做出重要贡献。如发现
重要错误、建议或者有什么交流的想法，欢迎在DC群里@我们或者私聊。
'''

def get_feature(mold):

    #--------------------------------------读取文件-----------------------------------------#

    #选取学号，和助学经作为merge的标签
    if mold!='test':
        data = pd.read_csv(r'C:\Users\CSW\Desktop\python\subsidy\input\%s\subsidy_%s.txt'% (mold, mold),header=None,names=['studentid','subsidy'])
    else:
        data = pd.read_csv(r'C:\Users\CSW\Desktop\python\subsidy\input\%s\studentID_%s.txt' % (mold, mold), header=None,
                           names=['studentid', 'subsidy'])
    data.set_index('studentid',inplace=True)
    #读取card表提取信息
    card_data = pd.read_csv(r'C:\Users\CSW\Desktop\python\subsidy\input\%s\card_%s.txt'% (mold, mold),header=None,names=['studentid','消费类别','消费地点','消费方式','消费时间','消费金额','剩余金额'])
    card_data['消费地点'] = card_data['消费地点'].apply(lambda x: x if x is np.nan else x.replace('地点', 'loc'))
    card_data['消费时间'] = card_data['消费时间'].apply(lambda x: datetime.datetime.strptime(x,'%Y/%m/%d %H:%M:%S'))
    card_data['day'] = card_data['消费时间'].apply(lambda x: x.dayofyear)
    card_data['week'] = card_data['消费时间'].apply(lambda x: x.weekofyear)
    card_data['month'] = card_data['消费时间'].apply(lambda x: x.month)
    card_data['quarter'] = card_data['消费时间'].apply(lambda x: x.quarter)
    #读取院系和成绩信息
    score_data = pd.read_csv(r'C:\Users\CSW\Desktop\python\subsidy\input\%s\score_%s.txt'% (mold, mold),header=None,names=['studentid','college','rank'])
    score_data.set_index('studentid',inplace=True)
    #读取寝室门禁文件
    dorm_data = pd.read_csv(r'C:\Users\CSW\Desktop\python\subsidy\input\%s\dorm_%s.txt'% (mold, mold),header=None,names=['studentid','time','IO'])
    dorm_data['time'] = dorm_data['time'].apply(lambda x: datetime.datetime.strptime(x,'%Y/%m/%d %H:%M:%S'))
    dorm_data['day'] = dorm_data['time'].apply(lambda x: x.date())
    #读取图书馆图书借阅记录
    borrow_data = pd.read_csv(r'C:\Users\CSW\Desktop\python\subsidy\input\%s\borrow_%s.txt'% (mold, mold),header=None,names=['studentid','time','book_name','bookid'])
    borrow_data['time'] = borrow_data['time'].apply(lambda x: datetime.datetime.strptime(x, '%Y/%m/%d'))
    #读取图书馆出入记录
    library_data = pd.read_csv(r'C:\Users\CSW\Desktop\python\subsidy\input\%s\library_%s.txt'%(mold, mold), header=None, names=['studentid', 'doorid', 'time'])
    library_data['time'] = library_data['time'].apply(lambda x: datetime.datetime.strptime(x, '%Y/%m/%d %H:%M:%S'))
    library_data['day'] = library_data['time'].apply(lambda x: x.date())
    library_data['doorid'] = library_data['doorid'].apply(lambda x: x if x is np.nan else x.replace('进门', 'in'))
    library_data['doorid'] = library_data['doorid'].apply(lambda x: x if x is np.nan else x.replace('出门', 'out'))
    library_data['doorid'] = library_data['doorid'].apply(lambda x: x if x is np.nan else x.replace('小门', 'door'))




    #-------------------------------------特征提取-----------------------------------#

    #选取年份信息作为特征，13~14为0,14~15为1
    time = datetime.datetime.strptime('2014-09-01 00:00:00','%Y-%m-%d %H:%M:%S')
    data['year'] = card_data.groupby('studentid')['消费时间'].apply(lambda x : 0 if x.min()<time else 1)
    #寝室门禁进出次数
    data['count_of_dorm'] = dorm_data.groupby('studentid').agg('size')
    #寝室门禁记录的天数(默认在校天数)
    data['day_of_dorm'] = dorm_data.groupby(['studentid'])['day'].agg(lambda x:len(x.unique()))
    #平均每天进出次数
    data['count_dorm_perday'] = data['count_of_dorm']/data['day_of_dorm']

    #添加院系信息(其实就是对院系进行onehot)
    data['college'] = score_data['college']
    colleges = list(score_data['college'].unique())
    score_data.reset_index(inplace=True)
    for college in colleges:
        name = 'college_of_' + str(college)
        id_of_college = list(score_data[score_data['college'] == college]['studentid'].values)
        data[name] = [(1 if id in id_of_college else 0) for id in list(data.index)]
    score_data.set_index('studentid',inplace=True)
    #添加院系人数（后期可删除）
    count_of_student = score_data.groupby('college')['rank'].max()+1
    score_data['count_of_student'] = score_data.apply(lambda x:count_of_student[int(x['college'])],axis=1)
    data['count_of_student'] = score_data['count_of_student']
    #添加排名（后期可删除）
    data['rank'] = score_data['rank']
    #排名归一化
    data['rank_of_relative'] = data['rank']/data['count_of_student']
    #添加院系获奖率
    subsidy_train = pd.read_csv(r'C:\Users\CSW\Desktop\python\subsidy\input\train\subsidy_train.txt', header=None,names=['studentid', 'subsidy'])
    score_train = pd.read_csv(r'C:\Users\CSW\Desktop\python\subsidy\input\train\score_train.txt', header=None,names=['studentid', 'college', 'rank'])
    subsidy_train = pd.merge(subsidy_train, score_train, on='studentid', how='left')
    percent_of_college = subsidy_train.groupby(['college', 'subsidy'])['studentid'].agg('count').unstack().fillna(
        0).divide(subsidy_train.groupby('college')['studentid'].agg('count'), axis=0)
    percent_of_college['percent_of_>0'] = 1 - percent_of_college[0]
    percent_of_college['percent_of_>1000'] = percent_of_college[[1500, 2000]].sum(axis=1) / percent_of_college['percent_of_>0']
    percent_of_college['percent_of_>1500'] = percent_of_college[2000] / percent_of_college[[1500, 2000]].sum(axis=1)
    percent_of_college.reset_index(inplace=True)
    data.reset_index(inplace=True)
    data = pd.merge(data,percent_of_college[['college','percent_of_>0','percent_of_>1000','percent_of_>1500']],on='college',how='left')
    data.set_index('studentid',inplace=True)

    #每个同学卡消费总额
    data['sum_of_consume'] = card_data[card_data['消费类别']=='POS消费'].groupby('studentid')['消费金额'].agg('sum')
    #每个同学卡消费平均值
    data['mean_of_consume'] = card_data[card_data['消费类别']=='POS消费'].groupby('studentid')['消费金额'].agg('mean')
    #每个同学卡消费标准差
    data['var_of_consume'] = card_data[card_data['消费类别']=='POS消费'].groupby('studentid')['消费金额'].agg('var')
    # 每个同学卡消费最大值
    data['max_of_consume'] = card_data[card_data['消费类别'] == 'POS消费'].groupby('studentid')['消费金额'].agg('max')
    # 每个同学卡消费标准差
    data['min_of_consume'] = card_data[card_data['消费类别'] == 'POS消费'].groupby('studentid')['消费金额'].agg('min')
    # 每个同学每天消费的偏度
    data['skew_of_consume'] = card_data[card_data['消费类别'] == 'POS消费'].groupby('studentid')['消费金额'].agg('skew') + 100
    # 每个同学每天消费的偏度
    data['kurt_of_consume'] = card_data[card_data['消费类别'] == 'POS消费'].groupby('studentid')['消费金额'].agg(lambda x:x.kurt()) + 100
    #每个同学卡消费次数
    data['count_of_consume'] = card_data[card_data['消费类别']=='POS消费'].groupby('studentid')['消费金额'].agg('size')
    #每天消费总额
    data['consume_of_day'] = card_data[card_data['消费类别']=='POS消费'].groupby(['studentid'])['day'].agg(lambda x:len(x.unique()))
    #消费类型的个数
    data['kind_of_consume'] = card_data[card_data['消费类别']=='POS消费'].groupby(['studentid'])['消费方式'].agg('nunique')
    #在校时校园卡使用频率
    day_of_dorm = pd.DataFrame(dorm_data.groupby(['studentid'])['day'].agg(lambda x:set(x)))
    day_of_consume = pd.DataFrame(card_data.groupby(['studentid'])['day'].agg(lambda x:set(x)))
    dorm_and_consume = pd.merge(day_of_dorm,day_of_consume,left_index=True,right_index=True)
    data['percent_of_day'] = dorm_and_consume.apply(lambda x:len(x[0].intersection(x[1])),axis=1)/day_of_dorm['day'].apply(len)

    # 每个同学吃饭消费总额
    data['sum_of_meal'] = card_data[card_data['消费方式']=='食堂'].groupby('studentid')['消费金额'].agg('sum')
    # 每个同学吃饭消费的偏度
    data['skew_of_meal'] = card_data[card_data['消费方式'] == '食堂'].groupby('studentid')['消费金额'].agg('skew') + 100
    # 每个同学吃饭消费的锋度
    data['skew_of_meal'] = card_data[card_data['消费方式'] == '食堂'].groupby('studentid')['消费金额'].agg(lambda x:x.kurt()) + 100
    #每个同学吃饭消费的次数(对于时间间隔小于两个小时的合并为一次)
    card_data_temp1 = list(card_data[card_data['消费方式']=='食堂'].values)
    card_data_temp2 = []
    row_temp = card_data_temp1.pop(0)
    for row in card_data_temp1:
        if np.abs((row[4]-row_temp[4]).total_seconds())<7200:
            row[5] = row_temp[5]+row[5]
        else:
            card_data_temp2.append(row_temp)
        row_temp = row
    card_data_temp2.append(row_temp)
    card_data_temp2 = pd.DataFrame(card_data_temp2,columns=card_data.columns)
    data['count_of_meal'] = card_data_temp2.groupby('studentid').size()
    #每个同学的平均一次的饭钱
    data['mean_of_meal'] = data['sum_of_meal']/data['count_of_meal']
    #每个同学每次食堂消费的标准差
    data['std_of_meal'] = card_data_temp2.groupby('studentid')['消费金额'].agg('std')
    #每个同学平均每顿饭的点餐次数
    data['count_of_permeal'] = card_data[card_data['消费方式']=='食堂'].groupby('studentid').size()/data['count_of_meal']
    #每位同学吃饭所占校园卡消费的总量
    data['percent_of_meal'] = data['sum_of_meal']/data['sum_of_consume']
    #每个同学吃饭的天数
    data['day_of_meal'] = card_data[card_data['消费方式']=='食堂'].groupby('studentid')['day'].agg('nunique')
    #在校吃饭天数占在校天数的比例
    data['percent_of_atSchool'] = data['day_of_meal']/data['day_of_dorm']
    #平均每天在校吃饭次数
    #data['count_meal_perday'] = data['count_of_meal']/data['day_of_meal']
    #每天吃饭金额的标准差
    temp  = pd.DataFrame(card_data[card_data['消费方式']=='食堂'].groupby(['studentid','day'])['消费金额'].sum())
    temp.reset_index(inplace=True)
    data['std_meal_perday'] = temp.groupby('studentid')['消费金额'].std()

    #每个同学卡余额关于时间的平均值
    card_data_temp = card_data.copy()
    mean_of_overage = {}
    for name,student in card_data_temp.groupby('studentid'):
        student = list(student.values)
        row_temp = student.pop(0)
        sum_of_seconds = 0
        sum_of_overage = 0
        for row in student:
            seconds = abs((row[4]-row_temp[4]).total_seconds())
            sum_of_seconds = sum_of_seconds + seconds
            sum_of_overage = sum_of_overage + seconds*row_temp[6]
            row_temp = row
        mean = sum_of_overage/sum_of_seconds if sum_of_seconds!=0 else sum_of_overage
        mean_of_overage[name] = mean
    data['mean_of_overage'] = pd.Series(mean_of_overage)


    #卡充值次数
    card_data_temp = card_data[card_data['消费类别'].isin(['圈存转账','卡充值','支付领取'])]
    data['count_of_storage'] = card_data_temp.groupby('studentid')['消费类别'].agg('size')
    #卡平均充值金额
    data['mean_of_storage'] = card_data_temp.groupby('studentid')['消费金额'].agg('mean')
    #卡最大充值金额
    data['max_of_storage'] = card_data_temp.groupby('studentid')['消费金额'].agg('max')
    # 卡最小充值金额
    data['min_of_storage'] = card_data_temp.groupby('studentid')['消费金额'].agg('min')
    # 卡充值金额标准差
    data['std_of_storage'] = card_data_temp.groupby('studentid')['消费金额'].agg('std')
    #充值时卡上的余额的平均值
    data['mean_of_over_storage'] = card_data_temp.groupby('studentid')['剩余金额'].agg('mean')-card_data_temp.groupby('studentid')['消费金额'].agg('mean')

    #卡挂失次数
    data['count_of_loss'] = card_data[card_data['消费类别']=='卡挂失'].groupby('studentid')['studentid'].agg('count')

    # 清理开水异常数据
    card_data_temp = card_data[~card_data['消费地点'].isin(['loc21'])]
    # 每个同学开水消费总额
    data['sum_of_water'] = card_data_temp[card_data_temp['消费方式'] == '开水'].groupby('studentid')['消费金额'].agg('sum')
    # 每个同学开水消费总次数
    data['count_of_water'] = card_data_temp[card_data_temp['消费方式'] == '开水'].groupby('studentid')['消费金额'].agg('count')
    # 每个同学开水消费平均值
    data['mean_of_water'] = card_data_temp[card_data_temp['消费方式'] == '开水'].groupby('studentid')['消费金额'].agg('mean')
    # 每个同学开水消费标准差
    data['std_of_water'] = card_data_temp[card_data_temp['消费方式'] == '开水'].groupby('studentid')['消费金额'].agg('std')
    # 每个同学开水消费偏度
    data['skew_of_water'] = card_data_temp[card_data_temp['消费方式'] == '开水'].groupby('studentid')['消费金额'].agg('skew') + 100
    # 每个同学开水消费锋度
    data['kurt_of_water'] = card_data_temp[card_data_temp['消费方式'] == '开水'].groupby('studentid')['消费金额'].agg(lambda x: x.kurt()) + 100
    # 每个同学开水消费的最大值
    data['max_of_water'] = card_data_temp[card_data_temp['消费方式'] == '开水'].groupby('studentid')['消费金额'].agg('max')
    # 每个同学使用开水的天数
    data['day_of_water'] = card_data_temp[card_data_temp['消费方式'] == '开水'].groupby('studentid')['day'].agg('nunique')
    # 每个同学开水消费总额所占花销比例
    data['percent_of_water'] = data['sum_of_water'] / data['sum_of_consume']

    # 清理淋浴异常数据
    card_data_temp = card_data[~card_data['消费地点'].isin(['loc73', 'loc6'])]
    # 每个同学淋浴消费总额
    data['sum_of_bath'] = card_data_temp[card_data_temp['消费方式'] == '淋浴'].groupby('studentid')['消费金额'].agg('sum')
    # 每个同学淋浴消费总次数
    data['count_of_bath'] = card_data_temp[card_data_temp['消费方式'] == '淋浴'].groupby('studentid')['消费金额'].agg('count')
    # 每个同学淋浴消费中位数
    data['median_of_bath'] = card_data_temp[card_data_temp['消费方式'] == '淋浴'].groupby('studentid')['消费金额'].agg('median')
    # 每个同学淋浴的标准差
    data['std_of_bath'] = card_data_temp[card_data_temp['消费方式'] == '淋浴'].groupby('studentid')['消费金额'].agg('std')
    # 每个同学淋浴消费偏度
    data['skew_of_bath'] = card_data_temp[card_data_temp['消费方式'] == '淋浴'].groupby('studentid')['消费金额'].agg('skew') + 100
    # 每个同学淋浴消费锋度
    data['kurt_of_bath'] = card_data_temp[card_data_temp['消费方式'] == '淋浴'].groupby('studentid')['消费金额'].agg(lambda x: x.kurt()) + 100
    # 淋浴消费的天数
    data['day_of_bath'] = card_data_temp[card_data_temp['消费方式'] == '淋浴'].groupby('studentid')['day'].agg('nunique')
    # 每个同学淋浴消费总额所占花销比例
    data['percent_of_bath'] = data['sum_of_bath'] / data['sum_of_consume']

    #每个同学校车消费总额
    data['sum_of_bus'] = card_data[card_data['消费方式']=='校车'].groupby('studentid')['消费金额'].agg('sum')
    #每个同学校车消费总次数
    data['count_of_bus'] = card_data[card_data['消费方式']=='校车'].groupby('studentid')['消费金额'].agg('count')
    #每个同学校车消费平均值
    data['mean_of_bus'] = card_data[card_data['消费方式']=='校车'].groupby('studentid')['消费金额'].agg('mean')
    # 每个同学校车消费最大值
    data['max_of_bus'] = card_data[card_data['消费方式'] == '校车'].groupby('studentid')['消费金额'].agg('max')
    # 每个同学校车消费偏度
    data['skew_of_bus'] = card_data[card_data['消费方式'] == '校车'].groupby('studentid')['消费金额'].agg('skew') + 100
    # 每个同学校车消费峰度
    data['kurt_of_bus'] = card_data[card_data['消费方式'] == '校车'].groupby('studentid')['消费金额'].agg(lambda x:x.kurt()) + 100
    #校车消费的天数
    data['day_of_bus'] = card_data[card_data['消费方式']=='校车'].groupby('studentid')['day'].agg('nunique')

    #每个同学超市消费总额
    data['sum_of_shopping'] = card_data[card_data['消费方式']=='超市'].groupby('studentid')['消费金额'].agg('sum')
    #每个同学超市消费总次数
    data['count_of_shopping'] = card_data[card_data['消费方式']=='超市'].groupby('studentid')['消费金额'].agg('count')
    #每个同学超市消费平均值
    data['mean_of_shopping'] = card_data[card_data['消费方式']=='超市'].groupby('studentid')['消费金额'].agg('mean')
    # 每个同学超市消费最大值
    data['max_of_shopping'] = card_data[card_data['消费方式'] == '超市'].groupby('studentid')['消费金额'].agg('max')
    # 每个同学超市消费偏度
    data['skew_of_shopping'] = card_data[card_data['消费方式'] == '超市'].groupby('studentid')['消费金额'].agg('skew') + 100
    # 每个同学超市消费峰度
    data['kurt_of_shopping'] = card_data[card_data['消费方式'] == '超市'].groupby('studentid')['消费金额'].agg(lambda x:x.kurt()) + 100
    #超市消费的天数
    data['day_of_shopping'] = card_data[card_data['消费方式']=='超市'].groupby('studentid')['day'].agg('nunique')
    #每个同学超市消费总额所占花销比例
    data['percent_of_shopping'] = data['sum_of_shopping']/data['sum_of_consume']

    # 清除洗衣房异常数据
    card_data_temp = card_data[~card_data['消费地点'].isin(['loc326'])]
    # 每个同学洗衣房消费总额
    data['sum_of_wash'] = card_data_temp[card_data_temp['消费方式'] == '洗衣房'].groupby('studentid')['消费金额'].agg('sum')
    # 每个同学洗衣房消费总次数
    data['count_of_wash'] = card_data_temp[card_data_temp['消费方式'] == '洗衣房'].groupby('studentid')['消费金额'].agg('count')
    # 每个同学洗衣房消费平均值
    data['mean_of_wash'] = card_data_temp[card_data_temp['消费方式'] == '洗衣房'].groupby('studentid')['消费金额'].agg('mean')
    # 每个同学洗衣房消费偏度
    data['skew_of_wash'] = card_data_temp[card_data_temp['消费方式'] == '洗衣房'].groupby('studentid')['消费金额'].agg('skew') + 100
    # 每个同学洗衣房消费峰度
    data['kurt_of_wash'] = card_data_temp[card_data_temp['消费方式'] == '洗衣房'].groupby('studentid')['消费金额'].agg(lambda x:x.kurt()) + 100
    # 洗衣房消费的天数
    data['day_of_wash'] = card_data_temp[card_data_temp['消费方式'] == '洗衣房'].groupby('studentid')['消费金额'].agg('mean')
    # 每个同学洗衣房消费总额所占花销比例
    data['percent_of_wash'] = data['sum_of_wash'] / data['sum_of_consume']

    #每个同学图书馆消费总额
    data['sum_of_library'] = card_data[card_data['消费方式']=='图书馆'].groupby('studentid')['消费金额'].agg('sum')
    #每个同学图书馆消费总次数
    data['count_of_library'] = card_data[card_data['消费方式']=='图书馆'].groupby('studentid')['消费金额'].agg('count')
    #每个同学图书馆消费平均值
    data['mean_of_library'] = card_data[card_data['消费方式']=='图书馆'].groupby('studentid')['消费金额'].agg('mean')
    # 每个同学图书馆消费最大值
    data['max_of_library'] = card_data[card_data['消费方式'] == '图书馆'].groupby('studentid')['消费金额'].agg('max')
    # 每个同学图书馆消费偏度
    data['skew_of_library'] = card_data[card_data['消费方式'] == '图书馆'].groupby('studentid')['消费金额'].agg('skew') + 100
    # 每个同学图书馆消费峰度
    data['kurt_of_shopping_of_library'] = card_data[card_data['消费方式'] == '图书馆'].groupby('studentid')['消费金额'].agg(lambda x:x.kurt()) + 100
    # 每个同学图书馆消费总额所占花销比例
    data['percent_of_library'] = data['sum_of_library']/data['sum_of_consume']

    # 每个同学文印中心消费总额
    data['sum_of_print'] = card_data[card_data['消费方式']=='文印中心'].groupby('studentid')['消费金额'].agg('sum')
    # 每个同学文印中心消费总次数
    data['count_of_print'] = card_data[card_data['消费方式']=='文印中心'].groupby('studentid')['消费金额'].agg('count')
    # 每个同学文印中心消费平均值
    data['mean_of_print'] = card_data[card_data['消费方式']=='文印中心'].groupby('studentid')['消费金额'].agg('mean')
    # 每个同学文印中心消费最大值
    data['max_of_print'] = card_data[card_data['消费方式'] == '文印中心'].groupby('studentid')['消费金额'].agg('max')
    # 每个同学文印中心消费总额所占花销比例
    data['percent_of_print'] = data['sum_of_print']/data['sum_of_consume']

    # 清除教务处异常数据
    card_data_temp = card_data[~card_data['消费地点'].isin(['loc27'])]
    # 每个同学教务处消费总额
    data['sum_of_dataing'] = card_data_temp[card_data_temp['消费方式']=='教务处'].groupby('studentid')['消费金额'].agg('sum')
    # 每个同学教务处消费总次数
    data['count_of_dataing'] = card_data_temp[card_data_temp['消费方式']=='教务处'].groupby('studentid')['消费金额'].agg('count')

    # 每个同学医院消费总额
    data['sum_of_hospital'] = card_data[card_data['消费方式']=='校医院'].groupby('studentid')['消费金额'].agg('sum')
    # 每个同学医院消费总次数
    data['count_of_hospital'] = card_data[card_data['消费方式']=='校医院'].groupby('studentid')['消费金额'].agg('count')

    # 每个同学其他消费总额
    data['sum_of_other'] = card_data[card_data['消费方式']=='其他'].groupby('studentid')['消费金额'].agg('sum')
    # 每个同学其他消费总次数
    data['count_of_other'] = card_data[card_data['消费方式']=='其他'].groupby('studentid')['消费金额'].agg('count')

    # 对比两年的变化
    data_temp1 = data[data.index%2==1]
    data_temp2 = data[data.index%2==0]
    data_temp1.index = data_temp1.index-1
    data_temp2.index = data_temp2.index+1
    data_temp = pd.concat([data_temp1,data_temp2])
    data_temp3 = (data+data_temp)/2
    # 计算第二年校园卡消费比第一年增加量
    data['difference_of_consume'] = data_temp['sum_of_consume']-data['sum_of_consume']
    # 计算第二年校园卡消费比第一年增加量所占百分比
    data['percent_difference_consume'] = data['difference_of_consume'] /(data_temp3['sum_of_consume']+1)
    # 计算第二年校园卡食堂消费比第一年增加量
    data['difference_of_meal'] = data_temp['sum_of_meal']-data['sum_of_meal']
    # 计算第二年校园卡食堂消费比第一年增加量所占百分比
    data['percent_difference_meal'] = data['difference_of_meal'] /(data_temp3['sum_of_meal']+1)
    # 计算第二年校园卡非食堂消费比第一年增加量
    data['difference_of_nomeal'] = data['difference_of_consume']-data['difference_of_meal']
    # 计算第二年校园卡非消费比第一年增加量所占百分比
    data['percent_difference_noconsume'] = data['difference_of_nomeal'] /(data_temp3['sum_of_consume']-data_temp3['sum_of_meal']+1)

    # 每个同学的图书借阅数量
    data['count_of_borrow'] = borrow_data.groupby('studentid')['time'].agg('count')
    # 每个同学借阅图书的天数
    data['day_of_borrow'] = borrow_data.groupby('studentid')['time'].agg('nunique')

    # 图书馆进出次数
    data['count_IO_library'] = library_data.groupby('studentid')['time'].agg('count')
    # 图书馆进出天数
    data['day_IO_library'] = library_data.groupby('studentid')['day'].agg('nunique')

    # 每个同学消费过的地点个数
    data['count_of_loc'] = card_data.groupby('studentid')['消费地点'].agg('nunique')
    # 添加地点消费次数特征
    data.reset_index(inplace=True)
    location = card_data.groupby(['studentid','消费地点'])['消费金额'].agg('sum').unstack()
    location.reset_index(inplace=True)
    data = pd.merge(data,location,on='studentid',how='left')

    # 添加图书馆门禁系统数据特征
    library = library_data.groupby(['studentid','doorid'])['doorid'].agg('count').unstack()
    library.reset_index(inplace=True)
    data = pd.merge(data,library,on='studentid',how='left')

    # 按季度统计消费特征
    card_data_temp = card_data[card_data['消费类别'] == 'POS消费'].copy()
    card_data_temp['quarter'] = card_data['消费时间'].apply(lambda x: x.quarter)
    # 统计每个季节的总消费
    sum_quarter_consume = pd.DataFrame(card_data_temp.groupby(['studentid', 'quarter'])['消费金额'].sum()).unstack()
    sum_quarter_consume.columns = ['sum_quar_1', 'sum_quar_2', 'sum_quar_3', 'sum_quar_4']
    sum_quarter_consume.reset_index(inplace=True)
    data = pd.merge(data, sum_quarter_consume, on='studentid', how='left')
    # 统计每个季节的平均消费
    count_quarter_consume = pd.DataFrame(card_data_temp.groupby(['studentid', 'quarter'])['消费金额'].count()).unstack()
    count_quarter_consume.columns = ['count_quar_1', 'count_quar_2', 'count_quar_3', 'count_quar_4']
    count_quarter_consume.reset_index(inplace=True)
    data = pd.merge(data, count_quarter_consume, on='studentid', how='left')
    # 统计每个季节的平均消费
    mean_quarter_consume = pd.DataFrame(card_data_temp.groupby(['studentid', 'quarter'])['消费金额'].mean()).unstack()
    mean_quarter_consume.columns = ['mean_quar_1', 'mean_quar_2', 'mean_quar_3', 'mean_quar_4']
    mean_quarter_consume.reset_index(inplace=True)
    data = pd.merge(data, mean_quarter_consume, on='studentid', how='left')

    # 按周统计消费特征
    card_data_temp['day_of_week'] = card_data_temp['消费时间'].apply(lambda x: x.dayofweek)
    data.set_index('studentid', inplace=True)
    data['percent_of_week'] = card_data_temp[card_data_temp['day_of_week'].isin([5, 6])].groupby('studentid')[
                                  '消费金额'].sum() / data['sum_of_consume']

    # 按天统计消费特征
    consume_of_day = card_data[card_data['消费类别']=='POS消费'].groupby(['studentid','day'])['消费金额'].agg('sum').unstack()
    # 最大值
    data['max_consum_day'] = consume_of_day.apply(lambda x: x.max(),axis=1)
    # 平均值
    data['mean_consum_day'] = consume_of_day.apply(lambda x: x.mean(), axis=1)
    # 标准差
    data['std_consum_day'] = consume_of_day.apply(lambda x: x.std(), axis=1)
    # 偏度
    data['skew_consum_day'] = consume_of_day.apply(lambda x: x.skew(), axis=1)
    # 峰度
    data['kurt_consum_day'] = consume_of_day.apply(lambda x: x.kurt(), axis=1)

    # 按周统计消费特征
    consume_of_week = card_data[card_data['消费类别'] == 'POS消费'].groupby(['studentid', 'week'])['消费金额'].agg('sum').unstack()
    # 最大值
    data['max_consum_week'] = consume_of_week.apply(lambda x: x.max(), axis=1)
    # 平均值
    data['mean_consum_week'] = consume_of_week.apply(lambda x: x.mean(), axis=1)
    # 标准差
    data['std_consum_week'] = consume_of_week.apply(lambda x: x.std(), axis=1)
    # 偏度
    data['skew_consum_week'] = consume_of_week.apply(lambda x: x.skew(), axis=1)
    # 峰度
    data['kurt_consum_week'] = consume_of_week.apply(lambda x: x.kurt(), axis=1)

    # 按月统计消费特征
    consume_of_month = card_data[card_data['消费类别'] == 'POS消费'].groupby(['studentid', 'month'])['消费金额'].agg( 'sum').unstack()
    # 最大值
    data['max_consum_month'] = consume_of_month.apply(lambda x: x.max(), axis=1)
    # 平均值
    data['mean_consum_month'] = consume_of_month.apply(lambda x: x.mean(), axis=1)
    # 标准差
    data['std_consum_month'] = consume_of_month.apply(lambda x: x.std(), axis=1)
    # 偏度
    data['skew_consum_month'] = consume_of_month.apply(lambda x: x.skew(), axis=1)
    # 峰度
    data['kurt_consum_month'] = consume_of_month.apply(lambda x: x.kurt(), axis=1)

    # 按季度统计消费特征
    consume_of_quarter = card_data[card_data['消费类别'] == 'POS消费'].groupby(['studentid', 'quarter'])['消费金额'].agg('sum').unstack()
    # 最大值
    data['max_consum_quarter'] = consume_of_quarter.apply(lambda x: x.max(), axis=1)
    # 平均值
    data['mean_consum_quarter'] = consume_of_quarter.apply(lambda x: x.mean(), axis=1)
    # 标准差
    data['std_consum_quarter'] = consume_of_quarter.apply(lambda x: x.std(), axis=1)
    # 偏度
    data['skew_consum_quarter'] = consume_of_quarter.apply(lambda x: x.skew(), axis=1)
    # 峰度
    data['kurt_consum_quarter'] = consume_of_quarter.apply(lambda x: x.kurt(), axis=1)


    # 按天统计消费特征
    meal_of_day = card_data[card_data['消费方式'] == '食堂'].groupby(['studentid', 'day'])['消费金额'].agg('sum').unstack()
    # 最大值
    data['max_meal_day'] = meal_of_day.apply(lambda x: x.max(), axis=1)
    # 平均值
    data['mean_meal_day'] = meal_of_day.apply(lambda x: x.mean(), axis=1)
    # 标准差
    data['std_meal_day'] = meal_of_day.apply(lambda x: x.std(), axis=1)
    # 偏度
    data['skew_meal_day'] = meal_of_day.apply(lambda x: x.skew(), axis=1)
    # 峰度
    data['kurt_meal_day'] = meal_of_day.apply(lambda x: x.kurt(), axis=1)

    # 按周统计消费特征
    meal_of_week = card_data[card_data['消费类别'] == 'POS消费'].groupby(['studentid', 'week'])['消费金额'].agg(
        'sum').unstack()
    # 最大值
    data['max_meal_week'] = meal_of_week.apply(lambda x: x.max(), axis=1)
    # 平均值
    data['mean_meal_week'] = meal_of_week.apply(lambda x: x.mean(), axis=1)
    # 标准差
    data['std_meal_week'] = meal_of_week.apply(lambda x: x.std(), axis=1)
    # 偏度
    data['skew_meal_week'] = meal_of_week.apply(lambda x: x.skew(), axis=1)
    # 峰度
    data['kurt_meal_week'] = meal_of_week.apply(lambda x: x.kurt(), axis=1)

    # 按月统计消费特征
    meal_of_month = card_data[card_data['消费类别'] == 'POS消费'].groupby(['studentid', 'month'])['消费金额'].agg(
        'sum').unstack()
    # 最大值
    data['max_meal_month'] = meal_of_month.apply(lambda x: x.max(), axis=1)
    # 平均值
    data['mean_meal_month'] = meal_of_month.apply(lambda x: x.mean(), axis=1)
    # 标准差
    data['std_meal_month'] = meal_of_month.apply(lambda x: x.std(), axis=1)
    # 偏度
    data['skew_meal_month'] = meal_of_month.apply(lambda x: x.skew(), axis=1)
    # 峰度
    data['kurt_meal_month'] = meal_of_month.apply(lambda x: x.kurt(), axis=1)

    # 按季度统计消费特征
    meal_of_quarter = card_data[card_data['消费类别'] == 'POS消费'].groupby(['studentid', 'quarter'])['消费金额'].agg(
        'sum').unstack()
    # 最大值
    data['max_meal_quarter'] = meal_of_quarter.apply(lambda x: x.max(), axis=1)
    # 平均值
    data['mean_meal_quarter'] = meal_of_quarter.apply(lambda x: x.mean(), axis=1)
    # 标准差
    data['std_meal_quarter'] = meal_of_quarter.apply(lambda x: x.std(), axis=1)
    # 偏度
    data['skew_meal_quarter'] = meal_of_quarter.apply(lambda x: x.skew(), axis=1)
    # 峰度
    data['kurt_meal_quarter'] = meal_of_quarter.apply(lambda x: x.kurt(), axis=1)

    data.reset_index(inplace=True)

    return data

