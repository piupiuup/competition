import numpy as np
import pandas as pd
from collections import Counter

data_path = 'C:/Users/csw/Desktop/python/JD/xindai/data/'
user_path = data_path + 't_user.csv'
order_path = data_path + 't_order.csv'
click_path = data_path + 't_click.csv'
loan_path = data_path + 't_loan.csv'
loan_sum_path = data_path + 't_loan_sum.csv'
bt_loan_path = data_path + 't_bt_loan.csv'

user = pd.read_csv(user_path)
order = pd.read_csv(order_path)
click = pd.read_csv(click_path)
loan = pd.read_csv(loan_path)
bt_loan = pd.read_csv(bt_loan_path)
loan_sum = pd.read_csv(loan_sum_path)

# 对金额进行处理
user['limit'] = np.round(5**(user['limit'])-1,2)
order['price'] = np.round(5**(order['price'])-1,2)
order['discount'] = np.round(5**(order['discount'])-1,2)
loan['loan_amount'] = np.round(5**(loan['loan_amount'])-1,2)
loan_sum['loan_sum'] = np.round(5**(loan_sum['loan_sum'])-1,2)

a = loan[loan['loan_time']>='2016-11-03 00:00:00'].groupby('uid',as_index=False)['loan_amount'].sum()
b = loan_sum.merge(a,on='uid',how='outer')
c = b[b['loan_sum']!=b['loan_amount']]
print(c.shape)


print('用户个数：{}'.format(user.shape[0]))
print('用户行为次数：{}'.format(order.shape[0]))
print('点击次数：{}'.format(click.shape[0]))
print('借贷次数：{}'.format(loan.shape[0]))
print('总借贷次数：{}'.format(loan_sum.shape[0]))


print('有用户行为的用户个数：{}'.format(order['uid'].nunique()))
print('有用户行为的用户个数：{}'.format(order['uid'].nunique()))

# 分类模型统计分析
def analyse(data,name,label='label'):
    result = data.groupby(name)[label].agg({'count':'count','sum':'sum'})
    result['rate'] = result['sum']/result['count']
    return result

# 计算pid_param转化率
click['pid_param'] = click['pid'].astype(str)+'_'+click['param'].astype(str)
data = click[click['click_time']<'2016-11-01'][['uid','pid_param']].drop_duplicates()
data = data.merge(loan_sum[['uid','loan_sum']],on='uid',how='left')
a = analyse(data,'pid_param','loan_sum')












