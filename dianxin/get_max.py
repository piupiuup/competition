from tool.tool import *
from dianxin.feat1 import *

data_path = 'C:/Users/cui/Desktop/python/dianxin/data/'

cc = ['service_type', 'is_mix_service','is_promise_low_consume',
'net_service', 'gender', 'age', 'online_time','contract_type',
'1_total_fee','2_total_fee', '3_total_fee', '4_total_fee',
'month_traffic','last_month_traffic', 'local_trafffic_month',
'local_caller_time', 'service1_caller_time','service2_caller_time',
'many_over_bill',  'contract_time','pay_times', 'pay_num']
print('读取train数据...')
train = pd.read_csv(data_path + 'train.csv')
# test = pd.read_csv(data_path + 'test.csv')
train_new = pd.read_csv( 'C:/Users/cui/Desktop/python/dianxin/data/b/' + 'train_new.csv')
test_new = pd.read_csv( 'C:/Users/cui/Desktop/python/dianxin/data/b/' + 'train_new.csv')
data = train.append(train_new).append(test_new)
data = data.drop_duplicates(cc)

fees = pd.DataFrame(columns=['first','second'])
for i in ['1_total_fee','2_total_fee', '3_total_fee', '4_total_fee']:
    for j in ['1_total_fee','2_total_fee', '3_total_fee', '4_total_fee']:
        if i == j:
            pass
        else:
            temp = data[[i,j]]
            temp.columns = ['first','second']
            fees = fees.append(temp)
temp3 = fees.groupby('first',as_index=False)['second'].median()
fees = fees[fees['first']!=fees['second']]
temp = fees.groupby(['first','second'],as_index=False)['first'].agg({'count':'size'})
temp['rate'] = temp['count'] / temp['second'].map(temp.groupby('second')['count'].sum())
temp = temp[temp['count']>2]
temp1 = temp.sort_values('count',ascending=False).drop_duplicates('first',keep='first')[['first','second']]
temp2 = temp.sort_values('rate',ascending=False).drop_duplicates('first',keep='first')[['first','second']]
temp1.columns = ['first','second_count']
temp2.columns = ['first','second_rate']
temp3.columns = ['first','second_median']

temp1.to_csv(data_path+'get_max_count.csv',index=False)
temp2.to_csv(data_path+'get_max_rate.csv',index=False)
temp3.to_csv(data_path+'get_max_median.csv',index=False)


