import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

data_path = 'C:/Users/csw/Desktop/python/mayi/data/eval/'
test_path = data_path + 'evaluation_public.csv'
shop_path = data_path + 'ccf_first_round_shop_info.csv'
train_path = data_path + 'ccf_first_round_user_shop_behavior.csv'

def acc(data,name='shop_id'):
    true_path = data_path + 'true.pkl'
    true = None
    try:
        true = pickle.load(open(true_path,'+rb'))
    except:
        print('没有发现真实数据，无法测评')
    return sum(data['row_id'].map(true)==data[name])/data.shape[0]

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

#构造规则
wifi_to_shops = defaultdict(lambda : defaultdict(lambda :0))
for line in train.values:
    wifi = sorted([wifi.split('|') for wifi in line[5].split(';')],key=lambda x:int(x[1]),reverse=True)[0]
    wifi_to_shops[wifi[0]][line[1]] = wifi_to_shops[wifi[0]][line[1]] + 1

result = []
for line in test['wifi_infos'].values:
    wifi = sorted([wifi.split('|') for wifi in line.split(';')],key=lambda x:int(x[1]),reverse=True)[0]
    counter = defaultdict(lambda : 0)
    for k,v in wifi_to_shops[wifi[0]].items():
        counter[k] += v
    try:
        pred = sorted(counter.items(),key=lambda x:x[1],reverse=True)[0][0]
    except:
        pred = np.nan
    result.append(pred)
test['pred'] = result
print(acc(test,'pred'))

# #预测
# preds = []
# for line in test.values:
#     index = 0
#     while True:
#         try:
#             if index==5:
#                 pred_one = None
#                 break
#             wifi = sorted([wifi.split('|') for wifi in line[6].split(';')],key=lambda x:int(x[1]),reverse=True)[index]
#             counter = defaultdict(lambda : 0)
#             for k,v in wifi_to_shops[wifi[0]].items():
#                 counter[k] += v
#             pred_one = sorted(counter.items(),key=lambda x:x[1],reverse=True)[0][0]
#             break
#         except:
#             index+=1
#     preds.append(pred_one)
#
# result = pd.DataFrame({'row_id':test.row_id,'shop_id':preds})
# result.fillna('s_666').to_csv('wifi_baseline.csv',index=None) #随便填的 这里还能提高不少
#








