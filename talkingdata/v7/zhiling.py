import pandas as pd
import datetime

data_path = 'C:/Users/cui/Desktop/python/talkingdata/data/'

test = pd.read_csv(data_path + 'test.csv')
submission = pd.read_csv(r'C:\Users\cui\Desktop\python\talkingdata\submission\sub20180508_044911.csv')
submission.info()
print('0的个数为：{}'.format(sum(submission['is_attributed']==0)))

test = test.merge(submission,on='click_id',how='left')
test = test.merge(test.groupby(['ip','device','os','channel','app','click_time'],as_index=False)['click_id'].agg({'max_rank':'max'}),
                  on=['ip','device','os','channel','app','click_time'],how='left')
test = test.merge(test.groupby(['ip','device','os','channel','app','click_time'],as_index=False)['is_attributed'].agg({'max_pred':'max'}),
                  on=['ip','device','os','channel','app','click_time'],how='left')
test['flag'] = (test['click_id']==test['max_rank']).astype(int)
test['is_attributed'] = test['max_pred']*test['flag']

submission = test[['click_id','is_attributed']]

submission.info()
print('最大值为：{}'.format(submission['is_attributed'].max()))
print('最小值为：{}'.format(submission['is_attributed'].min()))
print('空值个数为：{}'.format(submission['is_attributed'].isnull().sum()))
print('0的个数为：{}'.format(sum(submission['is_attributed']==0)))
submission.to_csv('C:/Users/cui/Desktop/python/talkingdata/submission/sub{}.csv'.format(
    datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), index=False,float_format='%.8f')




