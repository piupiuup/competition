import pandas as pd
import numpy as np
import datetime

chizi = pd.read_csv(r'C:\Users\cui\Desktop\python\talkingdata\submission\sub20180430_105854.csv')    # lgb 9825
chizi2 = chizi[['click_id']].merge(pd.read_csv(r'C:\Users\cui\Desktop\python\talkingdata\submission\sub20180506_125837.csv'),on='click_id',how='left') # lgb 9824
chizi3 = chizi[['click_id']].merge(pd.read_csv(r'C:\Users\cui\Desktop\python\talkingdata\submission\sub20180507_070704.csv'),on='click_id',how='left') # nn 9818
chizi4 = chizi[['click_id']].merge(pd.read_csv(r'C:\Users\cui\Desktop\python\talkingdata\submission\sub20180507_121831_nn2.csv'),on='click_id',how='left') # nn 9818 未提交
chizi5 = chizi[['click_id']].merge(pd.read_csv(r'C:\Users\cui\Desktop\python\talkingdata\submission\sub20180508_035902.csv'),on='click_id',how='left') # 9818未提交
# plantsgo = chizi[['click_id']].merge(pd.read_csv(r'C:\Users\cui\Downloads\all_model.csv'),on='click_id',how='left')
# feiyang = chizi[['click_id']].merge(pd.read_csv(r'C:\Users\cui\Downloads\merge_lgb_nn_7_3.csv'),on='click_id',how='left')

for sub in [chizi,chizi2,chizi3,chizi4,chizi5]:
    m = sub['is_attributed'].median()
    print('最大值为：{}'.format(sub['is_attributed'].max()))
    print('最小值为：{}'.format(sub['is_attributed'].min()))
    print('空值个数为：{}'.format(sub['is_attributed'].isnull().sum()))
    print('中位数为：{}'.format(m))
    sub['is_attributed'] = sub['is_attributed'] / m
for sub in [chizi,chizi2,chizi3,chizi4,chizi5]:
    m = sub['is_attributed'].median()
    print('中位数为：{}'.format(m))

submission = chizi[['click_id']].copy()
submission['is_attributed'] = 3*(0.0*chizi['is_attributed'] + 0.0*chizi2['is_attributed']
                               + 0.1 * chizi3['is_attributed'] + 0.1*chizi4['is_attributed']+ 0.1*chizi5['is_attributed'])
submission['is_attributed'] = submission['is_attributed']/(submission['is_attributed'].max()+1)
submission['is_attributed'] = np.clip(submission['is_attributed'], 0, 1)
print('最大值为：{}'.format(submission['is_attributed'].max()))
print('最小值为：{}'.format(submission['is_attributed'].min()))
print('空值个数为：{}'.format(submission['is_attributed'].isnull().sum()))
submission.to_csv('C:/Users/cui/Desktop/python/talkingdata/submission/sub{}.csv'.format(
    datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), index=False,float_format='%.8f')




