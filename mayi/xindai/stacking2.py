import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression



piupiu_11_pred = pd.read_csv(r'C:\Users\csw\Desktop\python\piupiu_11_pred.csv')
piupiu_12_pred = pd.read_csv(r'C:\Users\csw\Desktop\python\piupiu_12_pred.csv')
wenchao_11_pred = pd.read_csv(r'C:\Users\csw\Desktop\python\wenchao_11_pred.csv')
wenchao_12_pred = pd.read_csv(r'C:\Users\csw\Desktop\python\wenchao_12_pred.csv')
lida_11_pred = pd.read_csv(r'C:\Users\csw\Desktop\python\lida_11_pred.csv')
lida_12_pred = pd.read_csv(r'C:\Users\csw\Desktop\python\lida_12_pred.csv')

train_pred = piupiu_11_pred.merge(wenchao_11_pred,on='uid',how='left')
train_pred = train_pred.merge(lida_11_pred,on='uid',how='left')

test_pred = piupiu_12_pred.merge(wenchao_12_pred,on='uid',how='left')
test_pred = test_pred.merge(lida_12_pred,on='uid',how='left')


models = [f for f in train_pred.columns if f not in ['uid','loan_sum']]

lr = LinearRegression()
lr.fit(train_pred[models], train_pred.loan_sum)

train_final_pred = lr.predict(train_pred[models])

test_pred_temp = test_pred.copy()
test_pred_temp = test_pred_temp - test_pred_temp.mean() + 1.247
test_final_pred = lr.predict(test_pred_temp[models])


submission = pd.DataFrame({'uid':test_pred.uid.values,'pred':test_final_pred})[['uid','pred']]
submission['pred'] = submission['pred'].apply(lambda x: x if x>0 else 0)
submission.to_csv(r'C:\Users\csw\Desktop\python\JD\xindai\submission\sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
                  index=False, header=None, float_format='%.4f')
