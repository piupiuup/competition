import numpy as np
import pandas as pd
from xiecheng.tool import *

#寻找最佳系数用的线下测试
def  test(data, n=12, n_del=5, c=0.87):
    #n 算均值所使用的月份
    #n_del 删除掉的增长的月份
    #c 衰减系数

    def predict(data):
        result = []
        for name,arr in data.iloc[:,:-1].iterrows():
            arr_temp = []
            for i in range(len(arr)):
                if arr[i] == 0:
                    continue
                else:
                    arr_temp = arr[(i + 1):]
                    break
            if len(arr_temp) == 0 or (len(arr_temp) < 5 and np.max(arr_temp) < 170):
                result.append(10)
            elif len(arr_temp) < n_del:
                result.append((np.max(arr_temp) + arr_temp[-1]) / 2.)
            else:
                arr_temp = arr_temp[(n_del - 1):]
                arr_temp = arr_temp[-n:]
                coef = 1
                sum = 0.
                sum_of_coef = 0.
                for i in list(reversed(range(-len(arr_temp), 0))):
                    sum += arr_temp[i] * coef
                    sum_of_coef += coef
                    coef = coef * c
                ave = sum / sum_of_coef
                result.append(ave)

        return pd.DataFrame(result)


    #算分
    def calculate(y_true,y_pred):
        k = y_pred.values.mean()/y_true.values.mean()
        y_pred = y_pred/k
        return (((y_true.values-y_pred.values)*(y_true.values-y_pred.values)).sum()/np.sum(y_pred.shape))**0.5

    y_pred = predict(data.iloc[:,:-1])
    return calculate(data.iloc[:,-1:],y_pred)

def adjust(arr):
    n_del = 3
    n = 12
    c = 0.87
    valve = 170
    arr_temp = []
    for i in range(len(arr)):
        if arr[i]==0:
            continue
        else:
            arr_temp = arr[(i+1):]
            break
    if len(arr_temp)==0 or (len(arr_temp) < n_del and np.max(arr_temp)<valve):
        return 130
    elif len(arr_temp) < n_del:
        return (np.max(arr_temp)+arr_temp[-1])/2.
    else:
        arr_temp = arr_temp[(n_del-1):]
        arr_temp = arr_temp[-n:]
        coef = 1
        sum = 0.
        sum_of_coef = 0.
        for i in list(reversed(range(-len(arr_temp),0))):
            sum += arr_temp[i]*coef
            sum_of_coef += coef
            coef = coef*c
        ave = sum/sum_of_coef
        return ave

n = 11
result2 = (product_quantity_unstack.apply(lambda x: adjust(x),axis=1)).reset_index()
result2.columns = ['product_id','ciiquantity_month']
result = pd.merge(sample_result.iloc[:,:2],result2,on=['product_id'],how='outer').replace(0,100)
result['ciiquantity_month'] = result['ciiquantity_month'].astype('int')
result.to_csv(r'C:\Users\CSW\Desktop\python\xiecheng\submit\0321(4).csv',index=False)