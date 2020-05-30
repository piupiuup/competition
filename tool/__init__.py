import numpy as np
import pandas as pd




#判断df中是否还有空值
def is_null(data,value=np.nan):
    df_of_1 = pd.DataFrame(np.ones(data.shape))
    s1 = df_of_1[data.isnull()].sum().sum()
    print ('数据中包含'+str(s1)+'个空值')
