import pandas as pd
from joblib import Parallel, delayed
import itertools
import gc

def make_set(a):
    print(a)

def execu(func):
    print (func)
    x = func[0](*func[1])
    del x; gc.collect()
    print (func, 'done')

def applyParallel_feature(funcs, execu, num_threads):
    ''' 利用joblib来并行提取特征
    '''
    with Parallel(n_jobs=num_threads) as parallel:
        retLst = parallel( delayed(execu)(func) for func in funcs )
        return None

def applyParallel(dfGrouped, func, num_threads):
    '''利用joblib来生成特征样本
    '''
    with Parallel(n_jobs=num_threads) as parallel:
        retLst = parallel( delayed(func)(*group) for group in dfGrouped )
        return pd.concat(retLst)

# funcs_list = []
# for date, prediction_type in [1,2,3]:
#     print(date, prediction_type)
#     funcs_list.append([make_set, (date, prediction_type)])
#
# applyParallel_feature(iter(funcs_list), execu, 3)
from math import sqrt
from joblib import Parallel, delayed
if __name__ == '__main__':
  print(Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10)))
