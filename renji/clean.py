import numpy as np
import pandas as pd

test_path = r'C:\Users\csw\Desktop\python\360\data2\evaluation_public.tsv'
train_path = r'C:\Users\csw\Desktop\python\360\data2\train.tsv'
train_hdf_path = r'C:\Users\csw\Desktop\python\360\data2\train.hdf'
test_hdf_path = r'C:\Users\csw\Desktop\python\360\data2\evaluation_public.hdf'
cache_path = 'F:/360_cache/'
new = True

train = pd.read_csv(train_path,sep='\\t',header=None,encoding="utf-8",names=['id','headline','content','label'])
train = train[['id','label','headline','content']]
train.fillna('',inplace=True)
test = pd.read_csv(test_path,sep='\\t',header=None,encoding="utf-8",names=['id','headline','content'])
test['label'] = np.nan
test.fillna('',inplace=True)
train.to_hdf(train_hdf_path,'w', complib='blosc', complevel=5)
test.to_hdf(test_hdf_path,'w', complib='blosc', complevel=5)