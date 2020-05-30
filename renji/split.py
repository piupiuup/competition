import numpy as np
import pandas as pd

data1_path = 'C:/Users/csw/Desktop/python/360/data/'
data2_path = 'C:/Users/csw/Desktop/python/360/data2/'
eval_path = 'C:/Users/csw/Desktop/python/360/data2/eval/'
add_path = 'C:/Users/csw/Desktop/python/360/data2/add/'
train_path = 'train.tsv'
test_path = 'evaluation_public.tsv'

train_hdf_path = 'train.hdf'
test_hdf_path = 'evaluation_public.hdf'
cache_path = 'F:/360_cache/'
new = True

train1 = pd.read_csv(data1_path + train_path,sep='\\t',header=None,encoding="utf-8",names=['id','headline','content','label'])
train2 = pd.read_csv(data2_path + train_path,sep='\\t',header=None,encoding="utf-8",names=['id','headline','content','label'])
test1 = pd.read_csv(data1_path + test_path,sep='\\t',header=None,encoding="utf-8",names=['id','headline','content'])
test2 = pd.read_csv(data2_path + test_path,sep='\\t',header=None,encoding="utf-8",names=['id','headline','content'])

new_train_id = set(train2['id'].tolist())-set(train1['id'].tolist())
new_test_id = set(test2['id'].tolist())-set(test1['id'].tolist())
train_add = train2[train2['id'].isin(new_train_id)]
test_add = test2[test2['id'].isin(new_test_id)]
train_add = train_add[['id','label','headline','content']]
train_add.fillna('',inplace=True)
test_add.fillna('',inplace=True)

train_add.to_csv(add_path + train_path,header=None,index=False)
test_add.to_csv(add_path + test_path,header=None,index=False)


eval_data = train2[(train2['label']=='POSITIVE') | (train2['id'].isin(new_train_id))]
eval_data.to_hdf(eval_path+train_hdf_path, 'w', complib='blosc', complevel=5)
