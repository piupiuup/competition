import re
import sys
import jieba
import codecs
import os.path
import numpy as np
import pandas as pd
from tqdm import tqdm


test_path = r'C:\Users\csw\Desktop\python\360\data2\evaluation_public.tsv'
train_path = r'C:\Users\csw\Desktop\python\360\data2\train.tsv'
cache_path = 'F:/360_cache/'

train = pd.read_csv(train_path,sep='\\t',header=None,encoding="utf-8",names=['id','headline','content','label'])
train = train[['id','label','headline','content']]
train.fillna('',inplace=True)


content = ''.join(str(x) for x in train['content'].values)
se = set(content)
di = dict([(x,0) for x in se])
for i in tqdm(content):
    di[i] += 1
result = pd.Series(di)
result.sort_values(ascending=False,inplace=True)









for x in s:
    L[index] += x
    if index == 0:
        step = 1
    elif index == numRows -1:
        step = -1
    index += step

