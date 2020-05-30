import pandas as pd
import numpy as np
import os,random,gc, re
np.random.seed(2017)
random.seed(2017)

import gensim
import math,jieba
import datetime
import sys,logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.60
set_session(tf.Session(config=config))

from keras.layers import Input, Embedding, LSTM, Dense,Flatten, Dropout, merge, Bidirectional,concatenate,GRU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import  MaxPooling2D,GlobalAveragePooling1D,AveragePooling1D,GlobalMaxPooling1D
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D,Merge
from keras.layers import Embedding
#from keras.layers import GlobalAveragePooling1D
from keras.layers import Convolution1D
from keras.layers import Dense, Dropout, Activation
from keras import backend as K
from keras.optimizers import *
from keras.callbacks import Callback
from keras.regularizers import l1,l2
from sklearn.cross_validation import train_test_split
from JoinAttLayer import Attention

from keras.engine.topology import Layer,InputSpec
from keras.utils import conv_utils


vec_path = r'C:\Users\cui\Desktop\python\toxic\bird\vec'
valid1_path = r'C:\Users\cui\Desktop\python\toxic\bird\valid1.csv'
valid_test_path = r'C:\Users\cui\Desktop\python\toxic\bird\validtest.csv'
output_path = 'C:/Users/cui/Desktop/python/toxic/bird/output/'


embedw = []
#embedw.append([0] * 48)
#embedw.append([0] * 48)


for line in open(vec_path,encoding='utf-8'):
    if not line:
        continue
    array = line.strip().split(' ',-1)
    #cache[int(array[0])] = map(float,array[1:])
    embedw.append(list(map(float,array[1:])))
embedw = np.array(embedw)
wordnum=embedw.shape[0]
print(wordnum)

FEAT_LENGTH = 270
FIX_LEN=20


feat = []
id = []
featlen = []
featupper = []
y = []
for line in open(valid_test_path):
    if not line:
        continue
    array = list(map(float,line.strip().split(',')[1:]))
    y.append(array[0:6])
    id.append(line.strip().split(',')[0])
    #feat.append(array[14:][:FEAT_LENGTH-FIX_LEN])
    base = array[6:][:FEAT_LENGTH - FIX_LEN]
    feat.append(list(map(lambda x: x % 100000, base)))
    #featupper.append(map(lambda x: int((x % 1000000) / 100000), base))

id = np.array(id)
y = np.array(y)
feat = np.array(feat)
#featlen = np.array(featlen)
#featupper = np.array(featupper)

test_feat = feat
#test_featlen = featlen
#test_featupper = featupper
test_y = y

input3 = Input(shape=(FEAT_LENGTH-FIX_LEN,), dtype='int32')
embedding_layer0 = Embedding(wordnum,
                        50,
                        trainable=True)
x30 = embedding_layer0(input3)
la = []
t3 = Conv1D(25,3,activation='sigmoid',padding='same',dilation_rate = 1)
x3 = t3(x30)
x3 = GlobalMaxPooling1D()(x3)
la.append(x3)
t2 = Conv1D(20,1,activation='sigmoid',padding='same',dilation_rate = 1)
x3 = t2(x30)
x3 = GlobalMaxPooling1D()(x3)
la.append(x3)
t=Conv1D(30,2,activation='sigmoid',padding='same',dilation_rate = 1)
x3 = t(x30)
x3 = GlobalMaxPooling1D()(x3)
la.append(x3)
#x3 = Bidirectional(LSTM(5))(x31)
#la.append(x3)
#x30 = Cropping1D(cropping=(35,85))(x30)
embedding_layer1 = Embedding(wordnum,
                        300,
                             trainable=True)
x30 = embedding_layer1(input3)
#x30 = concatenate([x30,x311],axis = 2)
x30 = LSTM(300,dropout = 0.50,recurrent_dropout=0.40,return_sequences=True)(x30)
x30 = Dropout(0.25)(x30)
att = Attention(50)
x1 = att(x30)


la = []

embedding_layer12 = Embedding(wordnum,
                             300,
                             trainable=True)
x31 = embedding_layer12(input3)
x = Bidirectional(GRU(60, return_sequences=True))(x31)
avg_pool = GlobalAveragePooling1D()(x)
max_pool = GlobalMaxPooling1D()(x)
la.append(avg_pool)
la.append(max_pool)

x1 = BatchNormalization()(x1)

	#x1 = Dropout(0.1)(x1)
x = Dense(128, activation='sigmoid')(x1)
x3 = Dense(128)(x1)
x3 = PReLU()(x3)
x1 = merge([x,x3], mode='concat')
x1 = Dropout(0.1)(x1)
x = Dense(40, activation='sigmoid')(x1)
x3 = Dense(40)(x1)
x3 = PReLU()(x3)
x1 = merge([x,x3], mode='concat')
x1 = Dropout(0.1)(x1)
la.append(x1)
x1 = merge(la,mode = 'concat')
out = Dense(6, activation='sigmoid')(x1)


model = Model(input=[input3], output=out)



embedding_layer1.set_weights([embedw])
# embedding_layer0.trainable=False
model.compile(loss='mse',
              optimizer=Adam(lr=0.0010,epsilon=1e-4))

model.summary()
'''
model.load_weights('./modelss2.h5')
model.load_weights('./w32')
res = model.predict([test_feat],batch_size = 1028)

f = open('./res_'+str(datetime.datetime.now()).replace(" ","_")+'.csv','w')
print >>f ,"id,toxic,severe_toxic,obscene,threat,insult,identity_hate"
for i in range(res.shape[0]):
    print >>f,str(id[i])+","+",".join(map(lambda x:str(round(x,5)),res[i]))
'''

res = []
for i in range(1,7):
    model.load_weights('./modelk6'+str(i))
    model.load_weights('./w3k6'+str(i))
    res.append(model.predict([test_feat],batch_size = 1028))

resf = sum(res)/6
f = open('./res_'+str(datetime.datetime.now()).replace(" ","_").replace(':','_')[:19]+'.csv','w')
f.write("id,toxic,severe_toxic,obscene,threat,insult,identity_hate\r")
for i in range(resf.shape[0]):
    f.write(str(id[i])+","+",".join(list(map(lambda x:str(round(x,5)),resf[i])))+'\r')
# print >>f ,"id,toxic,severe_toxic,obscene,threat,insult,identity_hate"
# for i in range(resf.shape[0]):
#     print >>f,str(id[i])+","+",".join(list(map(lambda x:str(round(x,5)),resf[i])))
