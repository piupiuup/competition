
import pandas as pd
import numpy as np
import os, random, gc, re

np.random.seed(2017)
random.seed(2017)

import gensim
import math, jieba
import datetime
import sys, logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.80
set_session(tf.Session(config=config))

from keras.layers import Input, Embedding, LSTM, Dense, Flatten, Dropout, merge, Bidirectional, concatenate, GRU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling1D, AveragePooling1D, GlobalMaxPooling1D
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Merge
from keras.layers import Embedding
# from keras.layers import GlobalAveragePooling1D
from keras.layers import Convolution1D
from keras.layers import Dense, Dropout, Activation
from keras import backend as K
from keras.optimizers import *
from keras.callbacks import Callback
from keras.regularizers import l1, l2
from sklearn.cross_validation import train_test_split
from JoinAttLayer import Attention

from keras.engine.topology import Layer, InputSpec
from keras.utils import conv_utils
from sklearn.model_selection import KFold


vec_path = r'C:\Users\cui\Desktop\python\toxic\bird\vec'
valid1_path = r'C:\Users\cui\Desktop\python\toxic\bird\valid1.csv'
output_path = 'C:/Users/cui/Desktop/python/toxic/bird/output/'

embedw = []
# embedw.append([0] * 48)
# embedw.append([0] * 48)

wordnum = 30329
wordnum = 33516
for line in open(vec_path,encoding='utf-8'):
    if not line:
        continue
    array = line.strip().split(' ', -1)
    # cache[int(array[0])] = map(float,array[1:])
    embedw.append(list(map(float, array[1:])))
embedw = np.array(embedw)
print(embedw)
wordnum = embedw.shape[0]
print(wordnum)

FEAT_LENGTH = 270
FIX_LEN = 20

feat = []
id = []
featlen = []
featupper = []
y = []
'''
for line in open('./valid2.csv'):
    if not line:
        continue
    array = map(float,line.strip().split(',')[1:])
    y.append(array[0:6])
    id.append(line.strip().split(',')[0])
    #feat.append(array[14:][:FEAT_LENGTH-FIX_LEN])
    base = array[6:][:FEAT_LENGTH - FIX_LEN]
    feat.append(map(lambda x: x % 100000, base))
'''
start = len(y)
for line in open(valid1_path,encoding='utf-8'):
    if not line:
        continue
    array = list(map(float, line.strip().split(',')[1:]))
    y.append(array[0:6])
    id.append(line.strip().split(',')[0])
    # featlen.append(array[6:13])
    # feat.append(array[14:][:FEAT_LENGTH-FIX_LEN])
    base = array[6:][:FEAT_LENGTH - FIX_LEN]
    feat.append(list(map(lambda x: x % 100000, base)))

id = np.array(id)
y = np.array(y)
feat = np.array(feat)

v2 = range(start, len(y))
kf = KFold(n_splits=10)
i = 0
for train_index, test_index in kf.split(v2):
    i += 1
    if i >= 7:
        continue
    train_index = train_index + start
    test_index = test_index + start
    # train_index = np.concatenate((np.array(random.sample(range(start),start/3)),train_index),axis=0)
    print(train_index, test_index)
    train_feat = feat[train_index, :]
    train_y = y[train_index]
    #train_y[:, 0] = train_y[:, 0] * 0.8 + train_y[:, 1] * 0.2
    test_feat = feat[test_index, :]
    test_y = y[test_index]

    input3 = Input(shape=(FEAT_LENGTH - FIX_LEN,), dtype='int32')
    embedding_layer1 = Embedding(wordnum,
                                 300,
                                 trainable=True)
    x30 = embedding_layer1(input3)
    x30 = LSTM(300, dropout=0.50, recurrent_dropout=0.40, return_sequences=True)(x30)
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

    # x1 = Dropout(0.1)(x1)
    x = Dense(128, activation='sigmoid')(x1)
    x3 = Dense(128)(x1)
    x3 = PReLU()(x3)
    x1 = merge([x, x3], mode='concat')
    x1 = Dropout(0.1)(x1)
    x = Dense(40, activation='sigmoid')(x1)
    x3 = Dense(40)(x1)
    x3 = PReLU()(x3)
    x1 = merge([x, x3], mode='concat')
    x1 = Dropout(0.1)(x1)
    la.append(x1)
    x1 = merge(la, mode='concat')
    out = Dense(6, activation='sigmoid')(x1)

    model = Model(input=[input3], output=out)

    embedding_layer1.set_weights([embedw])
    embedding_layer12.set_weights([embedw])
    # embedding_layer0.trainable=False
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=0.0010, epsilon=5e-5))

    # model.summary()

    from keras.callbacks import EarlyStopping, ModelCheckpoint

    STAMP = 'modelk6' + str(i)
    print(STAMP)
    early_stopping = EarlyStopping(monitor='val_loss', patience=4)
    bst_model_path = STAMP + '.h5'
    # model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

    from sklearn.metrics import roc_auc_score


    class RocAucEvaluation(Callback):
        def __init__(self, validation_data=(), interval=1):
            super(Callback, self).__init__()

            self.interval = interval
            self.X_val, self.y_val = validation_data

        def on_epoch_end(self, epoch, logs={}):
            if epoch % self.interval == 0:
                y_pred = self.model.predict(self.X_val, verbose=0)
                score = roc_auc_score(self.y_val, y_pred)
                print("---ROC-AUC - epoch: %d - score: %.6f " % (epoch + 1, score))
                for i in range(6):
                    print(i)
                    score = roc_auc_score(self.y_val[:, i], y_pred[:, i])
                    print("ROC-AUC - i: %d - score: %.6f " % (epoch + 1, score))


    RocAuc = RocAucEvaluation(validation_data=(test_feat, test_y), interval=1)
    maxscore = 0.0
    bst_weight = STAMP
    for j in range(10):
        model.fit([train_feat], train_y,
                  batch_size=133,
                  epochs=1,
                  verbose=1,
                  validation_data=([test_feat], test_y),
                  callbacks=[early_stopping]
                  )
        y_pred = model.predict([test_feat], verbose=0)
        score = roc_auc_score(test_y, y_pred)
        if score > maxscore:
            maxscore = score
            bst_weight = STAMP
            model.save_weights(bst_weight, overwrite=True)

    model.load_weights(bst_weight)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adagrad(lr=0.0080))
    model.fit([train_feat], train_y,
              batch_size=233,
              epochs=1,
              verbose=1,
              validation_data=([test_feat], test_y),
              callbacks=[]
              )
    model.save_weights(output_path + 'w3k6' + str(i), overwrite=True)





