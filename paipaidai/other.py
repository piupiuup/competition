#coding:utf-8

'''
简单的baselin
公众号：麻婆豆腐AI
github：zs167275

网络基本结构：

   INPUT1      IINPUT2
    |          |
embedding_q1 embedding_q2
    |          |
    |          |
    ------------
          |
          |
        全连接
          |
         output

感受：本人没有GPU，训练的好慢啊，慢啊，慢啊。
收购二手1080，10块钱无限收

线下随机分割的数据集：
线下logloss 0.3623
线上logloss 0.376738
'''

import pandas as pd
import numpy as np
# 文本处理
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, TimeDistributed, Dense, Lambda, concatenate, Dropout, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils.data_utils import get_file
from keras import backend as K
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

data_path = 'C:/Users/cui/Desktop/python/paipaidai/data/'

# word level
question = pd.read_csv(data_path + 'question.csv')
question = question[['qid','words']]

train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')
train = pd.merge(train,question,left_on=['q1'],right_on=['qid'],how='left')
train = pd.merge(train,question,left_on=['q2'],right_on=['qid'],how='left')
train = train[['label','words_x','words_y']]
train.columns = ['label','q1','q2']

test = pd.merge(test,question,left_on=['q1'],right_on=['qid'],how='left')
test = pd.merge(test,question,left_on=['q2'],right_on=['qid'],how='left')
test = test[['words_x','words_y']]
test.columns = ['q1','q2']

all = pd.concat([train,test])

MAX_NB_WORDS = 1000

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(question['words'])

q1_word_seq = tokenizer.texts_to_sequences(all['q1'])
q2_word_seq = tokenizer.texts_to_sequences(all['q2'])
word_index = tokenizer.word_index


embeddings_index = {}
with open(data_path + 'word_embed.txt','r') as f:
    for i in f:
        values = i.split(' ')
        word = str(values[0])
        embedding = np.asarray(values[1:],dtype='float')
        embeddings_index[word] = embedding
print('word embedding',len(embeddings_index))

EMBEDDING_DIM = 300
nb_words = min(MAX_NB_WORDS,len(word_index))
word_embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(str(word).upper())
    if embedding_vector is not None:
        word_embedding_matrix[i] = embedding_vector

MAX_SEQUENCE_LENGTH = 25
q1_data = pad_sequences(q1_word_seq,maxlen=MAX_SEQUENCE_LENGTH)
q2_data = pad_sequences(q2_word_seq,maxlen=MAX_SEQUENCE_LENGTH)

train_q1_data = q1_data[:train.shape[0]]
train_q2_data = q2_data[:train.shape[0]]

test_q1_data = q1_data[train.shape[0]:]
test_q2_data = q2_data[train.shape[0]:]

labels = train['label']
print('Shape of question1 train data tensor:', train_q1_data.shape)
print('Shape of question2 train data tensor:', train_q2_data.shape)
print('Shape of question1 test data tensor:', test_q1_data.shape)
print('Shape of question1 test data tensor:', test_q2_data.shape)
print('Shape of label tensor:', labels.shape)


X = np.stack((train_q1_data, train_q2_data), axis=1)
y = labels

from sklearn.model_selection import StratifiedShuffleSplit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
Q1_train = X_train[:,0]
Q2_train = X_train[:,1]
Q1_test = X_test[:,0]
Q2_test = X_test[:,1]


DROPOUT = 0.25
# Define the model

question1 = Input(shape=(MAX_SEQUENCE_LENGTH,))
question2 = Input(shape=(MAX_SEQUENCE_LENGTH,))

q1 = Embedding(nb_words + 1,
                 EMBEDDING_DIM,
                 weights=[word_embedding_matrix],
                 input_length=MAX_SEQUENCE_LENGTH,
                 trainable=False)(question1)
q1 = TimeDistributed(Dense(EMBEDDING_DIM, activation='relu'))(q1)
q1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, ))(q1)

q2 = Embedding(nb_words + 1,
                 EMBEDDING_DIM,
                 weights=[word_embedding_matrix],
                 input_length=MAX_SEQUENCE_LENGTH,
                 trainable=False)(question2)
q2 = TimeDistributed(Dense(EMBEDDING_DIM, activation='relu'))(q2)
q2 = Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, ))(q2)

merged = concatenate([q1,q2])

merged = Dense(200, activation='relu')(merged)
merged = Dropout(DROPOUT)(merged)
merged = BatchNormalization()(merged)

is_duplicate = Dense(1, activation='sigmoid')(merged)

model = Model(inputs=[question1,question2], outputs=is_duplicate)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_crossentropy'])

history = model.fit([Q1_train, Q2_train],
                    y_train,
                    epochs=2,
                    validation_data=[[Q1_test,Q2_test],y_test],
                    verbose=1,
                    batch_size=128,
                    )

# 预测结果
result = model.predict([test_q1_data,test_q2_data],batch_size=1024)

# 提交结果
submit = pd.DataFrame()
submit['y_pre'] = list(result[:,0])
submit.to_csv('../submit/baseline.csv',index=False)

