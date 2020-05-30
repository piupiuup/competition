# -*- coding:utf-8 -*-
import os
import numpy as np
import pandas as pd
import time
import argparse
from collections import defaultdict
from sklearn.metrics import mean_squared_error
from gensim.models import word2vec
from keras.models import Sequential, load_model, Model
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler

from common import *
from nn_utils import *
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--gpu', default='0')
parser.add_argument('--max_seq_len', type=int, default=20)
parser.add_argument('--max_nb_words', type=int, default=30000)
parser.add_argument('--fc_dim', type=int, default=256)
parser.add_argument('--fc_dropout', type=float, default=.3)
parser.add_argument('--sp_dropout', type=float, default=.2)
parser.add_argument('--filter_len', type=int, default=5)
parser.add_argument('--nb_filter', type=int, default=64)
parser.add_argument('--pool_len', type=int, default=4)
parser.add_argument('--optimizer', default='adam')
parser.add_argument('--rnn', choices=['lstm', 'gru'], default='lstm')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_decay', type=float, default=1)
parser.add_argument('--decay_epoch', type=int, default=100)
parser.add_argument('--batchsize', type=int, default=1200)
parser.add_argument('--predict', type=int, default=0)
parser.add_argument('-m', '--message', type=str, default="")

# parser.add_argument('--embedding_dim', type=int, )
args = parser.parse_args()
GPU = args.gpu
MAX_SEQUENCE_LENGTH = args.max_seq_len
MAX_NB_WORDS = args.max_nb_words

FC_DIM = args.fc_dim
FC_DROPOUT = args.fc_dropout
SPATIAL_DROPOUT = args.sp_dropout
filter_length = args.filter_len
nb_filter = args.nb_filter
pool_length = args.pool_len
OPTIMIZER = args.optimizer
RNN_CELL = args.rnn
LEARNING_RATE = args.lr
LR_DECAY = args.lr_decay
MAX_DECAY_EPOCH = args.decay_epoch
BATCHSIZE = args.batchsize
PREDICT = args.predict
MESSAGE = args.message
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = GPU
t1 = time.time()

emb_dic = {}
with open("../input/word_embed.txt") as f:
    word_emb = f.readlines()
    print("Number of word embeddings:", len(word_emb))
    for w in word_emb:
        w = w.replace("\n", "")
        content = w.split(" ")
        emb_dic[content[0].lower()] = np.array(content[1:])
EMBEDDING_DIM = len(content) - 1
print("Embedding_dim:", EMBEDDING_DIM)
train = pd.read_csv('../input/train.csv')  #[:1000]
train_2 = train.copy()
train_2.columns = ["label", "q2", "q1"]
#train=train.append(train_2).drop_duplicates().reset_index(drop=True)

test = pd.read_csv('../input/test.csv')  #[:1000]
ques = pd.read_csv('../input/question.csv')
ques.columns = ["q1", "w1", "c1"]
train = train.merge(ques, on="q1", how="left")
test = test.merge(ques, on="q1", how="left")
ques.columns = ["q2", "w2", "c2"]
train = train.merge(ques, on="q2", how="left")
test = test.merge(ques, on="q2", how="left")

##########################################################
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
test_df["label"] = -1

data = pd.concat([train_df[['q1', 'q2']], \
                  test_df[['q1', 'q2']]], axis=0).reset_index(drop='index')
q_dict = defaultdict(set)
for i in range(data.shape[0]):
    q_dict[data.q1[i]].add(data.q2[i])
    q_dict[data.q2[i]].add(data.q1[i])


def q1_freq(row):
    return (len(q_dict[row['q1']]))

def q2_freq(row):
    return (len(q_dict[row['q2']]))

def q1_q2_intersect(row):
    return (len(set(q_dict[row['q1']]).intersection(set(q_dict[row['q2']]))))

print(test_df.head())
train_df['q1_q2_intersect'] = train_df.apply(q1_q2_intersect, axis=1, raw=True)
train_df['q1_freq'] = train_df.apply(q1_freq, axis=1, raw=True)
train_df['q2_freq'] = train_df.apply(q2_freq, axis=1, raw=True)
train_df['min_freq'] = train_df[['q1_freq', 'q2_freq']].min(axis=1)
train_df['max_freq'] = train_df[['q1_freq', 'q2_freq']].max(axis=1)
print(train_df.head())
test_df['q1_q2_intersect'] = test_df.apply(q1_q2_intersect, axis=1, raw=True)
test_df['q1_freq'] = test_df.apply(q1_freq, axis=1, raw=True)
test_df['q2_freq'] = test_df.apply(q2_freq, axis=1, raw=True)
test_df['min_freq'] = test_df[['q1_freq', 'q2_freq']].min(axis=1)
test_df['max_freq'] = test_df[['q1_freq', 'q2_freq']].max(axis=1)
print(test_df.head())

#another leakage
"""
df_cores = pd.read_csv("question_kcores.csv", index_col="question")
df_cores.index.names = ["qid"]
df_cores['max_kcore'] = df_cores.apply(lambda row: max(row), axis=1)
df_cores[['max_kcore']].to_csv("question_max_kcores.csv") # with index
cores_dict = pd.read_csv("question_max_kcores.csv", index_col="qid").to_dict()["max_kcore"]
def gen_qid1_max_kcore(row):
    return cores_dict[row["q1"]]
def gen_qid2_max_kcore(row):
    return cores_dict[row["q2"]]

#def gen_max_kcore(row):
#    return max(row["qid1_max_kcore"], row["qid2_max_kcore"])

train_df["qid1_max_kcore"] = train_df.apply(gen_qid1_max_kcore, axis=1)
test_df["qid1_max_kcore"] = test_df.apply(gen_qid1_max_kcore, axis=1)
train_df["qid2_max_kcore"] = train_df.apply(gen_qid2_max_kcore, axis=1)
test_df["qid2_max_kcore"] = test_df.apply(gen_qid2_max_kcore, axis=1)
#df_train["max_kcore"] = df_train.apply(gen_max_kcore, axis=1)
#df_test["max_kcore"] = df_test.apply(gen_max_kcore, axis=1)
"""

leaks = train_df[['q1_q2_intersect', 'min_freq', 'max_freq', 'q1_freq', 'q2_freq']]
test_leaks = test_df[['q1_q2_intersect', 'min_freq', 'max_freq', 'q1_freq', 'q2_freq']]

ss = StandardScaler()
ss.fit(np.vstack((leaks, test_leaks)))
leaks = ss.transform(leaks)
test_leaks = ss.transform(test_leaks)
######################################################

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS, )
tokenizer.fit_on_texts(
    list(train["w1"]) + list(test["w1"]) + list(train["w2"]) +
    list(test["w2"]))
column = "w1"
sequences_all = tokenizer.texts_to_sequences(list(train[column]))
sequences_test = tokenizer.texts_to_sequences(list(test[column]))
X_train_1 = pad_sequences(
    sequences_all, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
X_test_1 = pad_sequences(
    sequences_test, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
column = "w2"
sequences_all = tokenizer.texts_to_sequences(list(train[column]))
sequences_test = tokenizer.texts_to_sequences(list(test[column]))
X_train_2 = pad_sequences(
    sequences_all, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
X_test_2 = pad_sequences(
    sequences_test, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

word_index = tokenizer.word_index
nb_words = min(MAX_NB_WORDS, len(word_index)) + 1
print("nb_words:", nb_words)

ss = 0
word_embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
print(len(word_index.items()))
for word, i in word_index.items():
    if word in emb_dic.keys():
        ss += 1
        word_embedding_matrix[i] = emb_dic[word]
    else:
        pass
print(ss)
print(word_embedding_matrix)

y = train["label"]

# 建立模型
# Define the model
from esim import esim, decomposable_attention
model = esim(
    word_embedding_matrix,
    leaks.shape[1],
    maxlen=MAX_SEQUENCE_LENGTH,
    lstm_dim=EMBEDDING_DIM,
    dense_dim=FC_DIM,
    dense_dropout=FC_DROPOUT,
    spatial_dropout=SPATIAL_DROPOUT,
    lr=LEARNING_RATE,
    cell=RNN_CELL,
    clip_gradient=1.,
)

early_stop = EarlyStopping(patience=2)
check_point = ModelCheckpoint(
    'paipaidai{}.hdf5'.format('_'+GPU),
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    verbose=1)

lr_ruducer = LR_Reducer(1, LR_DECAY, max_epoch=MAX_DECAY_EPOCH)
lr_booster = LR_Reducer(1, 1, 10, 2, max_batch=3)

history = model.fit(
    [X_train_1, X_train_2, leaks],
    y,
    batch_size=BATCHSIZE,
    epochs=100,
    verbose=1,
    validation_split=0.2,
    callbacks=[early_stop, check_point, lr_ruducer, lr_booster])
if PREDICT:
    model.load_weights('paipaidai{}.hdf5'.format('_'+GPU))
    preds = model.predict([X_test_1, X_test_2, test_leaks], verbose=1)

    #保存概率文件
    test_prob = pd.DataFrame(preds)
    test_prob.columns = ["y_pre"]
    test_prob.to_csv('../sub/baseline_v2.csv', index=None)
