import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from multiprocessing import Pool
import pickle, gc, warnings, os, time

warnings.filterwarnings('ignore')
import itertools
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--save_name", default='model.h5',
                    help="filename to save the model")
parser.add_argument("--gpu", default='0',
                    help="which gpu(s) to use")
parser.add_argument("--predict", type=int, choices=[0, 1], default=1,
                    help="whether to predict the test set")
parser.add_argument("--n_folds", type=int, default=5,
                    help="num of folds")
parser.add_argument("--lr", type=float, default=0.002,
                    help="initial value of the learning rate")
parser.add_argument("--lr_decay", type=float, default=0.75,
                    help="the decay factor of learning rate at the end of each epochs")
parser.add_argument("--epoch", type=int, default=12,
                    help="number of epochs")
parser.add_argument("--opt", choices=['adam', 'rmsp', 'sgd'], default='adam',
                    help="Which Optimizer to use")
parser.add_argument("--batchsize", type=int, default=5000,
                    help="Batchsize while triaining (not predicting)")
parser.add_argument("--saveCV", type=int, default=0)

args = parser.parse_args()
SAVE_NAME = args.save_name
PREDICT_FLAG = int(args.predict)
N_FOLDS = args.n_folds
LR_INIT = args.lr
LR_DECAY = args.lr_decay
EPOCH = args.epoch
OPTIMIZER = args.opt
BATCHSIZE = args.batchsize
SAVE_CV = args.saveCV
print(args)
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import keras
from keras.models import Model, load_model
from keras.layers import Input, Embedding, Dense, Flatten, Dropout, Activation, LeakyReLU
from keras.layers import concatenate, merge, Add
from keras.layers import BatchNormalization, SpatialDropout1D
from keras.optimizers import Adam, RMSprop, SGD, TFOptimizer
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils.training_utils import multi_gpu_model
from keras.constraints import unit_norm
from keras import regularizers
from keras import backend as K
import tensorflow as tf
from talkingdata.v7.common import *
from talkingdata.v7.nn_utils import *

N_THREADS = 20
N_ITER = 1
NEG_SAMPLE_RATE = 1
N_CUTS = 32


def get_model(cate_dict={}, numerical_columns=[]):
    emb_n = 50
    reg_lambda = 3e-7  # 1e-7 ~ 3e-7 works fine for the L1 regularizers.
    # get embeddings for categorical columns
    inputs_cate = [Input(shape=[1], name=col) for col in cate_dict]
    embeddings = []
    for i, col in enumerate(cate_dict):
        embeddings.append(
            Flatten()(Embedding(cate_dict[col], emb_n)(inputs_cate[i])))
    fm_layers = []
    k = 0
    for emb1, emb2 in itertools.combinations(embeddings, 2):
        k += 1
        if k % 3 == 0:
            continue
        dot_layer = merge([emb1, emb2], mode='dot', dot_axes=1)
        fm_layers.append(dot_layer)
    l = concatenate(embeddings)
    l = Dropout(.3)(l)
    l = Dense(256, name='Embedding_FC',
              activation=LeakyReLU(),  # 'relu',
              kernel_regularizer=regularizers.l1(reg_lambda))(l)
    fm = concatenate(fm_layers)
    fm = Dropout(.2)(fm)

    # inputs_numerical = [Input(shape=[1], name=col) for col in numerical_columns]
    # fn = concatenate(inputs_numerical)
    inputs_numerical = [Input(shape=[len(numerical_columns)], name='numerical')]
    fn = inputs_numerical[0]
    # 192, 400 are worse than 256
    xn = fn
    for i in range(3):
        xn = Dense(200, activation='elu',
                   kernel_regularizer=regularizers.l1(reg_lambda))(xn)
        xn = Dropout(.2)(xn)

    x = concatenate([l, fm, xn])
    x = Dense(1024, name="Merge_FC",
              kernel_regularizer=regularizers.l1(reg_lambda))(x)
    # tested BN, good.
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Dropout(.3)(x)
    # 128, 192, 400 are all worse than 256
    x = Dense(256,
              activation='elu',
              name="Middle_FC",
              kernel_regularizer=regularizers.l1(reg_lambda))(x)
    x = Dropout(.2)(x)
    # l1 regularization is good
    outp = Dense(1, activation='sigmoid', name="Last_FC",
                 kernel_regularizer=regularizers.l1(reg_lambda))(x)

    model = Model(inputs=inputs_cate + inputs_numerical, outputs=outp)
    return model


def group_feature_model(cate_dict={}, numerical_columns=[], cate_groups=[]):
    emb_n = 64
    reg_lambda = 3e-7  # 1e-7 ~ 3e-7 works fine for the L1 regularizers.
    # get embeddings for categorical columns
    inputs_cate = []
    inputs_cate = [Input(shape=[1], name=col) for col in cate_dict]
    groups_concat = sum(cate_groups, [])
    cate_not_in_groups = [x for x in cate_dict if x not in groups_concat]
    cate_groups.append(cate_not_in_groups)
    embeddings = []
    for group in cate_groups:
        group_emb = []
        for col in group:
            inputs_cate.append(Input(shape=[1], name=col))
            group_emb.append(Flatten()(Embedding(cate_dict[col], emb_n)(inputs_cate[-1])))
        embeddings.append(group_emb)
    fm_layers = []
    k = 0
    for group1, group2 in itertools.combinations(embeddings, 2):
        for emb1 in group1:
            for emb2 in group2:
                k += 1
                if k % 3 == 0:
                    continue

                dot_layer = merge([emb1, emb2], mode='dot', dot_axes=1)
                fm_layers.append(dot_layer)
    embeddings = sum(embeddings, [])

    l = concatenate(embeddings)
    l = Dropout(.3)(l)
    l = Dense(256, name='Embedding_FC',
              activation=LeakyReLU(),  # 'relu',
              kernel_regularizer=regularizers.l1(reg_lambda))(l)
    fm = concatenate(fm_layers)
    fm = Dropout(.2)(fm)

    # inputs_numerical = [Input(shape=[1], name=col) for col in numerical_columns]
    # fn = concatenate(inputs_numerical)
    inputs_numerical = [Input(shape=[len(numerical_columns)], name='numerical')]
    fn = inputs_numerical[0]
    # 192, 400 are worse than 256
    xn = Dense(256, activation='elu',
               kernel_regularizer=regularizers.l1(reg_lambda))(fn)
    xn = Dropout(.2)(xn)

    x = concatenate([l, fm, xn])
    x = Dense(1024, name="Merge_FC",
              kernel_regularizer=regularizers.l1(reg_lambda))(x)
    # tested BN, good.
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Dropout(.3)(x)
    # 128, 192, 400 are all worse than 256
    x = Dense(256,
              activation='elu',
              name="Middle_FC",
              kernel_regularizer=regularizers.l1(reg_lambda))(x)
    x = Dropout(.2)(x)
    # l1 regularization is good
    outp = Dense(1, activation='sigmoid', name="Last_FC",
                 kernel_regularizer=regularizers.l1(reg_lambda))(x)

    model = Model(inputs=inputs_cate + inputs_numerical, outputs=outp)
    return model


def keras_train(model, train_x, train_y, val_x, val_y,
                loss='binary_crossentropy',
                batch_size=4096, epochs=3, lr_init=0.001, lr_decay=1,
                clipnorm=2., scale_pos_weight=1.,
                pretrain_df=None, pretrain_y=None, pretrain_epochs=0):
    print("Start training! batchsize {}, epochs {}, lr_init {}, lr_decay {}, clipnorm {}, scale_pos_weight {}".format(
        batch_size, epochs, lr_init, lr_decay, clipnorm, scale_pos_weight
    ))
    keras_opt = {'adam': Adam, 'rmsp': RMSprop, 'sgd': SGD}
    opt = Adam(lr=lr_init, clipnorm=clipnorm,
               amsgrad=True)  # if OPTIMIZER!='ftrl' else TFOptimizer(tf.train.FtrlOptimizer(lr_init))
    model.compile(loss=loss, optimizer=opt)
    # model.summary()
    train_gen = DataGenerator(train_x, train_y, batch_size)
    earlystopper = AucEarlyStopping(patience=10, verbose=1, val_x=val_x, val_y=val_y, filepath=SAVE_NAME,
                                    every_N_batch=200)
    lr_reducer = LR_Reducer(1, lr_decay)
    zero_blackhole = Zero_Blackhole(20, ["Embedding_FC", "Merge_FC"], eps=3e-5, verbose=0)
    model.fit_generator(
        train_gen, epochs=epochs,
        class_weight={0: 1., 1: scale_pos_weight},
        use_multiprocessing=False,
        # workers=2,
        callbacks=[earlystopper, lr_reducer, zero_blackhole])
    model.set_weights(earlystopper.weights)
    return model


path = 'E:/talkingdata/data/'
timer = Timer()
try:
    print("Load the training and testing data...")
    with pd.HDFStore(path + 'nn_data_piupiu.h5', "r") as hdf:
        train_df = pd.read_hdf(hdf, 'train_df')
        train_df.columns = [c.replace('&', '_and_') for c in train_df.columns]
        del train_df['ip']
        train_y = pd.read_hdf(hdf, 'train_y')
        train_y = pd.Series(train_y, train_df.index)
        test_df = pd.read_hdf(hdf, 'test_df')
        test_df.columns = [c.replace('&', '_and_') for c in test_df.columns]
        test_id = pd.read_hdf(hdf, 'test_id')
        del test_df['ip']
    train_df.info()
    test_df.info()
    timer.get_eclipse()
except:
    raise ValueError("Cannot find hdf5 data! \nPlease run lgb.py first to generate the data!")
target = 'is_attributed'
categorical = ['app', 'device', 'os', 'channel', 'hour']
# categorical这边还是需要写一下的，如果prepare文件里面改了
cate_dict = {}
numerical_columns = []
for col in train_df.columns:
    if col in categorical or 'encoded' in col:
        cate_dict[col] = max(train_df[col].max(), test_df[col].max()) + 1
    else:
        numerical_columns.append(col)
sub = pd.DataFrame()
sub['click_id'] = test_id

predictors = list(cate_dict.keys()) + numerical_columns


# get_nn_data = lambda df, col: {c: df[c].values for c in col}
def get_nn_data(df, cate_dict, numerical_columns):
    ans = {}
    for col in cate_dict:
        ans[col] = df[col].values
    ans['numerical'] = df[numerical_columns].values
    return ans


test_x = get_nn_data(test_df, cate_dict, numerical_columns)
if SAVE_CV:
    cv_hdf = pd.HDFStore("../sub/CV_nn_{}-folds.h5".format(N_FOLDS), "w")
for N in range(N_ITER):
    scores = []
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=77 + N).split(train_df, train_y)
    for i, (idx_train, idx_valid) in enumerate(kf):
        timer.reset()
        tr_x, tr_y = under_sample(train_df, train_y, idx_train, NEG_SAMPLE_RATE)
        val_x, val_y = under_sample(train_df, train_y, idx_valid, NEG_SAMPLE_RATE)
        tr_x, val_x = get_nn_data(tr_x, cate_dict, numerical_columns), get_nn_data(val_x, cate_dict, numerical_columns)
        model = get_model(cate_dict, numerical_columns)
        # model = group_feature_model(cate_dict, numerical_columns, cate_groups)
        print("Training...")
        clf = keras_train(
            model, tr_x, tr_y, val_x, val_y,
            batch_size=BATCHSIZE, epochs=EPOCH,
            lr_init=LR_INIT, lr_decay=LR_DECAY,
            clipnorm=.5,
            scale_pos_weight=5 * NEG_SAMPLE_RATE,
            # pretrain_df=train_other, pretrain_y=train_y_other,
            pretrain_epochs=0)
        print("Validating...")
        val_x, val_y = train_df.iloc[idx_valid], train_y.iloc[idx_valid]
        val_x = get_nn_data(val_x, cate_dict, numerical_columns)
        val_pred = clf.predict(val_x, batch_size=32768, verbose=1).reshape(-1)
        val_score = roc_auc_score(val_y, val_pred)
        if SAVE_CV:
            cv_hdf.put('groundtruth_{}'.format(i), val_y)
            cv_hdf.put('prediction_{}'.format(i), pd.Series(val_pred, index=val_y.index))
        print("valication score:", val_score)
        del tr_x, val_x
        gc.collect()
        timer.get_eclipse()
        if PREDICT_FLAG:
            print("Predicting...")
            sub['is_attributed_' + str(N * N_FOLDS + i)] = clf.predict(test_x, batch_size=32768, verbose=1).reshape(-1)
            print("Mean of predictions in this fold:", sub['is_attributed_' + str(i)].mean())
        scores.append(val_score)
        print("-" * 90)
        print("K-fold Cross Validation:")
        print(" | ".join(["{:6f}".format(x) for x in scores]))
        print("=" * 90)
        timer.get_eclipse()
if SAVE_CV:
    cv_hdf.close()
print()
print("=" * 90)
print("mean score:", np.mean(scores))
score = np.mean(scores)
if PREDICT_FLAG:
    print("writing...")
    # sub.to_csv('../sub/lgb_kfold_{:6f}.csv.gz'.format(score), index=None, compression='gzip')
    sub.set_index('click_id', inplace=True)
    sub['is_attributed'] = sub.mean(1)
    sub = sub[['is_attributed']]
    import datetime
    sub = sub.reset_index()
    sub.to_csv('C:/Users/cui/Desktop/python/talkingdata/submission/sub{}.csv'.format(
        datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), index=False, float_format='%.8f')
    print("Prediction of the test set is written to", '../sub/nn_{:6f}.csv'.format(score))
    sub.info()
print("done!")
