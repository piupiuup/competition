import numpy as np
import keras
import gc
from keras import backend as K
from keras.optimizers import Optimizer
from sklearn.metrics import roc_auc_score
from scipy.special import erfinv
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.preprocessing.sequence import pad_sequences
from common import *


def rank_gauss_normalization(x):
    """
    Learned from the 1st place solution of Porto competition.
    https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629

    input: x, a numpy array.
    """
    N = x.shape[0]
    temp = x.argsort()
    rank_x = temp.argsort() / N
    rank_x -= rank_x.mean()
    rank_x *= 2  # rank_x.max(), rank_x.min() should be in (-1, 1)
    efi_x = erfinv(rank_x)  # np.sqrt(2)*erfinv(rank_x)
    efi_x -= efi_x.mean()
    ans = efi_x.astype(np.float32)
    return ans


class Nonzero_Counter(keras.callbacks.Callback):
    def __init__(self, layer_list=[], eps=1e-7):
        super(Nonzero_Counter, self).__init__()
        self.layer_list = layer_list if type(layer_list) is list else [layer_list]
        self.eps = eps

    def on_epoch_begin(self, epoch, logs={}):
        if len(self.layer_list) >= 1:
            print("NonZero Counter msg:")
            for l in self.layer_list:
                weights = self.model.get_layer(l).get_weights()[0]
                tot_n = np.prod(weights.shape)
                nonzero_n = np.sum(weights != 0)
                above_eps_n = np.sum(np.abs(weights) > self.eps)
                print("    Layer {}: total {}, nonzero {}, {}%, significant {}, {}%.".format(
                    l, tot_n, nonzero_n, int(nonzero_n / tot_n * 100), above_eps_n, int(above_eps_n / tot_n * 100)))


class Zero_Blackhole(keras.callbacks.Callback):
    def __init__(self, n_batch, layer_list=[], eps=3e-5, verbose=25):
        super(Zero_Blackhole, self).__init__()
        self.layer_list = layer_list if type(layer_list) is list else [layer_list]
        self.eps = eps
        self.n_batch = n_batch
        self.cnt = 0
        self.verbose = verbose

    def on_batch_end(self, batch, logs={}):
        if batch % self.n_batch == 0:
            self.cnt += 1
            if self.verbose > 0 and self.cnt % self.verbose == 0:
                print("")
            for l in self.layer_list:
                weights = self.model.get_layer(l).get_weights()
                w = weights[0]
                w[np.abs(w) < self.eps] = 0.
                self.model.get_layer(l).set_weights(weights)
                tot_n = np.prod(weights[0].shape)
                nonzero_n = np.sum(weights[0] != 0)
                if self.verbose > 0 and self.cnt % self.verbose == 0:
                    print("    Layer {}: total {}, nonzero {}, {}%.".format(
                        l, tot_n, nonzero_n, int(nonzero_n / tot_n * 100)))


class LR_Reducer(keras.callbacks.Callback):
    '''
        n_epoch = no. of epochs after decay should happen.
        decay = decay value
    '''

    def __init__(self, n_epoch, decay, n_batch=0, decay_batch=1.,
                 max_epoch=None, max_batch=None,
                 verbose=1):
        super(LR_Reducer, self).__init__()
        self.n_epoch = n_epoch
        self.decay = decay
        self.n_batch = n_batch
        self.decay_batch = decay_batch
        self.batch_count = 0
        self.verbose = verbose
        self.max_epoch = max_epoch if max_epoch is not None else 10000
        self.max_batch = max_batch if max_batch is not None else 1000000

    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % self.n_epoch == 0 and self.decay != 1 and self.max_epoch > 0:
            old_lr = K.get_value(self.model.optimizer.lr)
            new_lr = self.decay * old_lr
            K.set_value(self.model.optimizer.lr, new_lr)
            print("\tSet learning rate to {}".format(new_lr))
            self.max_epoch += -1
        else:
            pass

    def on_batch_end(self, batch, logs={}):
        self.batch_count += 1
        if self.n_batch != 0 and self.decay_batch != 1 and self.max_batch > 0:
            if self.batch_count % self.n_batch == 0:
                old_lr = K.get_value(self.model.optimizer.lr)
                new_lr = self.decay_batch * old_lr
                K.set_value(self.model.optimizer.lr, new_lr)
                print("\n\tSet learning rate to {}".format(new_lr))
                self.max_batch += -1


class AucEarlyStopping(keras.callbacks.Callback):
    def __init__(self, min_delta=0, patience=0, verbose=0, predict_batch_size=32767,
                 val_gen=None, val_y=None, val_aid=None, filepath="", every_N_batch=0, revert=False,
                 workers=1, every_n_epochs=1):
        # FROM EARLY STOP
        super(AucEarlyStopping, self).__init__()
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.monitor_op = np.greater
        self.predict_batch_size = predict_batch_size
        self.val_gen = val_gen
        self.val_y = np.array(val_y)
        self.filepath = filepath
        self.N = every_N_batch
        self.batch = 0
        self.weights = []
        self.revert = revert
        self.workers = workers
        self.val_aid = np.array(val_aid)
        self.every_n_epochs = every_n_epochs

    def on_batch_end(self, batch, logs={}):
        self.batch += 1
        if self.N > 0 and self.best > 0:
            if self.batch % self.N == 0:
                y_hat_val = self.model.predict_generator(
                    self.val_gen, workers=self.workers,
                    use_multiprocessing=False if self.workers == 1 else True).reshape(-1)
                if self.revert:
                    y_hat_val = 1 - y_hat_val
                current = get_score(self.val_y, y_hat_val, self.val_aid)
                if self.monitor_op(current - self.min_delta, self.best):
                    self.best = current
                    print("\n    New best score {:3f}! Store weights to memory.".format(current))
                    self.weights = self.model.get_weights()
                    self.wait = 0
                    self.best_valid_pred = y_hat_val
                else:
                    print("\n    AUC score {:3f}, did not improve!".format(current))

    def on_train_begin(self, logs={}):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -np.Inf

    def on_train_end(self, logs={}):
        # FROM EARLY STOP
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch ', self.stopped_epoch, ': AucEarlyStopping')

    def on_epoch_end(self, epoch, logs={}):
        self.batch = 0
        if (epoch + 1) % self.every_n_epochs == 0:
            y_hat_val = self.model.predict_generator(
                self.val_gen, workers=self.workers,
                use_multiprocessing=False if self.workers == 1 else True,
                verbose=1
            ).reshape(-1)
            if self.revert:
                y_hat_val = 1 - y_hat_val
            current = get_score(self.val_y, y_hat_val, self.val_aid)
            if (self.verbose == 1):
                print("\n    AUC Callback:", current)
            if self.monitor_op(current - self.min_delta, self.best):
                self.best = current
                self.wait = 0
                print("    New best score! Store weights to memory.".format(self.filepath))
                self.weights = self.model.get_weights()
                self.best_valid_pred = y_hat_val
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True


class DataGenerator(keras.utils.Sequence):
    '''Generates data for TSA competition
    inputs:
        X: a pandas dataframe including all features
        y: a numpy array or pandas Series of labels
        batch_size: batchsize, default 1024
        num_words_dict: a dict for sequencial inputs
        maxlen_dict: a dict for sequencial inputs
    '''

    def __init__(self, X=None, y=None, batch_size=1024, shuffle=True,
                 num_words_dict={}, maxlen_dict={}):
        self.seq_columns = list(maxlen_dict.keys())
        self.batch_size = batch_size
        self.maxlen_dict = maxlen_dict
        self.tokenizers = {}
        for key in self.seq_columns:
            self.tokenizers[key] = myTokenizer(num_words=num_words_dict[key],
                                               sort=True)  # oov_token=num_words_dict[key]+1)
        self.X = X
        self.y = np.array(y, dtype='int') if y is not None else None
        self.len_train = self.X.shape[0]
        self.shuffle = shuffle if y is not None else False
        self.columns = list(num_words_dict.keys())
        self.on_epoch_end()
        self.len_train = self.X.shape[0]

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.len_train / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:min(self.len_train, (index + 1) * self.batch_size)]
        X_slice = self.X.iloc[indexes]
        y_slice = self.y[indexes] if self.y is not None else None
        X_dict = {}
        for col in self.columns:
            if col not in self.seq_columns:
                X_dict[col] = X_slice[col].values
            else:
                seqs = self.tokenizers[col].texts_to_sequences(X_slice[col].values)
                seqs = pad_sequences(seqs,
                                     maxlen=self.maxlen_dict[col],
                                     padding='post', truncating='post')
                X_dict[col] = seqs
        gc.collect()
        if self.y is not None:
            return X_dict, y_slice
        else:
            return X_dict

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.len_train)
        if self.shuffle:
            np.random.shuffle(self.indexes)


def tanh4(x):
    return K.tanh(x) * 4


from keras.utils.generic_utils import get_custom_objects

get_custom_objects().update({'tanh4': keras.layers.Activation(tanh4)})
