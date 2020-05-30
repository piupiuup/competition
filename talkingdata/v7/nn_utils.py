import numpy as np
import keras
from keras import backend as K
from keras.optimizers import Optimizer
from sklearn.metrics import roc_auc_score
from scipy.special import erfinv


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

    def __init__(self, n_epoch, decay):
        super(LR_Reducer, self).__init__()
        self.n_epoch = n_epoch
        self.decay = decay

    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % self.n_epoch == 0 and self.decay < 1:
            old_lr = K.get_value(self.model.optimizer.lr)
            new_lr = self.decay * old_lr
            K.set_value(self.model.optimizer.lr, new_lr)
            print("Set learning rate to {}".format(new_lr))
        else:
            pass
            # K.set_value(self.model.optimizer.lr, old_lr)


class AucEarlyStopping(keras.callbacks.Callback):
    def __init__(self, min_delta=0, patience=0, verbose=0, predict_batch_size=32767,
                 val_x=None, val_y=None, filepath="", every_N_batch=0):
        # FROM EARLY STOP
        super(AucEarlyStopping, self).__init__()
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.monitor_op = np.greater
        self.predict_batch_size = predict_batch_size
        self.val_df = val_x
        self.val_y = val_y
        self.filepath = filepath
        self.N = every_N_batch
        self.batch = 0
        self.weights = []

    def on_batch_end(self, batch, logs={}):
        self.batch += 1
        if self.N > 0 and self.best > 0:
            if self.batch % self.N == 0:
                if (self.val_df):
                    y_hat_val = self.model.predict(self.val_df, batch_size=self.predict_batch_size)
                    current = roc_auc_score(self.val_y, y_hat_val)
                    if self.monitor_op(current - self.min_delta, self.best):
                        self.best = current
                        print("\n    New best score {:3f}! Store weights to memory.".format(current, self.filepath))
                        self.weights = self.model.get_weights()
                        self.wait = 0
                    else:
                        print("\n    AUC score did not improve!")

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
        if (self.val_df):
            y_hat_val = self.model.predict(self.val_df, batch_size=self.predict_batch_size)
        # FROM EARLY STOP
        if (self.val_df):
            current = roc_auc_score(self.val_y, y_hat_val)
            if (self.verbose == 1):
                print("\n    AUC Callback:", current)
            if self.monitor_op(current - self.min_delta, self.best):
                self.best = current
                self.wait = 0
                print("    New best score! Store weights to memory.".format(self.filepath))
                self.weights = self.model.get_weights()
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, X, y=None, batch_size=1024, shuffle=True, autoencoder_return_cols=[], AE=False):
        self.X = X
        self.y = np.array(y) if y is not None else None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.AE = AE
        if y is not None:
            self.len_train = len(self.y)
        else:
            self.len_train = len(list(X.values())[0])
            self.ae_cols = autoencoder_return_cols
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.len_train / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:min(self.len_train, (index + 1) * self.batch_size)]
        # Find list of IDs
        get_slice = lambda arr: arr[indexes]
        X_slice = dict(map(lambda kv: (kv[0], get_slice(kv[1])), self.X.items()))
        if self.y is not None:
            y_slice = get_slice(self.y)
            return X_slice, y_slice
        else:
            if self.AE:
                len_here = len(indexes)
                y_slice = np.hstack([X_slice[k].reshape((-1, 1)) for k in self.ae_cols])
                for k in self.ae_cols:
                    X_slice[k] *= np.exp(np.random.standard_normal(size=len_here) / 10)
            return X_slice, y_slice

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.len_train)
        if self.shuffle:
            np.random.shuffle(self.indexes)


def tanh4(x):
    return K.tanh(x) * 4


from keras.utils.generic_utils import get_custom_objects

get_custom_objects().update({'tanh4': keras.layers.Activation(tanh4)})


class DataGenerator_fromAE(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, X, y=None, model_AE=None, batch_size=1024, shuffle=True, categorical=[]):
        self.X = X
        self.y = np.array(y) if y is not None else None
        self.batch_size = batch_size
        self.shuffle = shuffle
        if y is not None:
            self.len_train = len(self.y)
        else:
            self.len_train = len(list(X.values())[0])
        self.model_AE = model_AE
        self.categorical = categorical
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.len_train / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:min(self.len_train, (index + 1) * self.batch_size)]
        # Find list of IDs
        get_slice = lambda arr: arr[indexes]
        X_slice = dict(map(lambda kv: (kv[0], get_slice(kv[1])), self.X.items()))
        AE_out = self.model_AE.predict(X_slice, batch_size=self.batch_size + 1)
        X_slice['AE_output'] = AE_out
        if self.y is not None:
            y_slice = get_slice(self.y)
            return X_slice, y_slice
        else:
            return X_slice

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.len_train)
        if self.shuffle:
            np.random.shuffle(self.indexes)


class AucEarlyStopping_fromAE(keras.callbacks.Callback):
    def __init__(self, model_AE=None, min_delta=0, patience=0, verbose=0, predict_batch_size=32767,
                 val_x=None, val_y=None, filepath="", every_N_batch=0, categorical=[]):
        # FROM EARLY STOP
        super(AucEarlyStopping_fromAE, self).__init__()
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.monitor_op = np.greater
        self.predict_batch_size = predict_batch_size
        self.val_df = val_x
        self.val_y = val_y
        self.val_gen = DataGenerator_fromAE(
            self.val_df, model_AE=model_AE, batch_size=32768,
            shuffle=False, categorical=categorical)
        self.filepath = filepath
        self.N = every_N_batch
        self.batch = 0
        self.weights = []

    def on_batch_end(self, batch, logs={}):
        self.batch += 1
        if self.N > 0 and self.best > 0:
            if self.batch % self.N == 0:
                if (self.val_df):
                    y_hat_val = self.model.predict_generator(self.val_gen)
                    current = roc_auc_score(self.val_y, y_hat_val)
                    if self.monitor_op(current - self.min_delta, self.best):
                        self.best = current
                        print(" New best score {:3f}! Store weights to memory.".format(current, self.filepath))
                        self.weights = self.model.get_weights()
                    else:
                        print(" AUC score did not improve!")

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
        if (self.val_df):
            y_hat_val = self.model.predict_generator(self.val_gen)
        # FROM EARLY STOP
        if (self.val_df):
            current = roc_auc_score(self.val_y, y_hat_val)
            if (self.verbose == 1):
                print("\n    AUC Callback:", current)
            if self.monitor_op(current - self.min_delta, self.best):
                self.best = current
                self.wait = 0
                print("    New best score! Store weights to memory.".format(self.filepath))
                self.weights = self.model.get_weights()
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
