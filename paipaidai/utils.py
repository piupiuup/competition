
from keras.layers import *
import numpy as np
np.set_printoptions(suppress=True)


def as_num(x):
    y = '{:.11f}'.format(x)
    return (y)

def make_submission(predict_prob,modelname):
    with open(modelname+'submission.csv', 'w') as file:
        file.write(str('y_pre') + '\n')
        for line in predict_prob:
            file.write(str(line) + '\n')
    file.close()

def save_classes(predict_prob,modelname):
    with open(modelname+'feature.csv', 'w') as file:
        file.write(str('y_pre') + '\n')
        for line in predict_prob:
            if line>=0.5:
                file.write(str(1) + '\n')
            else:
                file.write(str(0) + '\n')
    file.close()

def model_save(modelname,model):
    #model.save(modelname + "s_model_1.h5")
    model.save_weights(modelname + "s_model_1.h5")


def model_result_print(modelname,history):

    print(history.history.keys())
    from matplotlib import pyplot as plt
    fig = plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.legend(['train'], loc='upper left')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower left')

    fig.savefig(modelname + 's_model_1.png')

def model_result_print1(modelname,history):

    print(history.history.keys())
    from matplotlib import pyplot as plt
    fig = plt.figure()
    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='lower left')

    fig.savefig(modelname + '3.png')

def f1_score(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0
    # How many selected items are relevant?
    precision = c1 / c2
    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def precision(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0
    # How many selected items are relevant?
    precision = c1 / c2
    return precision


def recall(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0
    recall = c1 / c3
    return recall


if __name__ == '__main__':
    print(as_num(0.681247))