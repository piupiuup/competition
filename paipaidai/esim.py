import numpy as np
import pandas as pd
from keras.layers import *
from keras.activations import softmax
from keras.models import Model
from keras.optimizers import Nadam, Adam
from keras.regularizers import l2
import keras.backend as K

MAX_LEN = 20


def create_pretrained_embedding(pretrained_weights, trainable=False, **kwargs):
    "Create embedding layer from a pretrained weights array"
    in_dim, out_dim = pretrained_weights.shape
    embedding = Embedding(in_dim, out_dim, weights=[pretrained_weights], trainable=False, **kwargs)
    return embedding


def unchanged_shape(input_shape):
    "Function for Lambda layer"
    return input_shape


def substract(input_1, input_2):
    "Substract element-wise"
    neg_input_2 = Lambda(lambda x: -x, output_shape=unchanged_shape)(input_2)
    out_ = Add()([input_1, neg_input_2])
    return out_


def submult(input_1, input_2):
    "Get multiplication and subtraction then concatenate results"
    mult = Multiply()([input_1, input_2])
    sub = substract(input_1, input_2)
    out_ = Concatenate()([sub, mult])
    return out_


def apply_multiple(input_, layers):
    "Apply layers to input then concatenate result"
    if not len(layers) > 1:
        raise ValueError('Layers list should contain more than 1 layer')
    else:
        agg_ = []
        for layer in layers:
            agg_.append(layer(input_))
        out_ = Concatenate()(agg_)
    return out_


def time_distributed(input_, layers):
    "Apply a list of layers in TimeDistributed mode"
    out_ = []
    node_ = input_
    for layer_ in layers:
        node_ = TimeDistributed(layer_)(node_)
    out_ = node_
    return out_


def soft_attention_alignment(input_1, input_2):
    "Align text representation with neural soft attention"
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1),
                     output_shape=unchanged_shape)(attention)
    w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2),
                                     output_shape=unchanged_shape)(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned


def decomposable_attention(pretrained_weights,
                           projection_dim=300, projection_hidden=0, projection_dropout=0.2,
                           compare_dim=500, compare_dropout=0.2,
                           dense_dim=300, dense_dropout=0.2,
                           lr=1e-3, activation='elu', maxlen=MAX_LEN):
    # Based on: https://arxiv.org/abs/1606.01933

    q1 = Input(name='q1', shape=(maxlen,))
    q2 = Input(name='q2', shape=(maxlen,))

    # Embedding
    embedding = create_pretrained_embedding(pretrained_weights,
                                            mask_zero=False)
    q1_embed = embedding(q1)
    q2_embed = embedding(q2)

    # Projection
    projection_layers = []
    if projection_hidden > 0:
        projection_layers.extend([
            Dense(projection_hidden, activation=activation),
            Dropout(rate=projection_dropout),
        ])
    projection_layers.extend([
        Dense(projection_dim, activation=None),
        Dropout(rate=projection_dropout),
    ])
    q1_encoded = time_distributed(q1_embed, projection_layers)
    q2_encoded = time_distributed(q2_embed, projection_layers)

    # Attention
    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)

    # Compare
    q1_combined = Concatenate()([q1_encoded, q2_aligned, submult(q1_encoded, q2_aligned)])
    q2_combined = Concatenate()([q2_encoded, q1_aligned, submult(q2_encoded, q1_aligned)])
    compare_layers = [
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
    ]
    q1_compare = time_distributed(q1_combined, compare_layers)
    q2_compare = time_distributed(q2_combined, compare_layers)

    # Aggregate
    q1_rep = apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    q2_rep = apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])

    # Classifier
    merged = Concatenate()([q1_rep, q2_rep])
    dense = BatchNormalization()(merged)
    dense = Dense(dense_dim, activation=activation)(dense)
    dense = Dropout(dense_dropout)(dense)
    dense = BatchNormalization()(dense)
    dense = Dense(dense_dim, activation=activation)(dense)
    dense = Dropout(dense_dropout)(dense)
    out_ = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=[q1, q2], outputs=out_)
    model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy',
                  metrics=['binary_crossentropy', 'accuracy'])
    return model


def esim(pretrained_weights,
         num_shape,
         maxlen=MAX_LEN,
         lstm_dim=300,
         dense_dim=256,
         dense_dropout=0.5,
         spatial_dropout=0.2,
         lr=1e-3,
         cell='lstm',
         clip_gradient=1.):
    RNNcell = CuDNNLSTM if cell == 'lstm' else CuDNNGRU
    # Based on arXiv:1609.06038
    q1 = Input(name='q1', shape=(maxlen,))
    q2 = Input(name='q2', shape=(maxlen,))

    # Embedding - Pretrain
    embedding = create_pretrained_embedding(pretrained_weights, mask_zero=False)
    sd = SpatialDropout1D(spatial_dropout)
    q1_embed = sd(embedding(q1))
    q2_embed = sd(embedding(q2))

    # Encode
    encode = Bidirectional(RNNcell(lstm_dim, return_sequences=True))
    sd = SpatialDropout1D(spatial_dropout)
    q1_encoded = sd(encode(q1_embed))
    q2_encoded = sd(encode(q2_embed))

    # Attention
    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)

    # Compose
    q1_combined = Concatenate()([q1_encoded, q2_aligned, submult(q1_encoded, q2_aligned)])
    q2_combined = Concatenate()([q2_encoded, q1_aligned, submult(q2_encoded, q1_aligned)])

    compose = Bidirectional(RNNcell(lstm_dim, return_sequences=True))
    q1_compare = compose(q1_combined)
    q2_compare = compose(q2_combined)

    # Aggregate
    q1_rep = apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    q2_rep = apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    # bn = BatchNormalization()
    # q1_rep, q2_rep = bn(q1_rep), bn(q2_rep)

    leaks_input = Input(shape=(num_shape,))
    leaks_dense = Dense(dense_dim // 5, use_bias=False)(leaks_input)
    leaks_dense = BatchNormalization()(leaks_dense)
    leaks_dense = Activation('elu')(leaks_dense)

    # Classifier
    dense = Concatenate()([q1_rep, q2_rep, leaks_dense])
    dense = Dropout(dense_dropout)(dense)

    dense = Dense(dense_dim, use_bias=False)(dense)
    dense = BatchNormalization()(dense)
    dense = Activation('elu')(dense)
    dense = Dropout(dense_dropout)(dense)
    dense_1 = dense  # backup for res-link.

    dense = Dense(dense_dim, use_bias=False)(dense)
    dense = BatchNormalization()(dense)
    dense = Activation('elu')(dense)
    dense = Dropout(dense_dropout)(dense)

    dense = Average()([dense, dense_1])  # res-link

    out_ = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=[q1, q2, leaks_input], outputs=out_)
    model.compile(
        optimizer=Adam(lr=lr, clipnorm=clip_gradient, ),
        loss='binary_crossentropy',
    )
    return model




