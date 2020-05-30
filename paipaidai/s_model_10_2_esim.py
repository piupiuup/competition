import numpy as np
from keras import *

from paipaidai.utils import *

from keras.activations import softmax


def substract(input_1, input_2):
    "Substract element-wise"
    neg_input_2 = Lambda(lambda x: -x, output_shape=unchanged_shape)(input_2)
    out_ = Add()([input_1, neg_input_2])
    return out_

def apply_multiple(input_, layers):
    "Apply layers to input then concatenate result"
    if not len(layers) > 1:
        raise ValueError('Layers list should contain more than s_model_1 layer')
    else:
        agg_ = []
        for layer in layers:
            agg_.append(layer(input_))
        out_ = Concatenate()(agg_)
    return out_

def submult(input_1, input_2):
    "Get multiplication and subtraction then concatenate results"
    mult = Multiply()([input_1, input_2])
    sub = substract(input_1, input_2)
    diff = Lambda(lambda x: K.abs(x[0] - x[1]))([input_1, input_2])
    out_ = Concatenate()([diff,sub, mult])
    return out_

def unchanged_shape(input_shape):
    "Function for Lambda layer"
    return input_shape

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

def att_parse(q1_embed,q2_embed):


    encode = Bidirectional(CuDNNLSTM(300,return_sequences=True))
    sd = SpatialDropout1D(0.3)

    q1_encoded = sd(encode(q1_embed))
    q2_encoded = sd(encode(q2_embed))

    translate = TimeDistributed(Dense(300, activation="relu"))
    q1_encoded = translate(q1_encoded)
    q2_encoded = translate(q2_encoded)

    # Attention
    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)

    # Compose
    q1_combined = Concatenate()([q1_encoded, q2_aligned, submult(q1_encoded, q2_aligned)])
    q2_combined = Concatenate()([q2_encoded, q1_aligned, submult(q2_encoded, q1_aligned)])

    #compose = Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    compose = Bidirectional(CuDNNLSTM(300,return_sequences=True))
    sd1 = SpatialDropout1D(0.3)

    q1_compare = sd1(compose(q1_combined))
    q2_compare = sd1(compose(q2_combined))

    # Aggregate
    q1_rep = apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    q2_rep = apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])

    # Classifier
    merged = Concatenate()([q1_rep, q2_rep])

    return merged



def build_model():
    emb_layer = Embedding(MAX_NB_WORDS + 1, EMBEDDING_DIM, weights=[word_embedding_matrix],
                          input_length=50, trainable=False)

    # Define inputs
    seq1 = Input(shape=(50,))
    seq2 = Input(shape=(50,))

    # Run inputs through embedding
    #bn = BatchNormalization(axis=2)
    bn=SpatialDropout1D(0.3)
    q1_embed = bn(emb_layer(seq1))
    q2_embed = bn(emb_layer(seq2))

    merged = att_parse(q1_embed, q2_embed)

    merge = Dropout(0.2)(merged)

    x = Dense(300, use_bias=False)(merge)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Dropout(0.2)(x)
    x1 = x

    x = Dense(300, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x2 = Dropout(0.2)(x)

    vec = Average()([x1, x2])
    pred = Dense(1, activation='sigmoid')(vec)

    # model = Model(inputs=[seq1, seq2, magic_input, distance_input], outputs=pred)
    model = Model(inputs=[seq1, seq2], outputs=pred)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

if __name__ == '__main__':
    import os
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    import pandas as pd

    os.chdir(r'C:\Users\cui\Desktop\python\paipaidai')
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    test = test.reindex(columns=["label", "q1", "q2"], fill_value=1)
    que = pd.read_csv('data/question.csv')

    word_dict = {}
    #char_dict = {}

    with open('data/word_embed.txt') as f:
        for line in f.readlines():
            s = line.strip('\n').split(' ')
            word_dict[s[0]] = [float(v) for v in s[1:]]

    with open('data/char_embed.txt') as f:
        for line in f.readlines():
            s = line.strip('\n').split(' ')
            word_dict[s[0]] = [float(v) for v in s[1:]]


    train = pd.merge(train, que[['qid', 'words',"chars"]], left_on='q1', right_on='qid', how='left')
    train = pd.merge(train, que[['qid', 'words',"chars"]], left_on='q2', right_on='qid', how='left')
    train.drop(['qid_x', 'qid_y'], axis=1, inplace=True)

    train.columns = ['label','q1', 'q2', 'word1', 'char1','word2','char2']

    test = pd.merge(test, que[['qid', 'words','chars']], left_on='q1', right_on='qid', how='left')
    test = pd.merge(test, que[['qid', 'words','chars']], left_on='q2', right_on='qid', how='left')
    test.drop(['qid_x', 'qid_y'], axis=1, inplace=True)
    test.columns = ['label','q1', 'q2', 'word1', 'char1','word2','char2']

    datas = que['words'] + " " + que['chars']
    MAX_NB_WORDS = 15200
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(datas)
    word_index = tokenizer.word_index
    print(len(word_index))


    q1_train = tokenizer.texts_to_sequences(train['word1'])
    q2_train = tokenizer.texts_to_sequences(train['word2'])
    q1_test = tokenizer.texts_to_sequences(test['word1'])
    q2_test = tokenizer.texts_to_sequences(test['word2'])

    c1_train = tokenizer.texts_to_sequences(train['char1'])
    c2_train = tokenizer.texts_to_sequences(train['char2'])
    c1_test = tokenizer.texts_to_sequences(test['char1'])
    c2_test = tokenizer.texts_to_sequences(test['char2'])


    EMBEDDING_DIM = 300
    word_embedding_matrix = np.zeros((MAX_NB_WORDS + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = word_dict.get(str(word).upper())
        if embedding_vector is not None:
            word_embedding_matrix[i] = embedding_vector

    # char_embedding_matrix = np.zeros((MAX_NB_CHARS + 1, EMBEDDING_DIM))
    # for char, i in char_index.items():
    #     if i > MAX_NB_CHARS:
    #         continue
    #     embedding_vector = char_dict.get(str(char).upper())
    #     if embedding_vector is not None:
    #         char_embedding_matrix[i] = embedding_vector


    MAX_SEQUENCE_LENGTH = 25
    q1_data_tr = pad_sequences(q1_train, maxlen=MAX_SEQUENCE_LENGTH)
    q2_data_tr = pad_sequences(q2_train, maxlen=MAX_SEQUENCE_LENGTH)
    q1_data_te = pad_sequences(q1_test, maxlen=MAX_SEQUENCE_LENGTH)
    q2_data_te = pad_sequences(q2_test, maxlen=MAX_SEQUENCE_LENGTH)

    MAX_SEQUENCE_LENGTH_CHAR = 25
    c1_data_tr = pad_sequences(c1_train, maxlen=MAX_SEQUENCE_LENGTH_CHAR)
    c2_data_tr = pad_sequences(c2_train, maxlen=MAX_SEQUENCE_LENGTH_CHAR)
    c1_data_te = pad_sequences(c1_test, maxlen=MAX_SEQUENCE_LENGTH_CHAR)
    c2_data_te = pad_sequences(c2_test, maxlen=MAX_SEQUENCE_LENGTH_CHAR)

    print(q1_data_tr)

    q1_data_tr=np.hstack((q1_data_tr,c1_data_tr))
    print(q1_data_tr)
    q2_data_tr=np.hstack((q2_data_tr,c2_data_tr))
    q1_data_te=np.hstack((q1_data_te,c1_data_te))
    q2_data_te = np.hstack((q2_data_te, c2_data_te))

    label=train['label'].values

    q_concat = np.stack([q1_data_tr, q2_data_tr], axis=1)

    resu = []
    from sklearn.model_selection import StratifiedKFold

    te_pred = np.zeros(q1_data_tr.shape[0])

    count=0
    for tr, va in StratifiedKFold(n_splits=5).split(q_concat, train['label'].values):
        count=count+1
        print("----------------------------------------------------------",count)
        Q1_train = q_concat[tr][:, 0]
        Q2_train = q_concat[tr][:, 1]


        Q1_test = q_concat[va][:, 0]
        Q2_test = q_concat[va][:, 1]


        model = build_model()

        from keras.callbacks import EarlyStopping, ModelCheckpoint
        early_stopping = EarlyStopping(monitor="val_loss", patience=5)
        best_model_path = str(count) + "best_s_model_10_weight.h5"
        model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, save_weights_only=True)

        hist = model.fit([Q1_train, Q2_train], train['label'].values[tr],
                         validation_data=([Q1_test, Q2_test], train['label'].values[va]), epochs=100, batch_size=1024,
                         shuffle=True,callbacks=[model_checkpoint,early_stopping])

        model.load_weights(best_model_path)
        preds = model.predict([Q1_test, Q2_test], verbose=1)
        te_pred[va] = preds[:, 0]

        pred = model.predict([q1_data_te, q2_data_te], batch_size=1024)

        avg = [v[0] for v in pred]
        resu.append(avg)
    path = "stack1/new/"
    model_name="model_10_esim_"

    t_p = pd.DataFrame()
    t_p[model_name+"x_pre"] = te_pred
    t_p.to_csv(path+model_name+"x.csv", index=False)

    test_pred=np.array(resu).mean(axis=0)
    print(test_pred)
    sub = pd.DataFrame()
    sub[model_name+"y_pre"] = test_pred
    sub.to_csv(path+model_name+"y.csv", index=False)
