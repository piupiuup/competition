


from keras import *

from utils import *




def build_model():
    emb_layer = Embedding(MAX_NB_WORDS + 1, EMBEDDING_DIM, weights=[word_embedding_matrix],
                          input_length=MAX_SEQUENCE_LENGTH, trainable=True)

    # Define inputs
    seq1 = Input(shape=(35,))
    seq2 = Input(shape=(35,))

    # Run inputs through embedding

    seq_embedding_layer = Bidirectional(GRU(256, recurrent_dropout=0.2, return_sequences=True))
    cnn1d_layer = Conv1D(64, kernel_size=2, padding="valid", kernel_initializer="he_uniform")

    emb1 = emb_layer(seq1)
    x1 = seq_embedding_layer(emb1)
    x1 = cnn1d_layer(x1)

    emb2 = emb_layer(seq2)
    x2 = seq_embedding_layer(emb2)
    x2 = cnn1d_layer(x2)

    avg_pool = GlobalAveragePooling1D()
    max_pool = GlobalMaxPooling1D()
    x1 = concatenate([avg_pool(x1), max_pool(x1)])
    x2 = concatenate([avg_pool(x2), max_pool(x2)])

    merge_layer = multiply([x1, x2])
    merge_layer = Dropout(0.2)(merge_layer)
    dense1_layer =Dense(64, activation='relu')(merge_layer)
    ouput_layer = Dense(1, activation='sigmoid')(dense1_layer)

    # model = Model(inputs=[seq1, seq2, magic_input, distance_input], outputs=pred)
    model = Model(inputs=[seq1, seq2], outputs=ouput_layer)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', precision, recall, f1_score])

    return model

if __name__ == '__main__':

    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    import pandas as pd
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    que = pd.read_csv('data/question.csv')

    word_dict = {}
    char_dict = {}

    with open('data/word_embed.txt') as f:
        for line in f.readlines():
            s = line.strip('\n').split(' ')
            word_dict[s[0]] = [float(v) for v in s[1:]]

    with open('data/char_embed.txt') as f:
        for line in f.readlines():
            s = line.strip('\n').split(' ')
            char_dict[s[0]] = [float(v) for v in s[1:]]

    train = pd.merge(train, que[['qid', 'words']], left_on='q1', right_on='qid', how='left')
    train = pd.merge(train, que[['qid', 'words']], left_on='q2', right_on='qid', how='left')
    train.drop(['qid_x', 'qid_y'], axis=1, inplace=True)
    train.columns = ['label', 'q1', 'q2', 'word1', 'word2']

    test = pd.merge(test, que[['qid', 'words']], left_on='q1', right_on='qid', how='left')
    test = pd.merge(test, que[['qid', 'words']], left_on='q2', right_on='qid', how='left')
    test.drop(['qid_x', 'qid_y'], axis=1, inplace=True)
    test.columns = ['q1', 'q2', 'word1', 'word2']

    MAX_NB_WORDS = 11200
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(que['words'])
    word_index = tokenizer.word_index
    print(len(word_index))
    # word_counts = tokenizer.word_counts
    # word_index = {}
    # count = 0
    # for key in word_counts.keys():
    #     if word_counts.get(key) < 10000 and word_counts.get(key) > 1:
    #         count = count + 1
    #         word_index[key] = count
    # print(count)
    # MAX_NB_WORDS = count

    q1_train = tokenizer.texts_to_sequences(train['word1'])
    q2_train = tokenizer.texts_to_sequences(train['word2'])
    q1_test = tokenizer.texts_to_sequences(test['word1'])
    q2_test = tokenizer.texts_to_sequences(test['word2'])

    # 构建embedding层
    EMBEDDING_DIM = 300
    word_embedding_matrix = np.zeros((MAX_NB_WORDS + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = word_dict.get(str(word).upper())
        if embedding_vector is not None:
            word_embedding_matrix[i] = embedding_vector

    # 构建输入张量
    MAX_SEQUENCE_LENGTH = 35
    q1_data_tr = pad_sequences(q1_train, maxlen=MAX_SEQUENCE_LENGTH)
    q2_data_tr = pad_sequences(q2_train, maxlen=MAX_SEQUENCE_LENGTH)
    q1_data_te = pad_sequences(q1_test, maxlen=MAX_SEQUENCE_LENGTH)
    q2_data_te = pad_sequences(q2_test, maxlen=MAX_SEQUENCE_LENGTH)

    label=train['label'].values

    q_concat = np.stack([q1_data_tr, q2_data_tr], axis=1)

    resu = []
    from sklearn.model_selection import StratifiedKFold

    te_pred = np.zeros(q1_data_tr.shape[0])

    for tr, va in StratifiedKFold(n_splits=5).split(q_concat, train['label'].values):
        print("----------------------------------------------------------")
        Q1_train = q_concat[tr][:, 0]
        Q2_train = q_concat[tr][:, 1]
        Q1_test = q_concat[va][:, 0]
        Q2_test = q_concat[va][:, 1]

        model = build_model()
        hist = model.fit([Q1_train, Q2_train], train['label'].values[tr],
                         validation_data=([Q1_test, Q2_test], train['label'].values[va]), epochs=10, batch_size=1024,
                         shuffle=True)

        preds = model.predict([Q1_test, Q2_test], verbose=1)
        te_pred[va] = preds[:, 0]

        pred = model.predict([q1_data_te, q2_data_te], batch_size=1024)

        avg = [v[0] for v in pred]
        resu.append(avg)

    t_p = pd.DataFrame()
    t_p["x_pre"] = te_pred
    t_p.to_csv("a_a_model_gru_cnn_1.csv", index=False)

    test_pred = np.array(resu).mean(axis=0)
    sub = pd.DataFrame()
    sub["y_pre"] = test_pred
    sub.to_csv("a_a_model_gru_cnn_1.csv", index=False)