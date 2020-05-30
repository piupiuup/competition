import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from scipy.special import logit, expit
from tqdm import tqdm

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

data_path = 'C:/Users/csw/Desktop/python/toxic/data/'


train = pd.read_csv(data_path + 'train.csv').fillna(' ')
test = train[80000:]
train = train[:80000]

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 1),
    max_features=15000)
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    ngram_range=(1, 5),
    max_features=20000)
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)

train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])
# train_features = train_word_features
# test_features = test_word_features

losses = []
predictions = {'id': test['id']}
train_target = train['toxic']
classifier = LogisticRegression(solver='liblinear',penalty='l1')

cv_loss = np.mean(cross_val_score(classifier, train_features, train_target, cv=5, scoring='roc_auc'))
losses.append(cv_loss)
print('Total CV score is {}'.format(np.mean(losses)))

classifier.fit(train_features, train_target)
predictions['toxic'] = classifier.predict_proba(test_features)[:, 1]
print('toxic的线下的分：{}'.format(roc_auc_score(test['toxic'] ,predictions['toxic'] )))
print('Total CV score is {}'.format(np.mean(losses)))




##################### lgb ##################

print('开始训练...')
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'max_depth': 8,
    'num_leaves': 150,
    'learning_rate': 0.05,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': 66,
}
lgb_train = lgb.Dataset(train_features, train['toxic'])
lgb_test = lgb.Dataset(test_features)

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10000,
                valid_sets=lgb_test,
                verbose_eval = 50,
                early_stopping_rounds=100)
predictions['toxic'] = gbm.predict(test_features)
print('toxic的线下的分：{}'.format(roc_auc_score(test['toxic'] ,predictions['toxic'] )))










train_text = train['comment_text'].map(clean_text2)
test_text = test['comment_text'].map(clean_text2)

def feature_lda(train_df, test_df):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    ### Fit transform the tfidf vectorizer ###
    print('fit tfidf')
    tfidf_vec = CountVectorizer(stop_words='english', ngram_range=(1, 3), max_features=10000)
    full_tfidf = tfidf_vec.fit_transform(train_df.values.tolist() + test_df.values.tolist())
    train_tfidf = tfidf_vec.transform(train_df.values.tolist())
    test_tfidf = tfidf_vec.transform(test_df.values.tolist())
    print('LDA')
    no_topics = 20
    lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,
                                    random_state=0).fit(full_tfidf)
    train_lda = pd.DataFrame(lda.transform(train_tfidf))
    test_lda = pd.DataFrame(lda.transform(test_tfidf))

    train_lda.columns = ['lda_' + str(i) for i in range(no_topics)]
    test_lda.columns = ['lda_' + str(i) for i in range(no_topics)]
    return train_lda, test_lda


train_lda, test_lda = feature_lda(train_text, test_text)

from scipy.sparse import csc_matrix

train_lda = csc_matrix(train_lda.values)
test_lda = csc_matrix(test_lda.values)


















