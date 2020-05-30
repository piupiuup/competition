import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from scipy.sparse import hstack
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction.text import TfidfVectorizer


# tfidf 之前增加正样本一倍 可以提高7个千分位
# ngram_range（1，1） 和 （1，2）*0.5 融合可以提高4个千分位
# lgb*0.5 和 lr 融合可以提及高一个千分位

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

data_path = 'C:/Users/csw/Desktop/python/toxic/data/'


train = pd.read_csv(data_path + 'train.csv').fillna(' ')
test = train[80000:]
train = train[:80000]
# for i in range(1):
#     train = pd.concat([train,train[train['toxic']==1]],axis=0)

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
predictions = pd.DataFrame({'id': test['id']})
classifier = LogisticRegression(solver='liblinear',penalty='l1')
classifier.fit(train_features, train['toxic'])
predictions['lr_{}'.format('toxic')] = classifier.predict_proba(test_features)[:, 1]
score = roc_auc_score(test['toxic'] ,predictions['lr_{}'.format('toxic')] )
print('{0}的线下的分：{1}'.format('toxic',score))

# cv_loss = np.mean(cross_val_score(classifier, train_features, train_target, cv=5, scoring='roc_auc'))
# losses.append(cv_loss)
# print('Total CV score is {}'.format(np.mean(losses)))
for col in class_names[:1]:
    classifier.fit(train_features, train[col])
    predictions['lr_{}'.format(col)] = classifier.predict_proba(test_features)[:, 1]
    score = roc_auc_score(test[col] ,predictions['lr_{}'.format(col)] )
    print('{0}的线下的分：{1}'.format(col,score))
    losses.append(score)
print(np.mean(losses))

train_predictions = pd.DataFrame({'id': train['id']})
for col in class_names:
    classifier.fit(test_features, test[col])
    train_predictions['lr_{}'.format(col)] = classifier.predict_proba(train_features)[:, 1]

losses2 = []
for col in class_names:
    classifier.fit(train_predictions.drop('id',axis=1), train[col])
    predictions['lr_{}'.format(col)] = classifier.predict_proba(predictions.drop('id',axis=1))[:, 1]
    score = roc_auc_score(test[col] ,predictions['lr_{}'.format(col)] )
    print('{0}的线下的分：{1}'.format(col,score))
    losses2.append(score)
print(np.mean(losses2))




train_target = train['toxic']
model = LogisticRegression(solver='sag')
sfm = SelectFromModel(model, threshold=0.2)
print(train_features.shape)
train_sparse_matrix = sfm.fit_transform(train_features, train_target)
print(train_sparse_matrix.shape)
test_sparse_matrix = sfm.transform(test_features)

##################### lgb ##################
print('开始训练...')
params = {'learning_rate': 0.2,
          'application': 'binary',
          'num_leaves': 31,
          'verbosity': -1,
          'metric': 'auc',
          'data_random_seed': 2,
          'bagging_fraction': 0.8,
          'feature_fraction': 0.6,
          'nthread': -1,
          'lambda_l1': 1,
          'lambda_l2': 1}
lgb_train = lgb.Dataset(train_features, train['toxic'])
lgb_test = lgb.Dataset(test_features,test['toxic'])

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10000,
                valid_sets=lgb_test,
                verbose_eval = 10,
                early_stopping_rounds=10)
predictions['lgb_toxic'] = gbm.predict(test_features)
print('toxic的线下的分：{}'.format(roc_auc_score(test['toxic'] ,predictions['lgb_toxic'] )))



##################### xgb ##################
print('开始训练...')
xgb_train = xgb.DMatrix(train_features, train['toxic'])
xgb_test = xgb.DMatrix(test_features,test['toxic'])

xgb_params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'max_depth': 6,
              'lambda':5,
              'subsample': 0.8,
              'colsample_bytree': 0.7,
              'min_child_weight': 10,  # 8~10
              'eta': 0.01,
              'seed':66,
              # 'nthread':12
              }
params['silent'] = 1
watchlist = [(xgb_train, 'train'), (xgb_test, 'eval')]
model = xgb.train(params, xgb_test, 5000, watchlist, early_stopping_rounds=20,verbose_eval = 30)
predictions['lgb_toxic'] = gbm.predict(test_features)
print('toxic的线下的分：{}'.format(roc_auc_score(test['toxic'] ,predictions['lgb_toxic'] )))




