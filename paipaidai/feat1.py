from tool.tool import *
data_path = 'C:/Users/cui/Desktop/python/paipaidai/data/'

test = pd.read_csv(data_path + 'test.csv')
train = pd.read_csv(data_path + 'train.csv')
question = pd.read_csv(data_path + 'question.csv')
char = pd.read_csv(data_path + 'char_embed.txt')
word = pd.read_csv(data_path + 'word_embed.txt')


data = pd.concat([train[['q1', 'q2']], \
                  test[['q1', 'q2']]], axis=0).reset_index(drop='index')
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
train['q1_q2_intersect'] = train.apply(q1_q2_intersect, axis=1, raw=True)
train['q1_freq'] = train.apply(q1_freq, axis=1, raw=True)
train['q2_freq'] = train.apply(q2_freq, axis=1, raw=True)


predictors = [c for c in train.columns if c not in ['q1','q2','id','label']]
trian_feat = train[:train.shape[0]//2]
test_feat = train[train.shape[0]//2:]

print('开始训练...')
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'max_depth': 8,
    'num_leaves': 150,
    'learning_rate': 0.05,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5,
    'verbose': 0,
    'seed': 66,
}
lgb_train = lgb.Dataset(trian_feat[predictors], trian_feat.label)
lgb_test = lgb.Dataset(test_feat[predictors], test_feat.label)

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10000,
                verbose_eval = 50,
                valid_sets=lgb_test,
                early_stopping_rounds=100)
pred1 = gbm.predict(test_feat[predictors])
print(log_loss(test_feat['label'],pred1))
# a = test_feat['q1_freq'].copy()
# test_feat['q1_freq'] = test_feat['q2_freq']
# test_feat['q2_freq'] = a
# pred2 = gbm.predict(test_feat[predictors])
# print(log_loss(test_feat['label'],pred2))
# print(log_loss(test_feat['label'],(pred1+pred2)/2))

train = pd.read_csv(data_path + 'train.csv')
test = train[-40000:]
train = train[:-40000]
train_ = train.copy()
train_['q1'] = train['q2']
train_['q2'] = train['q1']
stat = train.append(train_)
a = stat[stat['label']==1]
a.sort_values('q1',inplace=True)
b = stat[stat['label']==0]
b.sort_values('q1',inplace=True)

def fg(q_dict,test):
    pred =[]
    for q1,q2 in tqdm(zip(test['q1'].values,test['q2'].values)):
        p = np.nan
        if (q1 in q_dict[q2]) or (q2 in q_dict[q1]):
            pred.append(True)
        else:
            pred.append(np.nan)
    test['pred'] = pred
    test = test[~test['pred'].isnull()]
    print('覆盖率{}'.format(test.shape[0]/40000))

q_dict = defaultdict(set)
for q1,q2 in tqdm(zip(a['q1'].values,a['q2'].values)):
    q_dict[q1].add(q2)
# fg(q_dict,test)
black_q_dict = defaultdict(set)
for q1,q2 in tqdm(zip(b['q1'].values,b['q2'].values)):
    black_q_dict[q1].add(q2)


q_flag = {k:True for k in  q_dict}
def set_q(s_add, q_dict, i):
    new_s_add = set()
    for s_a in s_add:
        for q in q_dict[s_a]:
            if q_flag[q]==True:
                new_s_add.add(q)
        q_flag[s_a] = i
        q_dict.pop(s_a)
    print(len(q_dict))
    return new_s_add, q_dict, q_flag

q_flag = {k:True for k in  q_dict}
i = 0
while len(q_dict)>0:
    k,v = list(q_dict.items())[0]
    q_dict.pop(k)
    q_flag[k] = i
    s_add = v
    while len(s_add) > 0:
        new_s_add = set()
        for s_a in s_add:
            for q in q_dict[s_a]:
                if q_flag[q] == True:
                    new_s_add.add(q)
            q_flag[s_a] = i
            q_dict.pop(s_a)
        s_add = new_s_add
    print(len(q_dict))
    i += 1



# q_flag = {k:True for k in  train['q1'].values}
def add_dict(s, s_add, q_dict):
    s.update(set(s_add))
    new_s_add = set()
    for s_a in s_add:
        for i in q_dict[s_a]:
            if i in q_dict:
                # print(i)
                new_s_add.add(i)
        q_dict.pop(s_a)
    print(len(q_dict))
    return s, new_s_add, q_dict

# 白名单
L = []
l_temp = 0
l = 1
while len(q_dict)>0:
    k,v = list(q_dict.items())[0]
    q_dict.pop(k)
    s = set()
    s.add(k)
    s_add = v
    while len(s_add) > 0:
        s, s_add, q_dict = add_dict(s, s_add, q_dict)
    L.append(s)
D = {i:s for i,s in enumerate(L)}
D2 = dict()
for k,v in D.items():
    for q in v:
        D2[q] = k


# 黑名单
black_L = []
for s in tqdm(L):
    bs = set()
    bs2 = set()
    for q in s:
        bs.update(black_q_dict[q])
        bs2.add(D2[q])
    for q in bs2:
        bs.update(D[q])
    black_L.append(bs)
black_D = {i:s for i,s in enumerate(black_L)}


pred = []
for q1,q2 in tqdm(zip(test['q1'].values, test['q2'].values)):
    p = np.nan
    if q1 in D2:
        if q2 in D[D2[q1]]:
            p = True
        if q2 in black_D[D2[q1]]:
            p = False
    pred.append(p)















