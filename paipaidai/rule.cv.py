from tool.tool import *
data_path = 'C:/Users/cui/Desktop/python/paipaidai/data/'

Train = pd.read_csv(data_path + 'train.csv')



print('开始CV 5折训练...')
scores = []
t0 = time.time()
mean_score = []
train_preds = np.zeros(len(Train))
kf = KFold(len(Train), n_folds = 5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    train = Train.iloc[train_index]
    test = Train.iloc[test_index]

    train_ = train.copy()
    train_['q1'] = train['q2']
    train_['q2'] = train['q1']
    stat = train.append(train_)
    a = stat[stat['label']==1]
    a.sort_values('q1',inplace=True)
    b = stat[stat['label']==0]
    b.sort_values('q1',inplace=True)

    q_dict = defaultdict(set)
    for q1,q2 in tqdm(zip(a['q1'].values,a['q2'].values)):
        q_dict[q1].add(q2)
    black_q_dict = defaultdict(set)
    for q1,q2 in tqdm(zip(b['q1'].values,b['q2'].values)):
        black_q_dict[q1].add(q2)

    for i in range(3):
        new_q_dict = defaultdict(set)
        temp = q_dict.copy()
        for k,v in tqdm(q_dict.items()):
            new_v = v.copy()
            for i in v:
                new_v.update(temp[i])
            new_v.remove(k)
            new_q_dict[k] = new_v
        q_dict = new_q_dict
        new_black_q_dict = defaultdict(set)
        temp = black_q_dict
        for k, v in tqdm(black_q_dict.items()):
            new_v = v.copy()
            for i in v:
                if i in q_dict:
                    new_v.update(q_dict[i])
            if k in q_dict:
                for i in q_dict[k]:
                    if i in temp:
                        new_v.update(temp[i])
            new_black_q_dict[k] = new_v
        black_q_dict = new_black_q_dict

    pred = []
    for q1, q2 in tqdm(zip(test['q1'].values, test['q2'].values)):
        p = np.nan
        if (q1 in q_dict):
            if (q2 in q_dict[q1]):
                p = 1
        if (q2 in q_dict):
            if (q1 in q_dict[q2]):
                p = 1
        if (q1 in black_q_dict):
            if (q2 in black_q_dict[q1]):
                p = 0
        if (q2 in black_q_dict):
            if (q1 in black_q_dict[q2]):
                p = 0
        pred.append(p)
    train_preds[test_index] += pred



#
# print('开始CV 5折训练...')
# scores = []
# t0 = time.time()
# mean_score = []
# train_preds = np.zeros(len(train))
# kf = KFold(len(train), n_folds = 5, shuffle=True, random_state=520)
# for i, (train_index, test_index) in enumerate(kf):
#     train = train.iloc[train_index]
#     test = train.iloc[test_index]
#     train_preds[test_index] += train_preds_sub









