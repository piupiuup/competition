from tool.tool import *
data_path = 'C:/Users/cui/Desktop/python/paipaidai/data/'

test = pd.read_csv(data_path + 'test.csv')
train = pd.read_csv(data_path + 'train.csv')

# train = pd.read_csv(data_path + 'train.csv')
# test = train[-40000:]
# train = train[:-40000]

train_ = train.copy()
train_['q1'] = train['q2']
train_['q2'] = train['q1']
stat = train.append(train_)
a = stat[stat['label']==1]
a.sort_values('q1',inplace=True)
b = stat[stat['label']==0]
b.sort_values('q1',inplace=True)

def fg(q_dict,black_q_dict,test):
    pred =[]
    for q1,q2 in tqdm(zip(test['q1'].values,test['q2'].values)):
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
    test['pred'] = pred
    # test = test[~test['pred'].isnull()]
    print('覆盖率{}'.format(sum(~test['pred'].isnull())/test.shape[0]))
    if 'label' in test.columns:
        print('准确率{}'.format(sum(test['label']==test['pred']) / sum(~test['pred'].isnull())))

q_dict = defaultdict(set)
for q1,q2 in tqdm(zip(a['q1'].values,a['q2'].values)):
    q_dict[q1].add(q2)
q_dict_flag = {q:True for q in q_dict}

black_q_dict = defaultdict(set)
for q1,q2 in tqdm(zip(b['q1'].values,b['q2'].values)):
    black_q_dict[q1].add(q2)

for i in range(10):
    new_q_dict = defaultdict(set)
    temp = q_dict.copy()
    for k,v in tqdm(q_dict.items()):
        if q_dict_flag[k] == False:
            new_q_dict[k] = temp[k]
            continue
        new_v = v.copy()
        for i in v:
            new_v.update(temp[i])
        new_v.remove(k)
        if len(new_v)==len(temp[k]):
            q_dict_flag[k] = False
        new_q_dict[k] = new_v
    q_dict = new_q_dict


new_black_q_dict = defaultdict(set)
temp = black_q_dict
for k, v in tqdm(black_q_dict.items()):
    new_v = v.copy()
    for i in v:
        if i in q_dict:
            new_v.update(q_dict[i])
    new_black_q_dict[k] = new_v
black_q_dict = new_black_q_dict

new_black_q_dict = defaultdict(set)
temp = black_q_dict
for k, v in tqdm(black_q_dict.items()):
    new_v = v.copy()
    if k in q_dict:
        for i in q_dict[k]:
            if i in temp:
                new_v.update(temp[i])
    new_black_q_dict[k] = new_v
black_q_dict = new_black_q_dict

fg(q_dict,black_q_dict,test)





