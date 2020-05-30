from tool.tool import *
data_path = 'C:/Users/cui/Desktop/python/paipaidai/data/'

# test = pd.read_csv(data_path + 'test.csv')
# train = pd.read_csv(data_path + 'train.csv')

train = pd.read_csv(data_path + 'train.csv')
test = train[-40000:]
train = train[:-40000]
train_ = train.copy()
train_['q1'] = train['q2']
train_['q2'] = train['q1']
stat = train.append(train_)


white_q_dict = {q:set() for q in stat['q1']}
for q1,q2,label in tqdm(zip(stat['q1'].values,stat['q2'].values,stat['label'].values)):
    if label==1:
        white_q_dict[q1].add(q2)
white_q_dict_1 = {k:v for k,v in white_q_dict.items() if len(v)>0}
white_q_dict_2 = {k:v for k,v in white_q_dict.items() if len(v)==0}

# fg(q_dict,test)
black_q_dict = {q:set() for q in stat['q1']}
for q1,q2,label in tqdm(zip(stat['q1'].values,stat['q2'].values,stat['label'].values)):
    if label==0:
        black_q_dict[q1].add(q2)

q_flag = {k:-1 for k in white_q_dict_1}
i = 0
for k,v in white_q_dict_1.items():
    if q_flag[k] != -1:
        continue
    q_flag[k] = i
    s_add = v
    while len(s_add) > 0:
        new_s_add = set()
        for s_a in s_add:
            new_s_add.update(white_q_dict_1[s_a])
            q_flag[s_a] = i
        s_add = [q for q in new_s_add if q_flag[q]==-1]
    if i%100==0:
        print(i)
    i += 1
for k,v in white_q_dict_2.items():
    q_flag[k] = i
    i += 1

D = defaultdict(set)
for k,v  in q_flag.items():
    D[v].add(k)

black_D = defaultdict(set)
for k,v in D.items():
    for q in v:
        black_s = set()
        for i in black_q_dict[q]:
            black_s.add(q_flag[i])
    for g in black_s:
        black_D[k].update(D[g])

def fg(q_flag,D,black_D,test):
pred =[]
for q1,q2 in tqdm(zip(test['q1'].values,test['q2'].values)):
    p = np.nan
    if (q1 in q_flag):
        if q2 in D[q_flag[q1]]:
            p = 1
        if q2 in black_D[q_flag[q1]]:
            p = 0
    pred.append(p)
test['pred'] = pred
print('覆盖率{}'.format(sum(~test['pred'].isnull())/40000))
print('准确率{}'.format(sum(test['label']==test['pred']) / test.shape[0]))