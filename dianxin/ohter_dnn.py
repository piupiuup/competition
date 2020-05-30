from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import Callback
#from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras.callbacks import EarlyStopping
import keras.backend as K
from sklearn.preprocessing import OneHotEncoder

def f1_score(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # Calculate f1_score
    f1_score = 2 * c1 / (c2 + c3)
    return f1_score

early_stoping = EarlyStopping(patience=10)

data_path = 'C:/Users/cui/Desktop/python/dianxin/data/'
d = {89950166: 1, 89950167: 2, 89950168: 5, 90063345: 0, 90109916: 4,
 90155946: 8, 99999825: 10, 99999826: 7, 99999827: 6, 99999828: 3, 99999830: 9}
rd = {0: 90063345, 1: 89950166, 2: 89950167, 3: 99999828, 4: 90109916,
 5: 89950168, 6: 99999827, 7: 99999826, 8: 90155946, 9: 99999830, 10: 99999825}

train=pd.read_csv(data_path + 'train.csv')
test=pd.read_csv(data_path + 'test.csv')
data = train.append(test)

##需要独热编码的特征
feat_one_hot=['contract_type','net_service']
data = pd.get_dummies(data,columns=['contract_type'], dummy_na=-1)


feat_normal=[item for item in data.columns if item not in (feat_one_hot+['user_id','current_service','label'])]

# 对数值进行标准化
sc=StandardScaler()
data[feat_normal]=sc.fit_transform(data[feat_normal].fillna(-1))
target = pd.get_dummies(data['current_service'])
#x_train,x_test,y_train,y_test=train_test_split(train_,target,random_state=1)


dnn_Model=Sequential()
dnn_Model.add(Dense(512,input_dim=34,activation='relu'))
dnn_Model.add(Dropout(0.5))
dnn_Model.add(Dense(1024,activation='relu'))
dnn_Model.add(Dropout(0.5))
dnn_Model.add(Dense(1024,activation='relu'))
dnn_Model.add(Dropout(0.5))
dnn_Model.add(Dense(1024,activation='relu'))
dnn_Model.add(Dropout(0.5))
dnn_Model.add(Dense(1024,activation='relu'))
dnn_Model.add(Dropout(0.5))
dnn_Model.add(Dense(1024,activation='relu'))
dnn_Model.add(Dropout(0.5))
dnn_Model.add(Dense(11,activation='softmax'))


dnn_Model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy',f1_score])

predictors = [c for c in data.columns if c not in ['user_id','current_service', 'label']]
train_ = data[data['user_id'].isin(train['user_id'].values)][predictors].copy()
test_ = data[data['user_id'].isin(test['user_id'].values)][predictors].copy()
target = target[:train.shape[0]].copy()

dnn_Model.fit(train_,target,batch_size=10000,epochs=80,callbacks=[early_stoping],validation_split=0.2)

preds=dnn_Model.predict(test_)
preds=[np.argmax(i) for i in preds]
preds=la.inverse_transform(preds)
results['predict']=preds
results.to_csv('dnn_results.csv',index=0)



