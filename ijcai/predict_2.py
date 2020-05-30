import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

def predict_2(train_x,train_y,test):
    train_x = train_x.fillna(0)
    train_y = train_y.fillna(0)
    test = test.fillna(0)

    ET = ExtraTreesRegressor(n_estimators=1200, random_state=1, n_jobs=-1, min_samples_split=2,
                             min_samples_leaf=2, max_depth=25, max_features=140)
    ET.fit(train_x, train_y)
    result = ET.predict(test)

    return result