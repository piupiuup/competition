import pandas as pd
import numpy as np

data_path = 'C:/Users/csw/Desktop/python/zillow/data/'
prop_path = data_path + 'properties_2016.csv'
sample_path = data_path + 'sample_submission.csv'
train_path = data_path + 'train_2016_v2.csv'

prop_df = pd.read_csv(prop_path)
submission = pd.read_csv(sample_path)
train_df = pd.read_csv(train_path)