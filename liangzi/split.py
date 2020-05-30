import numpy as np
import pandas as pd



data_path = r'C:/Users/csw/Desktop/python/liangzi/data/'
eval_data_path = r'C:/Users/csw/Desktop/python/liangzi/data/eval/'


entbase = pd.read_csv(data_path + '1entbase.csv')
alter = pd.read_csv(data_path + '2alter.csv')
branch = pd.read_csv(data_path + '3branch.csv')
invest = pd.read_csv(data_path + '4invest.csv')
right = pd.read_csv(data_path + '5right.csv')
project = pd.read_csv(data_path + '6project.csv')
lawsuit = pd.read_csv(data_path + '7lawsuit.csv')
breakfaith = pd.read_csv(data_path + '8breakfaith.csv')
recruit = pd.read_csv(data_path + '9recruit.csv')
qualification = pd.read_csv(data_path + '10qualification.csv')
train = pd.read_csv(data_path + 'train.csv')


select_id = train['id'].tolist()
test = train[120000:]
train = train[:120000]

entbase = entbase[entbase['id'].isin(select_id)]
alter = alter[alter['id'].isin(select_id)]
branch = branch[branch['id'].isin(select_id)]
invest = invest[invest['id'].isin(select_id)]
right = right[right['id'].isin(select_id)]
project = project[project['id'].isin(select_id)]
lawsuit = lawsuit[lawsuit['id'].isin(select_id)]
breakfaith = breakfaith[breakfaith['id'].isin(select_id)]
recruit = recruit[recruit['id'].isin(select_id)]
qualification = qualification[qualification['id'].isin(select_id)]
train = train[train['id'].isin(select_id)]

entbase.to_csv(eval_data_path + '1entbase.csv',index=False,encoding='utf-8')
alter.to_csv(eval_data_path + '2alter.csv',index=False,encoding='utf-8')
branch.to_csv(eval_data_path + '3branch.csv',index=False,encoding='utf-8')
invest.to_csv(eval_data_path + '4invest.csv',index=False,encoding='utf-8')
right.to_csv(eval_data_path + '5right.csv',index=False,encoding='utf-8')
project.to_csv(eval_data_path + '6project.csv',index=False,encoding='utf-8')
lawsuit.to_csv(eval_data_path + '7lawsuit.csv',index=False,encoding='utf-8')
breakfaith.to_csv(eval_data_path + '8breakfaith.csv',index=False,encoding='utf-8')
recruit.to_csv(eval_data_path + '9recruit.csv',index=False,encoding='utf-8')
qualification.to_csv(eval_data_path + '10qualification.csv',index=False,encoding='utf-8')
test[['id']].to_csv(eval_data_path + 'evaluation_public.csv',index=False,encoding='utf-8')
train.to_csv(eval_data_path + 'train.csv',index=False,encoding='utf-8')

