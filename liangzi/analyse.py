import numpy as np
import pandas as pd

data_path = r'C:/Users/csw/Desktop/python/liangzi/data/'
entbase_path = data_path + '1entbase.csv'
alter_path = data_path + '2alter.csv'
branch_path = data_path + '3branch.csv'
invest_path = data_path + '4invest.csv'
right_path = data_path + '5right.csv'
project_path = data_path + '6project.csv'
lawsuit_path = data_path + '7lawsuit.csv'
breakfaith_path = data_path + '8breakfaith.csv'
recruit_path = data_path + '9recruit.csv'
qualification_path = data_path + '10qualification.csv'
test_path = data_path + 'evaluation_public.csv'
train_path = data_path + 'train.csv'

entbase = pd.read_csv(entbase_path)
alter = pd.read_csv(alter_path)
branch = pd.read_csv(branch_path)
invest = pd.read_csv(invest_path)
right = pd.read_csv(right_path)
project = pd.read_csv(project_path)
lawsuit = pd.read_csv(lawsuit_path)
breakfaith = pd.read_csv(breakfaith_path)
recruit = pd.read_csv(recruit_path)
qualification = pd.read_csv(qualification_path)
test = pd.read_csv(test_path)
train = pd.read_csv(train_path)

print('训练集样本个数：{}'.format(train.shape[0]))
print('测试集样本个数：{}'.format(test.shape[0]))
print()
print('alter记录个数：{}'.format(alter.shape[0]))
print('alter样本个数：{}'.format(alter['id'].nunique()))
print()
print('entbase记录个数：{}'.format(entbase.shape[0]))
print('entbase样本个数：{}'.format(entbase['id'].nunique()))
print()
print('branch记录个数：{}'.format(branch.shape[0]))
print('branch样本个数：{}'.format(branch['id'].nunique()))
print()
print('invest记录个数：{}'.format(invest.shape[0]))
print('invest样本个数：{}'.format(invest['id'].nunique()))
print()
print('right记录个数：{}'.format(right.shape[0]))
print('right样本个数：{}'.format(right['id'].nunique()))
print()
print('project记录个数：{}'.format(project.shape[0]))
print('project样本个数：{}'.format(project['id'].nunique()))
print()
print('lawsuit记录个数：{}'.format(lawsuit.shape[0]))
print('lawsuit样本个数：{}'.format(lawsuit['id'].nunique()))
print()
print('breakfaith记录个数：{}'.format(breakfaith.shape[0]))
print('breakfaith样本个数：{}'.format(breakfaith['id'].nunique()))
print()
print('recruit记录个数：{}'.format(recruit.shape[0]))
print('recruit样本个数：{}'.format(recruit['id'].nunique()))
print()
print('qualification记录个数：{}'.format(qualification.shape[0]))
print('qualification样本个数：{}'.format(qualification['id'].nunique()))








