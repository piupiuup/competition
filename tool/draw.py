import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
# 统计分布直方图
plt.figure(figsize=(12,8))
sns.distplot(data, bins=50, kde=False)
plt.xlabel('logerror', fontsize=12)
plt.show()

# 统计分布直方图
plt.figure(figsize=(12,8))
sns.distplot(data, bins=50, kde=False)
plt.xlabel('logerror', fontsize=12)
plt.show()


# 散点图
plt.figure(figsize=(8,6))
plt.scatter(range(train_df.shape[0]), np.sort(train_df.logerror.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('logerror', fontsize=12)
plt.show()


#
plt.figure(figsize=(12, 8))
plt.hist(train_genes.values, bins=50, log=True)
plt.xlabel('Number of times Gene appeared', fontsize=12)
plt.ylabel('log of Count', fontsize=12)
plt.show()


# 分析特征的信息量
def analyse_plot(data,name='id',label='label'):
    import matplotlib.pyplot as plt
    data_temp = data.sort_values(name)
    plt.figure(figsize=(8, 6))
    plt.scatter(range(data.shape[0]), data_temp[label].cumsum())

# 分析特征的信息量
def analyse_plot2(data,name='id',label='label', factor=10):
    import matplotlib.pyplot as plt
    grouping = pd.cut(data[name],factor)
    rate = data.groupby(grouping)[label].agg({'sum':'sum',
                                              'count':'count'})
    rate['rate'] = rate['sum']/rate['count']
    rate['rate'].plot()


# 绘画特征重要性
ind = np.arange(20)
width = 0.9
fig, ax = plt.subplots(figsize=(12,8))
rects = ax.barh(ind, a[:20].values, color='blue')
ax.set_yticks(ind)
ax.set_yticklabels(a[:20].index, rotation='horizontal')
ax.set_xlabel("importance of features")
plt.show()


# 密度图
fig, ax = plt.subplots(figsize=(12,8))
sns.distplot(train2[(train2.prev=='p') & (train2.label==1)].id)
sns.distplot(train2[(train2.prev=='s') & (train2.label==1)].id)
sns.distplot(train[train.TARGET==1].EID)
ax.set_ylabel("退出率")


# 热力图
f, ax = plt.subplots(figsize = (10, 4))
cmap = sns.cubehelix_palette(start = 1, rot = 3, gamma=0.8, as_cmap = True)
sns.heatmap(pt, cmap = cmap, linewidths = 0.05, ax = ax)
ax.set_title('Amounts per kind and region')
ax.set_xlabel('region')
ax.set_ylabel('kind')





