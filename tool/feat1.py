import os
import time
import hashlib
import numpy as np
import pandas as pd


cache_path = 'F:/ijcai_cache/'
inplace = False

############################### 工具函数 ###########################
# 合并节约内存
def concat(L):
    result = None
    for l in L:
        if result is None:
            result = l
        else:
            result[l.columns.tolist()] = l
    return result

############################### 预处理函数 ###########################
def pre_treatment(data,data_key):
    result_path = cache_path + 'data_{}.feature'.format(data_key)
    if os.path.exists(result_path) & 1:
        data = pd.read_feature(result_path)
    else:
        data.reset_index(drop=True,inplace=True)
        data.to_feature(result_path)
    return data


############################### 特征函数 ###########################
# 特征
def get__feat(data,data_key):
    result_path = cache_path + '_feat_{}.feature'.format(data_key)
    if os.path.exists(result_path) & (not inplace):
        feat = pd.read_feature(result_path)
    else:
        data_temp = data.copy()

        feat.to_feature(result_path)
    return feat



# 二次处理特征
def second_feat(result):
    return result

def make_feat(data,data_key):
    t0 = time.time()
    # data_key = hashlib.md5(data.to_string().encode()).hexdigest()
    # print('数据key为：{}'.format(data_key))
    result_path = cache_path + 'feat_set_{}.feature'.format(data_key)
    if os.path.exists(result_path) & 0:
        result = pd.read_feature(result_path, 'w')
    else:
        data = pre_treatment()

        result = [data[['']]]
        print('开始构造特征...')
        result.append(get_context_feat())     # context特征
        result.append(get_user_feat())     # 用户特征
        result.append(get_item_feat())     # 商品特征
        result.append(get_shop_feat())     # 商店特征

        print('开始合并特征...')
        result = concat([])

        result = second_feat(result)
        print('添加label')
        result['label'] = result['label'].map({'POSITIVE': 1, 'NEGATIVE': 0})
        print('存储数据...')
        result.to_feature(result_path)
    print('特征矩阵大小：{}'.format(result.shape))
    print('生成特征一共用时{}秒'.format(time.time() - t0))
    return result