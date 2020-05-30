import numpy as np
import pandas as pd


class csv2ffm:
    def __init__(self):
        self.field_index_ = dict()
        self.feature_index_ = dict()
        self.category_feat = []
        self.continus_feat = []
        self.min_count = 0
        self.label = None

    def fit(self, df, label, params):
        self.label = label
        self.category_feat = params.get('category_feat',[])
        self.continus_feat = params.get('continus_feat',[])
        self.min_count = params.get('min_count',0)
        self.field_index_ = {col: i for i, col in enumerate(df.columns)}
        last_idx = 0 if len(self.feature_index_) == 0 else sum([len(v) for k,v in self.feature_index_])
        for col in df.columns:
            if col in self.category_feat:
                value_counts_ = df[col].fillna('-1').value_counts()
                value_counts_ = (value_counts_[value_counts_>self.min_count].argsort()+last_idx).to_dict()
                self.feature_index_[col] = value_counts_
                last_idx += len(value_counts_)
        return self

    def fit_transform(self, df, label=None, params={}):
        self.fit(df, label, params)
        return self.transform(df)

    def transform(self, df):
        if self.label in df.columns:
            result = df[[self.label]].copy()
        else:
            result = pd.DataFrame({'label': [0] * df.shape[1]},index=df.index)
        for col in df.columns:
            if col in self.category_feat:
                result[col] = str(self.field_index_[col]) + ':' + df[col].fillna('-1').map(self.feature_index_[col]).map(str).fillna('-1') + ':' + '1'
            elif col in self.continus_feat:
                result[col] = str(self.field_index_[col]) + ':' + str(self.field_index_[col]) + ':' + df[col].astype('float32').map(str)
        return result

    def dump_ffm(self,data,data_path):
        print('存储ffm数据...')
        with open(data_path,'w') as ffm:
            for row in data.values:
                ffm.write(' '.join([str(i) for i in row]))
                ffm.write('\n')


