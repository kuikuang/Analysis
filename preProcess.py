'''
这一步是机器学习中数据预处理的步骤
'''
import pandas as pd
import numpy as np
from collections import Counter
class PreProcess(object):
    def __init__(self):
        self.idf_vec=None
        self.events=None
        self.mean_vec=None
    def fit_transform(self,X_seq,termWeighting):                                          #对训练数据提取某些我们需要的特征，并根据这些特征对train数据进行处理
        X_counts = []
        print(X_seq,'X_seq')
        for i in range(X_seq.shape[0]):
            event_counts = Counter(X_seq[i])
            X_counts.append(event_counts)
        X_df = pd.DataFrame(X_counts)
        X_df = X_df.fillna(0)
        self.events = X_df.columns
        print(self.events,'events')
        X = X_df.values
        print(X,'X')
        num_instance, num_event = X.shape                                    #instance 行，num_event 列
        if(termWeighting=='tf-idf'):
            print(num_instance,'num_instance',num_event,'num_event')
            df_vec = np.sum(X > 0, axis=0)
            print(df_vec,'df_vec')
            self.idf_vec = np.log(num_instance / (df_vec))
            print(self.idf_vec,'idf_vec')
            idf_matrix = X * np.tile(self.idf_vec, (num_instance, 1))
            X = idf_matrix
            mean_vec = X.mean(axis=0)
            self.mean_vec = mean_vec.reshape(1, num_event)
            X = X - np.tile(self.mean_vec, (num_instance, 1))
        X_new = X
        print(X_new,'X_new')
        print('Train data shape: {}-by-{}\n'.format(X_new.shape[0], X_new.shape[1]))
        return X_new

    def tranform(self,X_seq,termWeighting):                                                    #利用相同的特征，对test数据进行处理
        X_counts = []
        for i in range(X_seq.shape[0]):
            event_counts = Counter(X_seq[i])
            X_counts.append(event_counts)
        X_df = pd.DataFrame(X_counts)
        X_df = X_df.fillna(0)
        empty_events = set(self.events) - set(X_df.columns)
        print(empty_events,'empty_events')
        for event in empty_events:
            X_df[event] = [0] * len(X_df)
        X = X_df[self.events].values
        num_instance, num_event = X.shape
        print(X,'X')
        if(termWeighting=='tf-idf'):
            idf_matrix = X * np.tile(self.idf_vec, (num_instance, 1))
            X = idf_matrix
            X = X - np.tile(self.mean_vec, (num_instance, 1))
        X_new = X
        print('Test data shape: {}-by-{}\n'.format(X_new.shape[0], X_new.shape[1]))
        return X_new
