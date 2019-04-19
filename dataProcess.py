'''
This moudle is for data process
'''
import pandas as pd
import numpy as np
from collections import OrderedDict
import re
from sklearn.utils import shuffle
def HDFS(HDFSfilePath='D:/data/test_result/struct_log.csv',trainRatio=0.5,labelFile=None):                                              #对处理HDFS日志数据生成的结构化日志进行处理，使用session窗口
    log=pd.read_csv(HDFSfilePath, engine='c',na_filter=False,memory_map=True)
    blk_dict=OrderedDict()                                                                                                              #使用有序字典来存储blkId以及其对应的事件
    for line,row in log.iterrows():
        blk_list=re.findall(r'(blk_-?\d+)',row['content'])
        blk_set=set(blk_list)
        for i in blk_set:
            if not i in blk_dict:
                blk_dict[i]=[]
            blk_dict[i].append(row['event_id'])
    data_df = pd.DataFrame(list(blk_dict.items()), columns=['BlockId', 'EventSequence'])
    if labelFile:
        label_data = pd.read_csv(labelFile, engine='c', na_filter=False, memory_map=True)
        label_data = label_data.set_index('BlockId')
        label_dict = label_data['Label'].to_dict()
        data_df['Label'] = data_df['BlockId'].apply(lambda x: 1 if label_dict[x] == 'Anomaly' else 0)


        (x_train, y_train), (x_test, y_test) = splitData(data_df['EventSequence'].values,
                                                           data_df['Label'].values, trainRatio, split_type='uniform')
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    num_total = num_train + num_test
    num_train_pos = sum(y_train)
    num_test_pos = sum(y_test)
    num_pos = num_train_pos + num_test_pos

    print('Total: {} instances, {} anomaly, {} normal' \
              .format(num_total, num_pos, num_total - num_pos))
    print('Train: {} instances, {} anomaly, {} normal' \
              .format(num_train, num_train_pos, num_train - num_train_pos))
    print('Test: {} instances, {} anomaly, {} normal\n' \
              .format(num_test, num_test_pos, num_test - num_test_pos))

    return (x_train, y_train), (x_test, y_test)


def splitData(x_data, y_data=None, trainRatio=0, split_type='uniform'):                                                                             #将数据分为测试集和训练集
    if split_type == 'uniform' and y_data is not None:
        pos_idx = y_data > 0
        x_pos = x_data[pos_idx]
        y_pos = y_data[pos_idx]
        x_neg = x_data[~pos_idx]
        y_neg = y_data[~pos_idx]
        train_pos = int(trainRatio * x_pos.shape[0])
        train_neg = int(trainRatio * x_neg.shape[0])
        x_train = np.hstack([x_pos[0:train_pos], x_neg[0:train_neg]])
        y_train = np.hstack([y_pos[0:train_pos], y_neg[0:train_neg]])
        x_test = np.hstack([x_pos[train_pos:], x_neg[train_neg:]])
        y_test = np.hstack([y_pos[train_pos:], y_neg[train_neg:]])
    else:
        print('wrong')
    indexes = shuffle(np.arange(x_train.shape[0]))
    x_train = x_train[indexes]
    if y_train is not None:
        y_train = y_train[indexes]
    return (x_train, y_train), (x_test, y_test)
HDFS(labelFile='D:\data\HDFS\\anomaly_label.csv')

