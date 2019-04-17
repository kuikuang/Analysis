'''
This moudle is for data process
'''
import pandas as pd
import numpy as np
from collections import OrderedDict
import re
from sklearn.utils import shuffle
def HDFS(HDFSfilePath='D:/data/test_result/struct_log.csv',trainRatio=0.5):                                              #对处理HDFS日志数据生成的结构化日志进行处理，使用session窗口
    log=pd.read_csv(HDFSfilePath, engine='c',na_filter=False,memory_map=True)
    blk_dict=OrderedDict()                                                                                               #使用有序字典来存储blkId以及其对应的事件
    for line,row in log.iterrows():
        blk_list=re.findall(r'(blk_-?\d+)',row['content'])
        blk_set=set(blk_list)
        for i in blk_set:
            if not i in blk_dict:
                blk_dict[i]=[]
            blk_dict[i].append(row['event_id'])
    data_df = pd.DataFrame(list(blk_dict.items()), columns=['BlockId', 'EventSequence'])
    x_data = data_df['EventSequence'].values
    (x_train, _), (x_test, _) = splitData(x_data, trainRatio=trainRatio)
    print('Total: {} instances, train: {} instances, test: {} instances'.format(
                  x_data.shape[0], x_train.shape[0], x_test.shape[0]))
    return (x_train, None), (x_test, None)

def splitData(x_data,trainRatio=0.0):                                                                             #将数据分为测试集和训练集
    num_train = int(trainRatio * x_data.shape[0])
    x_train = x_data[0:num_train]
    x_test = x_data[num_train:]
    y_train = None
    y_test = None
    indexes = shuffle(np.arange(x_train.shape[0]))
    x_train = x_train[indexes]
    return (x_train, y_train), (x_test, y_test)
HDFS()

