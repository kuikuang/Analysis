'''
This moudle is for data process
'''
import pandas as pd
import numpy as np
from collections import OrderedDict
import re
from sklearn.utils import shuffle
def HDFS(HDFSfilePath,trainRatio=0.5,labelFile=None):                                              #对处理HDFS日志数据生成的结构化日志进行处理，使用session窗口
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

        print(data_df['Label'].values)
        (x_train, y_train), (x_test, y_test),(blk_id_train,blk_id_test) = splitData(data_df['BlockId'].values,data_df['EventSequence'].values,
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

    return (x_train, y_train), (x_test, y_test),(blk_id_train,blk_id_test)


def splitData(blk_id,x_data, y_data=None, trainRatio=0, split_type='uniform'):                                                                            #将数据分为测试集和训练集
    for i,j in zip(x_data,blk_id):
        i.append(j)
    if split_type == 'uniform' and y_data is not None:
        pos_idx = y_data > 0
        print(pos_idx)
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
    '''
    为了达到最终既能检测出异常又可以输出异常blk_id 的目的，我们需要将blk_id和event sequence对应起来，
    因为上一步有一个shuffle步骤，所以我们需要获得打乱顺序之后blk_id与event sequence的对应关系，所以在上一步shuffle中我们把
    blk_id放在event sequence的最后，这样我们就可以使得evente sequnce和blk_id的顺序在经过shuffle之后依然对应，接下来我们将blk_id
    提取出来。之所以要提取出来，是因为如果继续将blk_id放在event sequence最后可能会影响后面的data preprocess。因为我们的数据分为train和test集，
    所以实际上我们只需要test集的blk_id，不过为了实验效果，我们将train和test的blk_id都提取出来
    '''
    blk_id_train=[]
    for i in x_train:
        blk_id_train.append(i[-1])
        i.pop()

    blk_id_test=[]
    for j in x_test:
        blk_id_test.append(j[-1])
        j.pop()
    print('x_train',x_train)
    print('blk_id_train',blk_id_train)
    print('x_test', x_test)
    print('blk_id_test', blk_id_test)


    return (x_train, y_train), (x_test, y_test),(blk_id_train,blk_id_test)


