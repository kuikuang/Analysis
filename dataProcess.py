'''
This moudle is for data process
'''
import pandas as pd
import numpy as np
from collections import OrderedDict
import re
def HDFS(HDFSfilePath='D:/data/test_result/struct_log.csv'):                                              #对处理HDFS日志数据生成的结构化日志进行处理，使用session窗口
    log=pd.read_csv(HDFSfilePath, engine='c',na_filter=False,memory_map=True)
    blk_dict=OrderedDict()                                                                                #使用有序字典来存储blkId以及其对应的事件
    for line,row in log.iterrows():
        blk_list=re.findall(r'(blk_-?\d+)',row['content'])
        blk_set=set(blk_list)
        for i in blk_set:
            if not i in blk_dict:
                blk_dict[i]=[]
            blk_dict[i].append(row['event_id'])
    data_df = pd.DataFrame(list(blk_dict.items()), columns=['BlockId', 'EventSequence'])

    print(blk_dict)
HDFS()

