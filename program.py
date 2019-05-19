import dataProcess
import  PCA
import preProcess
import LR
HDFSfilePath='D:/data/test_result/struct_log.csv'
labelFile='D:\data\HDFS\\anomaly_label.csv'
'''trainRatio=0.5
(x_train,y_train),(x_test,y_test),(blk_id_train,blk_id_test)=dataProcess.HDFS(HDFSfilePath,trainRatio,labelFile)

pre=preProcess.PreProcess()
termWeighting='tf-idf'
x_train=pre.fit_transform(x_train,termWeighting)
x_test=pre.tranform(x_test,termWeighting)

model=PCA.PCA()
model.fit(x_train)
print('Train validation:')
precision,recall,f1=model.evaluate(x_train,y_train,blk_id_train)
print('Test validation:')
precision, recall, f1 = model.evaluate(x_test, y_test,blk_id_test)
'''

trainRatio=0.5
(x_train1,y_train1),(x_test1,y_test1),(blk_id_train,blk_id_test)=dataProcess.HDFS(HDFSfilePath,trainRatio,labelFile)
pre=preProcess.PreProcess()
termWeighting='tf-idf'
x_train1=pre.fit_transform(x_train1,termWeighting)
x_test1=pre.tranform(x_test1,termWeighting)
model_LR=LR.LR()
model_LR.fit(x_train1,y_train1)
print('Train validation:')
precision_LR_train,recall_iForest_train,f1_iForest_train=model_LR.evaluate(x_train1,y_train1)
print('Test validation:')
precision_LR_test, recall_iForest_test, f1_iForest_test = model_LR.evaluate(x_test1, y_test1)

