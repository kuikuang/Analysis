import numpy as np
from sklearn.ensemble import IsolationForest as iForest
from sklearn.metrics import precision_score, recall_score, f1_score
class IsolationForest(iForest):
    def _init_(self,n_estimators=100,max_samples='auto',contamination=0.03,**kwargs):
        super(IsolationForest, self).__init__(n_estimators=n_estimators, max_samples=max_samples,
                                              contamination=contamination, behaviour="new", **kwargs)
    def fit(self,X):
        print('====== Model summary ======')
        super(IsolationForest, self).fit(X)

    def predict(self, X):
        y_pred = super(IsolationForest, self).predict(X)
        y_pred = np.where(y_pred > 0, 0, 1)
        return y_pred

    def evaluate(self, X, y_true):
        print('====== Evaluation summary ======')
        y_pred = self.predict(X)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))
        return precision, recall, f1





