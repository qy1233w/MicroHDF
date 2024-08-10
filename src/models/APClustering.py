import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.utils import resample
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import pdb
from sklearn.cluster import KMeans

class APClustering:
    def __init__(self, damping=0.5, max_iter=100, convergence_iter=30):
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter

    def cal_simi(self, X):
        dataLen = len(X)
        simi = []
        for m in X:
            temp = []
            for n in X:
                s = -np.sqrt(np.sum((m - n) ** 2))
                temp.append(s)
            simi.append(temp)
        p = np.median(simi)
        for i in range(dataLen):
            simi[i][i] = p
        return simi

    def init_R(self, dataLen):
        return np.zeros((dataLen, dataLen))

    def init_A(self, dataLen):
        return np.zeros((dataLen, dataLen))

    def iter_update_R(self, dataLen, R, A, simi):
        old_r = 0
        for i in range(dataLen):
            for k in range(dataLen):
                old_r = R[i][k]
                if i != k:
                    max1 = max([A[i][j] + R[i][j] for j in range(dataLen) if j != k])
                    R[i][k] = (1 - self.damping) * (simi[i][k] - max1) + self.damping * old_r
                else:
                    max2 = max([simi[i][j] for j in range(dataLen) if j != k])
                    R[i][k] = (1 - self.damping) * (simi[i][k] - max2) + self.damping * old_r
        return R

    def iter_update_A(self, dataLen, R, A):
        old_a = 0
        for i in range(dataLen):
            for k in range(dataLen):
                old_a = A[i][k]
                if i == k:
                    A[i][k] = (1 - self.damping) * sum([max(0, R[j][k]) for j in range(dataLen) if j != k]) + self.damping * old_a
                else:
                    A[i][k] = (1 - self.damping) * min(0, R[k][k] + sum([max(0, R[j][k]) for j in range(dataLen) if j != k and j != i])) + self.damping * old_a
        return A

    def cal_cls_center(self, dataLen, simi, R, A):
        curr_iter = 0
        curr_comp = 0
        class_cen = []
        while curr_iter < self.max_iter and curr_comp < self.convergence_iter:
            R = self.iter_update_R(dataLen, R, A, simi)
            A = self.iter_update_A(dataLen, R, A)
            new_centers = [k for k in range(dataLen) if R[k][k] + A[k][k] > 0]
            if set(new_centers) == set(class_cen):
                curr_comp += 1
            else:
                class_cen = new_centers
                curr_comp = 0
            curr_iter += 1
        return class_cen

    def fit_resample(self, X, y):
        majority_class = X[y == 1]
        minority_class = X[y == 2]
        # pdb.set_trace()
        simi = self.cal_simi(majority_class)
        dataLen = len(majority_class)
        R = self.init_R(dataLen)
        A = self.init_A(dataLen)
        class_centers = self.cal_cls_center(dataLen, simi, R, A)
        # pdb.set_trace()
        sampled_majority = []
        for center in class_centers:
            cluster_samples = [i for i in range(dataLen) if np.argmax(simi[i]) == center]
            if len(cluster_samples) == 0:
                continue
            sampled_cluster = resample(majority_class[cluster_samples], replace=True, n_samples=len(minority_class), random_state=0)
            sampled_majority.append(sampled_cluster)
        
        sampled_majority = np.vstack(sampled_majority)
        balanced_X = np.vstack([sampled_majority, minority_class])
        balanced_y = np.hstack([np.zeros(len(sampled_majority)), np.ones(len(minority_class))])
        return balanced_X, balanced_y
class APClusteringRandomForest(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, damping=0.5, max_iter=100, convergence_iter=30, random_state=None):
        self.n_estimators = n_estimators
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.random_state = random_state
        self.ap_cluster = APClustering(damping=self.damping, max_iter=self.max_iter, convergence_iter=self.convergence_iter)
        # self.ap_cluster = KMeans(n_clusters=2, random_state=9)
        self.rf_classifier = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)

    def fit(self, X, y):
        # 使用APClustering进行聚类下采样
        X_resampled, y_resampled = self.ap_cluster.fit_resample(X, y)
        # 使用随机下采样进一步平衡数据集
        majority_class = X_resampled[y_resampled == 0]
        minority_class = X_resampled[y_resampled == 1]
        if len(majority_class) > len(minority_class):
            majority_class_resampled = resample(majority_class, replace=False, n_samples=len(minority_class), random_state=self.random_state)
            X_resampled = np.vstack([majority_class_resampled, minority_class])
            y_resampled = np.hstack([np.zeros(len(majority_class_resampled)), np.ones(len(minority_class))])


        
        self.rf_classifier.fit(X_resampled, y_resampled)
        return self

    def predict(self, X):
        return self.rf_classifier.predict(X)

    def predict_proba(self, X):
        return self.rf_classifier.predict_proba(X)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
    

class APClusteringExtraTrees(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, damping=0.5, max_iter=100, convergence_iter=30, random_state=None):
        self.n_estimators = n_estimators
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.random_state = random_state
        self.ap_cluster = APClustering(damping=self.damping, max_iter=self.max_iter, convergence_iter=self.convergence_iter)
        self.et_classifier = ExtraTreesClassifier(n_estimators=self.n_estimators, random_state=self.random_state)

    def fit(self, X, y):
        # 使用APClustering进行聚类下采样
        X_resampled, y_resampled = self.ap_cluster.fit_resample(X, y)
        # 使用随机下采样进一步平衡数据集
        majority_class = X_resampled[y_resampled == 0]
        minority_class = X_resampled[y_resampled == 1]
        if len(majority_class) > len(minority_class):
            majority_class_resampled = resample(majority_class, replace=False, n_samples=len(minority_class), random_state=self.random_state)
            X_resampled = np.vstack([majority_class_resampled, minority_class])
            y_resampled = np.hstack([np.zeros(len(majority_class_resampled)), np.ones(len(minority_class))])
        self.et_classifier.fit(X_resampled, y_resampled)  # 修改为ExtraTreesClassifier
        return self

    def predict(self, X):
        return self.et_classifier.predict(X)  # 修改为ExtraTreesClassifier

    def predict_proba(self, X):
        return self.et_classifier.predict_proba(X)  # 修改为ExtraTreesClassifier

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

