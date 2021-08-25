import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
import argparse
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from sklearn.svm import SVC
#from utils.baggingPU import BaggingClassifierPU

class ElkanotoPuClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, hold_out_ratio=0.1, threshold=0.7):
        self.estimator = estimator
        # c is the constant proba that a example is positive, init to 1
        self.c = 1.0
        self.hold_out_ratio = hold_out_ratio
        self.estimator_fitted = False
        self.threshold = threshold

    def __str__(self):
        return 'Estimator: {}\np(s=1|y=1,x) ~= {}\nFitted: {}'.format(
            self.estimator,
            self.c,
            self.estimator_fitted,
            self.threshold,
        )

    def split_hold_out(self, data):
        np.random.permutation(data)
        hold_out_size = int(np.ceil(data.shape[0] * self.hold_out_ratio))
        hold_out_part = data[:hold_out_size]
        rest_part = data[hold_out_size:]

        return hold_out_part, rest_part

    def fit(self, pos, unlabeled):
        # 打乱 pos 数据集, 按比例划分 hold_out 部分和非 hold_out 部分
        pos_hold_out, pos_rest = self.split_hold_out(pos)
        pos_hold_out, valid_pos_hold_out = self.split_hold_out(pos_hold_out)
        unlabeled_hold_out, unlabeled_rest = self.split_hold_out(unlabeled)
        unlabeled_hold_out ,valid_unlabeled_hold_out = self.split_hold_out(unlabeled_hold_out)

        all_rest = np.concatenate([pos_rest, unlabeled_rest], axis=0)
        all_rest_label = np.concatenate([np.full(shape=pos_rest.shape[0], fill_value=1, dtype=np.int),
                                             np.full(shape=unlabeled_rest.shape[0], fill_value=-1, dtype=np.int)])

        self.estimator.fit(all_rest, all_rest_label)

        # c is calculated based on holdout set predictions
        hold_out_predictions = self.estimator.predict_proba(pos_hold_out)
        hold_out_predictions = hold_out_predictions[:, 1]
        c = np.mean(hold_out_predictions)
        self.c = c
        self.estimator_fitted = True

        # valid here
        pos_train_prob = self.predict_proba(pos_hold_out)
        pos_valid_prob = self.predict_proba(valid_pos_hold_out)
        unlabel_train_prob = self.predict_proba(unlabeled_hold_out)
        unlabel_valid_prob = self.predict_proba(valid_unlabeled_hold_out)

        return pos_train_prob,pos_valid_prob,unlabel_train_prob,unlabel_valid_prob

    def predict_proba(self, X):
        if not self.estimator_fitted:
            raise NotFittedError(
                'The estimator must be fitted before calling predict_proba().'
            )
        probabilistic_predictions = self.estimator.predict_proba(X)
        probabilistic_predictions = probabilistic_predictions[:, 1]
        return probabilistic_predictions / self.c

    def predict(self, X):
        if not self.estimator_fitted:
            raise NotFittedError(
                'The estimator must be fitted before calling predict(...).'
            )
        return np.array([
            1.0 if p > self.threshold else -1.0
            for p in self.predict_proba(X)
        ])


def train_pu_model():

    parser = argparse.ArgumentParser()
    parser.add_argument('--pu_data_text_save_path',default= '../data/game_data/PU_text.npy')
    parser.add_argument('--pu_data_label_save_path',default= '../data/game_data/PU_label.npy')
    parser.add_argument('--pu_model_save_path',default= '../model/baseline/pu_model.bin')
    parser.add_argument('--hold_out_ratio',default= 0.1,type=float)
    parser.add_argument('--threshold',default= 0.72,type=float)
    args = parser.parse_args()
    print(args)

    print("\nStart fitting...")
    estimator = RandomForestClassifier(
        n_estimators=100,
        criterion='gini',
        bootstrap=True,
        n_jobs=8
    )
    #estimator = BaggingClassifierPU(
    #    DecisionTreeClassifier(),
    #    n_estimators = 100,  # 1000 trees as usual
    #    max_samples = 0.05, # Balance the positives and unlabeled in each bag
    #    n_jobs = 8           # Use all cores
    #)
    pu_classifier = ElkanotoPuClassifier(estimator, hold_out_ratio=args.hold_out_ratio)

    X = np.load(args.pu_data_text_save_path)
    y = np.load(args.pu_data_label_save_path)

    n_postive = (y == 1).sum()
    n_unlabeled = (y == 0).sum()
    print("total n_positive: ", n_postive)
    print("total n_unlabel:  ", n_unlabeled)
    # 随机筛选正样本和负样本
    # positive_random_index = np.random.choice(n_postive, RANDOM_POSITIVE_NUM)
    # unlabeled_random_index = np.random.choice(n_unlabeled, RANDOM_NEGATIVE_NUM)
    y_unlabel = np.ones(n_unlabeled)

    X_positive = X[y == 1]
    print("len of X_positive: ", X_positive.shape)
    y_positive_train = np.ones(n_postive)

    X_unlabel = X[y == 0]
    print("len of X_unlabeled: ", X_unlabel.shape)
    pos_train_prob,pos_valid_prob,unlabel_train_prob,unlabel_valid_prob = pu_classifier.fit(X_positive, X_unlabel)
    #plt.hist(pos_train_prob,bins=100,histtype="step",label="Pos Train")
    plt.hist(pos_valid_prob,bins=100,histtype="step",label="Pos Valid")
    plt.hist(unlabel_valid_prob,bins=100,label="Unlabel Valid")
    #plt.hist(unlabel_train_prob,bins=100,label="Unlabel Train")
    plt.legend()
    plt.savefig('posVSunlabel.pdf')
    joblib.dump(pu_classifier, args.pu_model_save_path)
    print("Fitting done!")

if __name__ == "__main__":
    train_pu_model()
