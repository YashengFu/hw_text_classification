import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError

class ElkanotoPuClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, hold_out_ratio=0.1):
        self.estimator = estimator
        # c is the constant proba that a example is positive, init to 1
        self.c = 1.0
        self.hold_out_ratio = hold_out_ratio
        self.estimator_fitted = False

    def __str__(self):
        return 'Estimator: {}\np(s=1|y=1,x) ~= {}\nFitted: {}'.format(
            self.estimator,
            self.c,
            self.estimator_fitted,
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
        unlabeled_hold_out, unlabeled_rest = self.split_hold_out(unlabeled)

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
        return self

    def predict_proba(self, X):
        if not self.estimator_fitted:
            raise NotFittedError(
                'The estimator must be fitted before calling predict_proba().'
            )
        probabilistic_predictions = self.estimator.predict_proba(X)
        probabilistic_predictions = probabilistic_predictions[:, 1]
        return probabilistic_predictions / self.c

    def predict(self, X, threshold=0.99999):
        if not self.estimator_fitted:
            raise NotFittedError(
                'The estimator must be fitted before calling predict(...).'
            )
        return np.array([
            1.0 if p > threshold else -1.0
            for p in self.predict_proba(X)
        ])
