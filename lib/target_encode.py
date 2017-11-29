import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Adapted from https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features

class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns, noise_level = 0):
        self.columns = columns
        self.maps = {}
        self.noise_level = noise_level

    def fit(self, X, y):
        for col in self.columns:
            self.maps[col] = self.target_encode(trn_series = X[col],
                                                target = y)
        return self

    def transform(self, X):
        for col in self.columns:
            trn_series = X[col]
            averages = self.maps[col]
            ft_trn_series = pd.merge(
                trn_series.to_frame(trn_series.name),
                averages,
                left_on = trn_series.name,
                right_index = True,
                how='left')['target'].rename(trn_series.name + '_mean').fillna(averages[0])
            # pd.merge does not keep the index so restore it
            ft_trn_series.index = trn_series.index
            X[col] =  self.add_noise(ft_trn_series, self.noise_level)
        return X

    def add_noise(self, series, noise_level):
        return series * (1 + noise_level * np.random.randn(len(series)))

    def target_encode(self,
                      trn_series=None,
                      target=None,
                      min_samples_leaf=1,
                      smoothing=1):
        """
        Smoothing is computed like in the following paper by Daniele Micci-Barreca
        https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
        trn_series : training categorical feature as a pd.Series
        tst_series : test categorical feature as a pd.Series
        target : target data as a pd.Series
        min_samples_leaf (int) : minimum samples to take category average into account
        smoothing (int) : smoothing effect to balance categorical average vs prior
        """
        assert len(trn_series) == len(target)
        temp = pd.concat([trn_series, target], axis=1)
        # Compute target mean
        averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
        # Compute smoothing
        smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
        # Apply average function to all target data
        prior = target.mean()
        # The bigger the count the less full_avg is taken into account
        averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
        averages.drop(["mean", "count"], axis=1, inplace=True)
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'})
        return averages
        # Apply averages to trn and tst series

