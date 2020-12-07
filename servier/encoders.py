import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from servier.feature_extractor import df_to_features


class MorganFingerprintEncoder(BaseEstimator, TransformerMixin):
    """
    Custom Sklearn Pipeline encoder to encode SMILE mol
    """

    def __init__(self, radius=3, size=2048, verbose=False):
        self.verbose = verbose
        self.radius = radius
        self.size = size

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X_out = df_to_features(X, radius=self.radius, size=self.size, only_fingerprint=True)
        if self.verbose:
            print(X_out.head())
        return X_out



class TokenizerEncoder(BaseEstimator, TransformerMixin):
    """
    Custom Sklearn Pipeline encoder to encode SMILE mol
    """

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.vocab = []

    def fit(self, X, y=None):
        self.vocab = [{l for word in X.smiles.values for l in word}]
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X_out = df_to_features(X, radius=self.radius, size=self.size, only_fingerprint=True)
        if self.verbose:
            print(X_out.head())
        return X_out
