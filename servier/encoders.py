import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.base import BaseEstimator, TransformerMixin

from servier.data import MAX_LENGTH, VOCAB
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

    def __init__(self, vocab=VOCAB, max_len=MAX_LENGTH, verbose=False):
        self.verbose = verbose
        self.vocab = vocab
        self.max_len = max_len

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        vect_cols = ['v_' + str(k) for k in range(self.max_len)]
        d = {k: v for k, v in zip(self.vocab, range(1, self.max_len))}
        d['0'] = 0
        X['zfill_smiles'] = X.smiles.str.pad(width=self.max_len, side='right', fillchar='0')
        X[vect_cols] = X.zfill_smiles.apply(lambda x: pd.Series(list(x)))
        X_out = X[vect_cols].replace(d)
        if self.verbose:
            print(X_out.shape)
            print(X_out.head())
        return X_out


