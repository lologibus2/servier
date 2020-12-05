import time

import pandas as pd
import numpy as np
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score,
                             confusion_matrix, classification_report)
from termcolor import colored


def perf_eval_classif(y, y_pred, verbose=True):
    """
    Compute binary calssification metrics
    :param y: (array) true labels
    :param y_pred: (array) predictions
    :param verbose: output results
    :return: dict with all metrics
    """
    d_res = classification_report(y, y_pred, output_dict=True)
    precision = d_res['1']["precision"]
    recall = d_res['1']["recall"]
    f1 = d_res['1']["f1-score"]
    roc = roc_auc_score(y, y_pred)
    conf_mat = confusion_matrix(y_true=y, y_pred=y_pred)
    res = {'f1': f1, 'ROC': roc, 'precision': precision, "recall": recall, "confusion_matrix": conf_mat}
    if verbose:
        print(pd.DataFrame(d_res).T)
    return res


def perf_eval_multiclassif(model, X, y, average='macro'):
    y_pred = model.predict(X)
    f1 = f1_score(y, y_pred, average=average)
    precision = precision_score(y, y_pred, average=average)
    recall = recall_score(y, y_pred, average=average)
    return f1, precision, recall


def show_perf(model_name, f1, precision, recall, y_true, y_pred):
    print(colored('== {} perfs'.format(model_name), 'green'))
    print(' ')
    print(colored('  F1: {} , PRECISION: {} , RECALL:{}'.format(f1, precision, recall), 'blue'))
    print(' ')
    print(colored('  Confusion matrix', 'red'))
    print(colored(confusion_matrix(y_true, y_pred), 'red'))


################
#  DECORATORS  #
################

def simple_time_tracker(method):
    """
    decorator to measure execution time of a function
    """

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts))
        else:
            print(method.__name__, round(te - ts, 2))
        return result

    return timed


################################
#   Preprocessing functions    #
################################

def describe_df(df):
    res = pd.DataFrame(
        index=["type", "nan", "unique", "values", "hist"], columns=list(df)
    )
    for col in df.columns:
        df_col = df[col]
        res[col] = [
            df_col.dtype,
            df_col.isnull().sum(),
            len(df_col.unique()),
            df_col.unique(),
            (df_col.value_counts(ascending=False, normalize=True) * 100)
                .apply(int)
                .to_json(),
        ]
    return res.T


def get_class_weights(y):
    (unique, counts) = np.unique(y, return_counts=True)
    d = {}
    for k, v in zip(unique, counts):
        d[k] = round(v / len(y), 2)
    return d
