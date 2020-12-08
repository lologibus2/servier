import glob

import pandas as pd
import os
import joblib
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

LOCAL_PATH = os.getenv('DATA_PATH', '/Users/jeanbizot/Documents/projets/PERSO/servier/servier/data/')
TARGET = 'P1'
VOCAB = ['N', '3', '\\', '4', 'o', 'C', 'H', '1', '5', 'c', '=', 's', '-', 'F', 'r', ']', '[', ')', 'B', '2', 'n', '(',
         '6', '#', 'S', 'l', 'O', '/', '+']
EMBEDDING_VECTOR_LENGTH, MAX_LENGTH, VOCAB_SIZE = 32, 74, 29 + 1


def get_data(kind='single', test=False, all=True, path=LOCAL_PATH, *args, **kwargs):
    if kind == 'single':
        if test:
            file_path = path + 'dataset_single_test.csv'
        else:
            file_path = path + 'dataset_single_train.csv'
        if all:
            file_path = path + 'dataset_single.csv'
    else:
        file_path = path + 'dataset_multi.csv'
    df = pd.read_csv(file_path)
    return df


def split_train_test():
    file_path = LOCAL_PATH + 'dataset_single.csv'
    df = pd.read_csv(file_path)
    X_train, X_test = train_test_split(df, stratify=df.P1, test_size=0.10)
    X_train.to_csv(LOCAL_PATH + 'dataset_single_train.csv', index=False)
    X_test.to_csv(LOCAL_PATH + 'dataset_single_test.csv', index=False)


def get_X_y(df, target='P1'):
    y = df.pop(target)
    X = df
    return X, y


def resample_data(df, up=False, target=TARGET):
    # Separate majority and minority classes
    majority_class = df[target].mode().values[0]
    df_majority = df[df[target] == majority_class]
    df_minority = df[df[target] == 1 - majority_class]

    n_minority = df_minority.shape[0]
    n_majority = df_majority.shape[0]
    seed = 123

    if up:
        # Upsample minority class
        df_minority_upsampled = resample(df_minority,
                                         replace=True,  # sample with replacement
                                         n_samples=n_majority,  # to match majority class
                                         random_state=seed)  # reproducible results

        # Combine majority class with upsampled minority class
        df_new = pd.concat([df_majority, df_minority_upsampled])
    else:
        # Downsample majority class
        df_majority_downsampled = resample(df_majority,
                                           replace=False,  # sample without replacement
                                           n_samples=n_minority,  # to match minority class
                                           random_state=seed)  # reproducible results
        # Combine minority class with downsampled majority class
        df_new = pd.concat([df_majority_downsampled, df_minority])
    return df_new


def load_final_model(path=LOCAL_PATH + 'final_models/', model=1, dl=True):
    if dl:
        # Load the pipeline first:
        suff = '_model'+str(model)
        pipeline = joblib.load(path+'keras_pipeline'+suff+'.joblib')
        # Then, load the Keras model:
        pipeline.named_steps['clf'].model = load_model(path + 'keras_model'+suff+'.h5')
    else:
        pipeline = joblib.load(path + 'sklearn_pipeline.joblib')
    return pipeline


def load_model_from_path(path_pipeline, path_model):
    pipeline = joblib.load(path_pipeline)
    pipeline.named_steps['clf'].model = load_model(path_model)
    return pipeline


def get_pipeline_model_from_path(path=LOCAL_PATH + 'models/', archi='cnn'):
    l = glob.glob(path+'/*')
    l_pipelines = [m for m in l if 'keras_pipeline_'+archi in m]
    l_model = [m for m in l if 'keras_model_'+archi in m]
    l_model.sort()
    l_pipelines.sort()
    res = []
    for p, m in zip(l_pipelines, l_model):
        suff_p, suff_m = p.split('-')[-1], m.split('-')[-1]
        end_p = suff_p.replace('.joblib', '')
        end_m = suff_m.replace('.h5', '')
        if end_m == end_p or suff_p.replace('_pipeline', '_model') == suff_m.replace('.h5', '.joblib'):
            res.append((p, m))
    return res
