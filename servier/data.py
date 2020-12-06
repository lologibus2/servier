import pandas as pd
import joblib
from keras.models import load_model

LOCAL_PATH = '/Users/jeanbizot/Documents/projets/PERSO/servier/servier/data/'


def get_data(kind='single', local=True, *args, **kwargs):
    if kind == 'single':
        file_path = LOCAL_PATH + 'dataset_single.csv'
    else:
        file_path = LOCAL_PATH + 'dataset_multi.csv'
    df = pd.read_csv(file_path)
    return df


def get_X_y(df, target='P1'):
    y = df.pop(target)
    X = df
    return X, y


def model_load(path=LOCAL_PATH + 'final_models/', dl=False):
    if dl:
        # Load the pipeline first:
        pipeline = joblib.load(path + 'keras_pipeline.joblib')
        # Then, load the Keras model:
        pipeline.named_steps['clf'].model = load_model(path + 'keras_model.h5')
    else:
        pipeline = joblib.load(path + 'sklearn_pipeline.joblib')
    return pipeline
