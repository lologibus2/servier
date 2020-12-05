import pandas as pd

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
