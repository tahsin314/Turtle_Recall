import os
# from config import *
import pandas as pd
from sklearn.model_selection import StratifiedKFold


dirname = '../../data/Turtle/'

def get_data(dirname, csvfile, n_fold=5, random_state=42):
    
    paths = []
    classname = []
    train_idx = []
    val_idx = []
    df = pd.read_csv(os.path.join(dirname, csvfile))
    df['path'] = df['image_id'].apply(lambda x: os.path.join(dirname, 'train', x))
    if n_fold:
        skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=random_state)
        for i, (train_index, val_index) in enumerate(skf.split(df['path'], df['turtle_id'])):
            train_idx = train_index
            val_idx = val_index
            df.loc[val_idx, 'fold'] = i

        df['fold'] = df['fold'].astype('int')

    return df

print(get_data(dirname, 'train.csv').head())