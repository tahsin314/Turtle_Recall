import os
import numpy as np
# from config import *
import pandas as pd
from sklearn.model_selection import StratifiedKFold
data_dir = '../data/Turtle/'

def get_class_id(dirname, csvfile):
    df = pd.read_csv(os.path.join(dirname, csvfile))
    classes = df['turtle_id'].unique()
    classes = np.append(classes, 'new_turtle')
    class_id = {c: i for i, c in enumerate(classes)}
    id_class = {i: c for i, c in enumerate(classes)}
    return class_id, id_class

def get_data(dirname, csvfile, class_id, n_fold=5, random_state=42):
    
    val_idx = []
    df = pd.read_csv(os.path.join(dirname, csvfile))
    df['path'] = df['image_id'].apply(lambda x: os.path.join(dirname, 'images', f"{x}.JPG"))
    if n_fold:
        df['target'] = df['turtle_id'].apply(lambda x: class_id[x] if x in class_id.keys() else class_id['new_turtle'])

        skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=random_state)
        for i, (_, val_index) in enumerate(skf.split(df['path'], df['target'])):
            # print(val_index)
            val_idx = val_index
            df.loc[val_idx, 'fold'] = i

        df['fold'] = df['fold'].astype('int')
    try:
        df = df[df['target']!=class_id['new_turtle']]
    except: pass
    return df

def ensemble_predictions(npys, id_class, test_df):
    pred = np.load(npys[0])
    img_ids = test_df['image_id'].to_list()
    predictions = pred
    for npy in npys[1:]:
        pred = np.load(npy)
        predictions += pred
    print(np.sort(predictions, axis=1)[:, -5:][:, ::-1])
    predictions_id = np.argsort(predictions)[:, -5:][:, ::-1]
    test_probs = np.array([id_class[i] for i in np.array(predictions_id).reshape(-1)]).reshape(-1, 5)
    # print(test_probs)
    test_df = pd.DataFrame({'image_id':img_ids, 'prediction1':test_probs[:, 0], 
    'prediction2':test_probs[:, 1], 'prediction3':test_probs[:, 2],
     'prediction4':test_probs[:, 3], 'prediction5':test_probs[:, 4]})
    # predictions_k = np.sort(predictions, 1)[:, ::-1][:, :5]
    
    return test_df

class_id, id_class = get_class_id(data_dir, 'train.csv')
test_df = get_data(data_dir, 'test.csv', class_id, None, random_state=42)
ensemble =  ensemble_predictions([f'SUBMISSION_PROB_fold{i}.npy' for i in  range(5)], id_class, test_df)
ensemble.to_csv('ensemble.csv', index=False)