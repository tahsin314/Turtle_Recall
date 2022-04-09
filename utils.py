import os
from config import *
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold
import cv2
import torch
from torch.nn import functional as F
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm as T
from gradcam.gradcam import GradCAM, GradCAMpp
# from captum.attr import LRP 


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

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
        # print(len(df))
        # df = df[df['target']!=class_id['new_turtle']]
        # print(len(df))
        # print(df.loc[1560, 'fold'])
        # df = df.reindex(columns=['path', 'target'])
        # df = df[['path', ]]
        skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=random_state)
        for i, (_, val_index) in enumerate(skf.split(df['path'], df['target'])):
            # print(val_index)
            val_idx = val_index
            df.loc[val_idx, 'fold'] = i

        df['fold'] = df['fold'].astype('int')
    try:
        df = df[df['target']!=class_id['new_turtle']]
        # df_old = df[df['target']!=class_id['new_turtle']]
        # df_new = df[df['target']==class_id['new_turtle']].sample(n=25, random_state=random_state)
        # df = pd.concat([df_old, df_new])
    except: pass
    return df

def ensemble_predictions(npys, id_class, test_df):
    pred = np.load(npys[0])
    img_ids = df['image_id'].to_list()
    predictions = pred
    for npy in npys[1:]:
        pred = np.load(npy)
        predictions += pred
    predictions_id = np.argsort(predictions)[:, -5:][:, ::-1]
    test_probs = np.array([id_class[i] for i in np.array(predictions_id).reshape(-1)]).reshape(-1, 5)
    # print(test_probs)
    test_df = pd.DataFrame({'image_id':img_ids, 'prediction1':test_probs[:, 0], 
    'prediction2':test_probs[:, 1], 'prediction3':test_probs[:, 2],
     'prediction4':test_probs[:, 3], 'prediction5':test_probs[:, 4]})
    # predictions_k = np.sort(predictions, 1)[:, ::-1][:, :5]
    
    return test_df

def apk(actual, predicted, k=5):
  """Computes the average precision at k.

  Args:
    actual: The turtle ID to be predicted.
    predicted : A list of predicted turtle IDs (order does matter).
    k : The maximum number of predicted elements.

  Returns:
    The average precision at k.
  """
  predicted = np.argsort(predicted)[::-1][:k]
  if len(predicted) > k:
    predicted = predicted[:k]

  score = 0.0
  num_hits = 0.0

  for i, p in enumerate(predicted):
    if p == actual and p not in predicted[:i]:
      num_hits += 1.0
      score += num_hits / (i + 1.0)

  return score


def mapk(actual, predicted, k=5):
  """ Computes the mean average precision at k.

    The turtle ID at actual[i] will be used to score predicted[i][:k] so order
    matters throughout!

    actual: A list of the true turtle IDs to score against.
    predicted: A list of lists of predicted turtle IDs.
    k: The size of the window to score within.

    Returns:
      The mean average precision at k.
  """
  return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def get_test_data(dirname, n_fold=5, random_state=42):
    
    paths = []
    classname = []
    train_idx = []
    val_idx = []
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=random_state)
    for root, dirs, files in os.walk(dirname, topdown=False):
        for name in files:
            path = os.path.join(root, name)
            paths.append(path)
            classname.append(int(path.split('/test/')[-1].split('/')[0]))
    df = pd.DataFrame(list(zip(paths, classname)), columns=['id', 'target'])
    for i, (train_index, val_index) in enumerate(skf.split(paths, classname)):
        train_idx = train_index
        val_idx = val_index
        df.loc[val_idx, 'fold'] = i

    df['fold'] = df['fold'].astype('int')

    return df


def plot_confusion_matrix(predictions, actual_labels, labels):
    cm = confusion_matrix(predictions, actual_labels, labels)
    # Normalise
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(12,12))
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('conf.png')

def visualize_cam(mask, img, alpha=0.8, beta=0.15):
    
    """
    Courtesy: https://github.com/vickyliin/gradcam_plus_plus-pytorch/blob/master/gradcam/utils.py
    Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]
    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    heatmap = (255 * mask.squeeze()).type(torch.uint8).cpu().numpy()
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b]) * alpha

    result = heatmap+img.cpu()*beta
    result = result.div(result.max()).squeeze()

    return heatmap, result


def grad_cam_gen(model, img, cam_layer_name='layer4', device = 'cuda'):     
    configs = [dict(model_type='resnet', arch=model, layer_name=cam_layer_name)]
    for config in configs:
        config['arch'].to(device).eval()

    cams = [
    [cls.from_config(**config) for cls in (GradCAM, GradCAMpp)]
        for config in configs]

    for _, gradcam_pp in cams:
        mask_pp, _ = gradcam_pp(img)
        heatmap_pp, result_pp = visualize_cam(mask_pp, img, 0.985, 0.015)
        result_pp = result_pp.cpu().numpy()
        #convert image back to Height,Width,Channels
        heatmap_pp = np.transpose(heatmap_pp, (1,2,0))
        result_pp = np.transpose(result_pp, (1,2,0))
        return result_pp

def LRP_Captum(model, img, device = 'cuda'):
    model.to(device)
    img = img.to(device)
    img = img.unsqueeze(0)
    img = img.float()
    img = Variable(img).to(device)
    lrp = LRP(model)
    attribution = lrp.attribute(input, target=5)
    return attribution

def plot_heatmap(model, valid_df, val_aug, device='cuda', cam_layer_name='layer4', num_class=9, sz=384):
    
    fig = plt.figure(figsize=(70, 56))
    valid_df['id'] = valid_df['id'].map(lambda x: x)
    print('Plotting heatmaps...')
    class_ids = sorted(valid_df['target'].unique())
    for class_id in T(range(len(class_ids))):
        for i, (idx, row) in enumerate(valid_df.loc[valid_df['target'] == 
        class_ids[class_id]].sample(5, random_state=42).iterrows()):
            ax = fig.add_subplot(num_class, 5, class_id * 5 + i + 1, xticks=[], yticks=[])
            path=f"{row['id']}"
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (sz, sz))
            aug = val_aug(image=image)
            image = aug['image'].transpose(2, 0, 1)
            image = torch.FloatTensor(image)
            prediction = torch.sigmoid(model(torch.unsqueeze(image.to(device), dim=0)))
            prediction = prediction.data.cpu().numpy()
            image = grad_cam_gen(model.model.backbone, torch.unsqueeze(image, dim=0).cuda(), cam_layer_name=cam_layer_name)
            image = (image-np.min(image))/(np.max(image)-np.min(image))
            plt.imshow(image)
            # ax.set_title(f"Label: {row['target']} Prediction: {int(np.argmax(prediction))} Confidence: {np.max(prediction) :.3f}", fontsize=40)
    plt.savefig('heatmap.png')
