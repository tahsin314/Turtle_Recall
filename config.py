import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
from configparser import ConfigParser as cfg
import string
from catalyst.data.sampler import BalanceClassSampler
import cv2
import pandas as pd
import torch 
from torch import optim
from augmentations.augmix import RandomAugMix
from augmentations.gridmask import GridMask
from augmentations.hair import Hair, AdvancedHairAugmentationAlbumentations
from augmentations.microscope import MicroscopeAlbumentations
from augmentations.color_constancy import ColorConstancy
from losses.arcface import ArcFaceLoss
from losses.focal import criterion_margin_focal_binary_cross_entropy
from model.effnet import EffNet
from utils import *
# from albumentations.augmentations.transforms import Equalize, Posterize, Downscale, Rotate 
from albumentations import (
    PadIfNeeded, HorizontalFlip, VerticalFlip, CenterCrop,  RandomSizedCrop,  
    RandomCrop, Resize, Crop, Compose, HueSaturationValue,
    Transpose, RandomRotate90, ElasticTransform, GridDistortion, 
    OpticalDistortion, RandomSizedCrop, Resize, CenterCrop,
    VerticalFlip, HorizontalFlip, OneOf, CLAHE, Normalize,
    RandomBrightnessContrast, Cutout, RandomGamma, ShiftScaleRotate ,
    GaussNoise, Blur, MotionBlur, GaussianBlur, 
)

br_config = cfg()
br_config.read('tr_config.ini')
params = br_config['params']
random_id = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(10))
data_dir = params['data_dir']
# image_path = params['image_path']
# test_image_path = params['test_image_path']
n_fold = int(params['n_fold'])
val_fold = int(params['val_fold'])
test_fold = int(params['test_fold'])
SEED = int(params['SEED'])
batch_size = int(params['batch_size'])
sz = int(params['sz'])
learning_rate = float(params['learning_rate'])
patience = int(params['patience'])
accum_step = int(params['accum_step'])
num_class = int(params['num_class'])
choice_weights = [1.0, 0.0]
cam_layer_name = params['cam_layer_name']
gpu_ids = [int(i) for i in params['gpu_ids'].split(',')]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mixed_precision = bool(int(params['mixed_precision']))
mode = params['mode']
oof = bool(int(params['oof']))
model_list = ['resnet18d', 'gluon_resnet34_v1b', 'gluon_resnet50_v1b', 'gluon_resnet101_v1b', 
'gluon_resnext101_64x4d', 'gluon_seresnext101_32x4d', 'gluon_resnext50_32x4d',
'gluon_seresnext50_32x4d', 'gluon_seresnext101_32x4d', 'resnest50d_1s4x24d', 'resnest101e', 'tf_efficientnet_b0',
'tf_efficientnet_b1', 'tf_efficientnet_b2', 'tf_efficientnet_b3', 'tf_efficientnet_b4',
'tf_efficientnet_b5', 'vit_base_patch16_384', 'lambda_resnet50', 'nfnet_l0', 'efficientnet_v2s']
model_type = params['model_type']
pretrained_model = [i for i in model_list if params['pretrained_model'] in i][0]
model_name = f'{pretrained_model}_fold_{val_fold}'
if model_type is not 'Normal':
    model_name = f'{model_type}_{pretrained_model}'
model_dir = params['model_dir']
history_dir = params['history_dir']
load_model = bool(int(params['load_model']))
distributed_backend = params['distributed_backend']
if load_model and os.path.exists(os.path.join(history_dir, f'history_{model_name}.csv')):
    history = pd.read_csv(os.path.join(history_dir, f'history_{model_name}.csv'))
else:
    history = pd.DataFrame()

imagenet_stats = ([0.485, 0.456, 0.406, 0.485, 0.456, 0.406], 
[0.229, 0.224, 0.225, 0.229, 0.224, 0.225])
ratio = 273*6/256
w, h = sz, int(ratio*sz)
n_epochs = int(params['n_epochs'])
TTA = int(params['TTA'])
balanced_sampler = bool(int(params['balanced_sampler']))
if balanced_sampler:
    sampler = BalanceClassSampler
else: sampler = None
np.random.seed(SEED)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(history_dir, exist_ok=True)
class_id, id_class = get_class_id(data_dir, 'train.csv')
df1 = pd.read_csv(os.path.join(data_dir, 'train.csv'))
df1 = df1[['image_id', 'turtle_id']]
# print(len(df1))
df2 = pd.read_csv(os.path.join(data_dir, 'extra_images.csv'))
df1 = df1[['image_id', 'turtle_id']]
df = pd.concat([df1, df2])
df.to_csv(os.path.join(data_dir, 'extra_train.csv'), index=False)
df = get_data(data_dir, 'extra_train.csv', class_id, n_fold, SEED)

df1 = get_data(data_dir, 'train.csv', class_id, n_fold, SEED)
# print(len(df1))
# df = df1.copy()
train_df = df[df['fold'] != val_fold] 
valid_df = df[df['fold'] == val_fold]
test_df = get_data(data_dir, 'test.csv', class_id, None, random_state=SEED)
# test_df = df[df['fold'] == test_fold]
# test_df2 = get_test_data(test_image_path, 5, SEED)
# print(len(df), len(train_df), len(valid_df), len(test_df))

# train_aug = Compose([
#   ShiftScaleRotate(p=0.4,rotate_limit=360, border_mode= cv2.BORDER_CONSTANT, value=[0, 0, 0], scale_limit=0.25),
#     OneOf([
#     Cutout(p=0.3, max_h_size=sz//16, max_w_size=sz//16, num_holes=10, fill_value=0),
#     # GridMask(num_grid=7, p=0.7, fill_value=0)
#     ], p=0.20),
#     RandomSizedCrop(min_max_height=(int(sz*0.8), int(sz*0.8)), height=sz, width=sz, p=0.5),
#     RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.4),
#     # RandomAugMix(severity=3, width=1, alpha=1., p=0.3), 
#     # OneOf([
#     #     # Equalize(p=0.2),
#     #     Posterize(num_bits
#     #     =4, p=0.4),
#     #     Downscale(0.40, 0.80, cv2.INTER_LINEAR, p=0.3)                  
#     #     ], p=0.2),
#     # OneOf([
#     #     GaussNoise(var_limit=0.1),
#     #     Blur(),
#     #     GaussianBlur(blur_limit=3),
#     #     # RandomGamma(p=0.7),
#     #     ], p=0.1),
#     HueSaturationValue(p=0.1),
#     HorizontalFlip(0.1),
#     VerticalFlip(0.1),
#     Resize(sz, sz, p=1, always_apply=True)
#     # Rotate(limit=360, border_mode=2, p=0.6), 
#     # ColorConstancy(p=0.3, always_apply=False),
#     # Normalize(always_apply=True)
#     ]
#       )
# val_aug = Compose([Normalize(always_apply=True)])
# val_aug = None

train_aug = Compose([
    OneOf([
        Cutout(p=0.3, max_h_size=sz//16, max_w_size=sz//16, num_holes=10, fill_value=0),
        GaussNoise(var_limit=0.2),
        ShiftScaleRotate(p=0.4,rotate_limit=45, border_mode= cv2.BORDER_REFLECT101, value=[0, 0, 0], scale_limit=0.25)

    ], p=0.30),
    # HorizontalFlip(0.4),
    # VerticalFlip(0.4),
    # Rotate(limit=360, border_mode=2, p=0.4), 
    Resize(sz, sz, p=1, always_apply=True),
    RandomSizedCrop(min_max_height=(int(sz*0.8), int(sz*0.8)), height=sz, width=sz, p=0.4),
    Resize(sz, sz, p=1, always_apply=True)
    ],    
      )
      
val_aug = Compose([Resize(sz, sz, p=1, always_apply=True)])