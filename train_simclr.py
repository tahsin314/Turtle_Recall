import os
import glob
from functools import partial
import gc
from matplotlib.pyplot import axis
from augmentations.augmix import normalize
from config import *
import shutil
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from tqdm import tqdm as T
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize

import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data.distributed import DistributedSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (ModelCheckpoint, 
LearningRateMonitor, StochasticWeightAveraging,) 
from pytorch_lightning.loggers import WandbLogger
from TurtleDataset import TurtleDataset, TurtleDataModule
from catalyst.data.sampler import BalanceClassSampler
from losses.ohem import ohem_loss
from losses.mix import MixupLoss, mixup, MixupLoss
from losses.regression_loss import *
from losses.focal import (criterion_margin_focal_binary_cross_entropy,
FocalLoss, FocalCosineLoss)
from utils import *
from model.simclr import *# from optimizers.over9000 import AdamW, Ralamb
from TurtleModule import ImageEmbeddingTurtle, LightningTurtle
import wandb

seed_everything(SEED)
os.system("rm -rf *.png *.csv")
if mode == 'lr_finder':
  wandb.init(mode="disabled")
  wandb_logger = WandbLogger(project="Turtle", config=params, settings=wandb.Settings(start_method='fork'))
else:
  wandb_logger = WandbLogger(project="Turtle", config=params, settings=wandb.Settings(start_method='fork'))
  wandb.init(project="Turtle", config=params, settings=wandb.Settings(start_method='fork'))
  wandb.run.name= model_name

optimizer = optim.RMSprop
# base_criterion = nn.BCEWithLogitsLoss(reduction='sum')
base_criterion = criterion_margin_focal_binary_cross_entropy
# mixup_criterion_ = partial(mixup_criterion, criterion=base_criterion, rate=1.0)
mixup_criterion = MixupLoss(base_criterion, 1.0)
# ohem_criterion = partial(ohem_loss, rate=1.0, base_crit=base_criterion)
criterions = [base_criterion, mixup_criterion]
# criterion = criterion_margin_focal_binary_cross_entropy

lr_reduce_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau

for f in range(n_fold):
    print(f"FOLD #{f}")
    train_df = df[(df['fold']!=f)]
    valid_df = df[df['fold']==f]
    base = ImageEmbedding(pretrained_model, 128)

    wandb.watch(base)
    plist = [ 
        {'params': base.parameters(),  'lr': learning_rate},
        # {'params': base.head.parameters(),  'lr': learning_rate}
    ]
    train_ds = TurtleDataset(list(df1.path.values)+list(test_df.path.values), None, dim=sz, num_class=num_class,
    embedding=True, transforms=train_aug)

    valid_ds = TurtleDataset(list(df1.path.values)+list(test_df.path.values), None, dim=sz, num_class=num_class, 
    embedding=True, transforms=val_aug)

    test_ds = TurtleDataset(test_df.path.values, None, dim=sz,num_class=num_class, 
    embedding=True, transforms=val_aug)
    data_module = TurtleDataModule(train_ds, valid_ds, test_ds,  sampler= sampler, 
    batch_size=batch_size)
    cyclic_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer(plist, 
    lr=learning_rate), 
    5*len(data_module.train_dataloader()), 1, learning_rate/5, -1)
    # data_module = TurtleDataModule(base, optimizer, plist, batch_size, lr_reduce_scheduler,
    # learning_rate, cyclic_scheduler)
    data_module = TurtleDataModule(train_ds, valid_ds, test_ds,  sampler= sampler,
    batch_size=batch_size)
    # cyclic_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer(plist, 
    # lr=learning_rate), learning_rate/5, 4*learning_rate/5, step_size_up=3*len(data_module.train_dataloader()), 
    # step_size_down=1*len(data_module.train_dataloader()), mode='exp_range', gamma=1.0, scale_fn=None, scale_mode='cycle', 
    # cycle_momentum=False, base_momentum=0.8, max_momentum=0.8, last_epoch=-1, verbose=False)
    # cyclic_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer(plist, 
    # lr=learning_rate), [learning_rate/5, learning_rate], epochs=n_epochs, steps_per_epoch=len(data_module.train_dataloader()), 
    # pct_start=0.7, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.80, max_momentum=0.80, 
    # div_factor=5.0, final_div_factor=20.0, three_phase=True, last_epoch=-1, verbose=False)

    if mode == 'lr_finder': cyclic_scheduler = None
    # model = LightningTurtle(model=base, choice_weights=choice_weights, loss_fns=criterions,
    # optim= optimizer, plist=plist, batch_size=batch_size, 
    # lr_scheduler= lr_reduce_scheduler, num_class=num_class, fold=f, cyclic_scheduler=cyclic_scheduler, 
    # learning_rate = learning_rate, random_id=random_id)
    model = ImageEmbeddingTurtle(base, optimizer, plist, batch_size, lr_reduce_scheduler,
    learning_rate, cyclic_scheduler)
    checkpoint_callback1 = ModelCheckpoint(
        monitor=f'train_loss',
        dirpath='model_dir',
        filename=f"{model_name}_loss_fold_{f}",
        save_top_k=1,
        mode='min',
    )

    # checkpoint_callback2 = ModelCheckpoint(
    #     monitor=f'val_mAP@5_fold_{f}',
    #     dirpath='model_dir',
    #     filename=f"{model_name}_mAP@5_fold_{f}",
    #     save_top_k=1,
    #     mode='max',
    # )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    # swa_callback = StochasticWeightAveraging()

    trainer = pl.Trainer(max_epochs=n_epochs, precision=16, 
                      # auto_lr_find=True,  # Usually the auto is pretty bad. You should instead plot and pick manually.
                      gradient_clip_val=100,
                      num_sanity_val_steps=10,
                      profiler="simple",
                      weights_summary='top',
                      accumulate_grad_batches = accum_step,
                      logger=[wandb_logger], 
                      checkpoint_callback=True,
                      gpus=gpu_ids, num_processes=4*len(gpu_ids),
                      # stochastic_weight_avg=True,
                      # auto_scale_batch_size='power',
                      benchmark=True,
                      distributed_backend=distributed_backend,
                      # plugins='deepspeed', # Not working 
                      # early_stop_callback=False,
                      progress_bar_refresh_rate=1, 
                      callbacks=[checkpoint_callback1,
                      lr_monitor])

    if mode == 'lr_finder':
      model.choice_weights = [1.0, 0.0]
      trainer.train_dataloader = data_module.train_dataloader
      # Run learning rate finder
      lr_finder = trainer.tuner.lr_find(model, data_module.train_dataloader(), min_lr=1e-6, 
      max_lr=500, num_training=2000)
      # Plot with
      fig = lr_finder.plot(suggest=True, show=True)
      fig.savefig('lr_finder.png')
      fig.show()
    # Pick point based on plot, or get suggestion
      new_lr = lr_finder.suggestion()
      print(f"Suggested LR: {new_lr}")
      exit()

    wandb.log(params)
    trainer.fit(model, datamodule=data_module)
    print(gc.collect())
    try:
      print(f"FOLD: {f} \
        Best Model path: {checkpoint_callback1.best_model_path} Best Score: {checkpoint_callback1.best_model_score:.4f}")
    except:
      pass
    chk_path = checkpoint_callback1.best_model_path
    # chk_path = '/home/UFAD/m.tahsinmostafiz/Playground/Turtle_Recall/model_dir/Normal_tf_efficientnet_b2_loss_fold_0-v4.ckpt'
    train_ds = TurtleDataset(df1.path.values, None, dim=sz, num_class=num_class,
    embedding=True, transforms=val_aug)

    valid_ds = TurtleDataset(test_df.path.values, None, dim=sz, num_class=num_class, 
    embedding=True, transforms=val_aug)

    test_ds = TurtleDataset(test_df.path.values, None, dim=sz,num_class=num_class, 
    embedding=True, transforms=val_aug)
    data_module = TurtleDataModule(train_ds, valid_ds, test_ds,  sampler= None, 
    batch_size=batch_size)
    cyclic_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer(plist, 
    lr=learning_rate), 
    5*len(data_module.train_dataloader()), 1, learning_rate/5, -1)
    # data_module = TurtleDataModule(base, optimizer, plist, batch_size, lr_reduce_scheduler,
    # learning_rate, cyclic_scheduler)
    data_module = TurtleDataModule(train_ds, valid_ds, test_ds,  sampler= sampler,
    batch_size=batch_size)
    model2 = ImageEmbeddingTurtle.load_from_checkpoint(chk_path, model=base, optim=optimizer, plist=plist,
    batch_size= batch_size, lr_scheduler= lr_reduce_scheduler,
    learning_rate=learning_rate, cyclic_scheduler=cyclic_scheduler)

    # trainer.test(model=model2, test_dataloaders=data_module.val_dataloader())
    trainer.test(model=model2, test_dataloaders=data_module.test_dataloader())
    TEST_Z = np.load('embedding.npy', allow_pickle=True)
    TEST_IMG_ID = np.load('img_id.npy', allow_pickle=True)
    
    # print(TEST_Z.shape)
    trainer.test(model=model2, test_dataloaders=data_module.train_dataloader())
    TRAIN_Z = np.load('embedding.npy', allow_pickle=True)
    TRAIN_IMG_ID = np.load('img_id.npy', allow_pickle=True)
    similarity_matrix = cosine_similarity(TRAIN_Z, TEST_Z)
    # plt.figure(figsize=(20,20))
    plt.imshow(similarity_matrix.astype(np.float), cmap='hot', interpolation='nearest')
    plt.savefig('similarity_matrix.png')

    preds = []
    sim_scores = []
    for t in T(range(len(test_df))):
      sims = similarity_matrix[:,t]
      turts = []
      scores = []
      top_idxs = np.array(sims).argsort()[-60:][::-1]
      top_labels = [df1.turtle_id.values[idx] for idx in top_idxs]
      top_scores = [sims[idx] for idx in top_idxs]
      for i, l in enumerate(top_labels):
        if len(turts)>5: # We only need the top 5
          break
        if l in turts:
          pass
        else:
          turts.append(l)
          scores.append(top_scores[i])
  
      # Occasionally all top matches are for a couple of turts
      # SO here we pad out the preds/scores to include 'new_turtle' in these cases
      turts = (turts+['new_turtle']*5)[:5]
      scores = (scores+[0.5]*5)[:5]

      preds.append(turts)
      sim_scores.append(scores)

    for i in range(5):
      test_df[f'prediction{i+1}'] = [p[i] for p in preds]
      test_df[f'score{i+1}'] = [s[i] for s in sim_scores]

    # test_df.sample(5)
    ss = pd.merge(test_df['image_id'], 
              test_df[['image_id']+[f'prediction{i+1}' for i in range(5)]+[f'score{i+1}' for i in range(5)]], 
              how='left', left_on='image_id', right_on='image_id')
    
    thresh=0.8
    ss.loc[ss.score1.map(lambda x: float(x))<thresh, 'prediction5'] = ss[ss.score1.map(lambda x: float(x))<thresh]['prediction4']
    ss.loc[ss.score1.map(lambda x: float(x))<thresh, 'prediction4'] = ss[ss.score1.map(lambda x: float(x))<thresh]['prediction3']
    ss.loc[ss.score1.map(lambda x: float(x))<thresh, 'prediction3'] = ss[ss.score1.map(lambda x: float(x))<thresh]['prediction2']
    ss.loc[ss.score1.map(lambda x: float(x))<thresh, 'prediction2'] = ss[ss.score1.map(lambda x: float(x))<thresh]['prediction1']
    ss.loc[ss.score1.map(lambda x: float(x))<thresh, 'prediction1'] = 'new_turtle'
    ss[ss.columns[:6]].to_csv('contrastive_learning_jax_demo.csv', index=False)
    # CAM Generation
    model2.eval()
    # plot_heatmap(model2, test_df, val_aug, cam_layer_name=cam_layer_name, num_class=num_class, sz=sz)
    # cam = cv2.imread('./heatmap.png', cv2.IMREAD_COLOR)
    # cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    # wandb.log({"CAM": [wandb.Image(cam, caption="Class Activation Mapping")]})
    # model2.model.backbone.fc = nn.Identity()
    # # print(model2.model.backbone)
    # lrp = LRP_Captum(model2.model.backbone, torch.randn(3, sz, sz))
    # print(lrp)
    if not oof:
      break

# oof_df = pd.concat([pd.read_csv(fname) for fname in glob.glob('oof_*.csv')])
# oof_df.to_csv(f'oof.csv', index=False)