from random import choice, choices

from cv2 import log
from config import *
from losses.mix import *
from utils import *
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import label_binarize
import wandb

class LightningTurtle(pl.LightningModule):
  def __init__(self, model, choice_weights, loss_fns, optim, plist, 
  batch_size, lr_scheduler, random_id, fold=0, distributed_backend='dp',
  cyclic_scheduler=None, num_class=1, patience=3, factor=0.5,
   learning_rate=1e-3):
      super().__init__()
      self.model = model
      self.num_class = num_class
      self.loss_fns = loss_fns
      self.optim = optim
      self.plist = plist 
      self.lr_scheduler = lr_scheduler
      self.cyclic_scheduler = cyclic_scheduler
      self.random_id = random_id
      self.fold = fold
      self.distributed_backend = distributed_backend
      self.patience = patience
      self.factor = factor
      self.learning_rate = learning_rate
      self.batch_size = batch_size
      self.choice_weights = choice_weights
      self.criterion = self.loss_fns[0]
      self.train_loss  = 0
      self.test_imgs = []
      self.test_probs = []
      self.TEST_PROBS_ALL = []
      self.epoch_end_output = [] # Ugly hack for gathering results from multiple GPUs
      self.softmax = nn.Softmax(dim=1)
  
  def forward(self, x):
      out = self.model(x)
      out = out.type_as(x)
      return out

  def configure_optimizers(self):
        optimizer = self.optim(self.plist, self.learning_rate)
        lr_sc = self.lr_scheduler(optimizer, mode='max', factor=0.5, 
        patience=patience, verbose=True, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=1e-7, eps=1e-08)
        return ({
       'optimizer': optimizer,
       'lr_scheduler': lr_sc,
       'monitor': f'val_mAP@5_fold_{self.fold}',
       'cyclic_scheduler': self.cyclic_scheduler}
        )
 
  def loss_func(self, logits, labels):
      return self.criterion(logits, labels)
  
  def step(self, batch):
    try:
      _, x, y = batch
      x, y = x.float(), y.float()
      if len(self.loss_fns) >1:
        if self.criterion == self.loss_fns[1]:
          x, y1, y2, lam = mixup(x, y)
          y = [y1, y2, lam]
    except:
      img_id, x = batch
    
    logits = torch.squeeze(self.forward(x))
    if len(batch) > 2:
      loss = self.loss_func(logits, y)
      return loss, logits, y  
    else:
      return img_id, logits
  
  def training_step(self, train_batch, batch_idx):
    # if self.current_epoch < 4:
    #   loss, _, _ = self.step(train_batch, [1.0, 0.0])
    # else:
    if self.current_epoch >=30:
      self.cyclic_scheduler = None
    self.criterion = choices(self.loss_fns, weights=choice_weights)[0]
    loss, _, _ = self.step(train_batch)
    self.train_loss  += loss.detach()
    self.log(f'train_loss_fold_{self.fold}', self.train_loss/batch_idx, prog_bar=True)
    if self.cyclic_scheduler is not None:
      self.cyclic_scheduler.step()
    return loss

  def validation_step(self, val_batch, batch_idx):
      self.criterion = self.loss_fns[0]
      self.train_loss  = 0
      loss, logits, y = self.step(val_batch)
      self.log(f'val_loss_fold_{self.fold}', loss, on_epoch=True, sync_dist=True) 
      val_log = {'val_loss':loss, 'probs':logits, 'gt':y}
      self.epoch_end_output.append({k:v.cpu() for k,v in val_log.items()})
      return val_log

  def test_step(self, test_batch, batch_idx):
      self.criterion = self.loss_fns[0]
      self.train_loss  = 0
      img_id, logits = self.step(test_batch)
      predictions = logits.detach().cpu().numpy()
      self.TEST_PROBS_ALL.extend(self.softmax(logits).detach().cpu().numpy())
      # self.TEST_PROBS_ALL.extend(logits.sigmoid().detach().cpu().numpy())
      # predictions = logits.sigmoid().detach().cpu().numpy()
      predictions_id = np.argsort(predictions)[:, -5:][:, ::-1]
      predictions_k = np.sort(predictions, 1)[:, ::-1][:, :5]
      # for i, p in enumerate(predictions_k):
      #   if p[-1] <= 1/2:
      #     predictions_id[i, -1] = 100
      self.test_imgs.extend([i.split('/')[-1].split('.')[0].replace("images\\", "") for i in img_id])
      self.test_probs.extend(predictions_id)
      

  def label_processor(self, probs, gt):
    # pr = probs.sigmoid().detach().cpu().numpy()
    pr = self.softmax(probs).detach().cpu().numpy()
    # pr = probs.detach().cpu().numpy()
    la = gt.detach().cpu().numpy()
    return pr, la

  def distributed_output(self, outputs):
    if torch.distributed.is_initialized():
      print('TORCH DP')
      torch.distributed.barrier()
      gather = [None] * torch.distributed.get_world_size()
      torch.distributed.all_gather_object(gather, outputs)
      outputs = [x for xs in gather for x in xs]
    return outputs

  def epoch_end(self, mode, outputs):
    if self.distributed_backend:
      outputs = self.epoch_end_output
    if not mode == 'test':
      avg_loss = torch.Tensor([out[f'{mode}_loss'].mean() for out in outputs]).mean()
      probs = torch.cat([torch.tensor(out['probs']) for out in outputs], dim=0)
      gt = torch.cat([torch.tensor(out['gt']) for out in outputs], dim=0)
      pr, la = self.label_processor(torch.squeeze(probs), torch.squeeze(gt))
      pr = np.nan_to_num(pr, 0.5)
      # labels = [i for i in range(self.num_class)]
      # pr = np.argmax(pr, axis=1)
      la = np.argmax(la, axis=1)
      map_k = torch.tensor(mapk(la, pr, k=5))
      # f_score = torch.tensor(f1_score(la, pr, labels=None, average='micro', sample_weight=None))
      print(f'Epoch: {self.current_epoch} Loss : {avg_loss.numpy():.2f}, mAP@5: {map_k:.4f}')
      logs = {f'{mode}_loss': avg_loss, f'{mode}_mAP@5': map_k}
      self.log(f'{mode}_loss_fold_{self.fold}', avg_loss)
      self.log( f'{mode}_mAP@5_fold_{self.fold}', map_k)
      self.epoch_end_output = []
      # plot_confusion_matrix(pr, la, labels)
      # hist = cv2.imread('./conf.png', cv2.IMREAD_COLOR)
      # hist = cv2.cvtColor(hist, cv2.COLOR_BGR2RGB)
      # # wandb.log({"histogram": [wandb.Image(hist, caption="Histogram")]})
      # plot_heatmap(self.model, valid_df, val_aug, sz)
      # cam = cv2.imread('./heatmap.png', cv2.IMREAD_COLOR)
      # cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
      # wandb.log({"CAM": [wandb.Image(cam, caption="Class Activation Mapping")]})
      return pr, la, {f'avg_{mode}_loss': avg_loss, 'log': logs}
    else:
      return {'log': logs}

  def validation_epoch_end(self, outputs):
    _, _, log_dict = self.epoch_end('val', outputs)
    self.epoch_end_output = []
    return log_dict

  def test_epoch_end(self, outputs):
    # log_dict = self.epoch_end('test', outputs)
    # print(self.epoch_end_output)
    # self.epoch_end_output = []
    # print(np.array(self.test_imgs).shape, np.array(self.test_probs).shape)
    test_probs = np.array([id_class[i] for i in np.array(self.test_probs).reshape(-1)]).reshape(-1, 5)
    # print(test_probs)
    test_df = pd.DataFrame({'image_id':self.test_imgs, 'prediction1':test_probs[:, 0], 
    'prediction2':test_probs[:, 1], 'prediction3':test_probs[:, 2],
     'prediction4':test_probs[:, 3], 'prediction5':test_probs[:, 4]})
    test_prob_df = pd.DataFrame({'image_id':self.test_imgs, 'probabilities':self.TEST_PROBS_ALL})
    np.save(f'SUBMISSION_PROB_fold{self.fold}.npy', self.TEST_PROBS_ALL)
    test_df.to_csv(f'SUBMISSION_fold{self.fold}.csv', index=False)
    