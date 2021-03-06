from copy import deepcopy
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
getattr(ssl, '_create_unverified_context', None)):ssl._create_default_https_context = ssl._create_unverified_context
import torch
from torch import nn
from torch.nn import *
from torch.nn import functional as F
from torchvision import models
from typing import Optional
from .utils import *
from .triplet_attention import *
from .cbam import *
from .botnet import *
from losses.arcface import ArcMarginProduct
import sys
sys.path.append('../pytorch-image-models/pytorch-image-models-master')
import timm
from pprint import pprint

class Resne_t(nn.Module):

    def __init__(self, model_name='resnest50_fast_1s1x64d', num_class=3):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, in_chans=3)
        self.in_features = self.backbone.fc.in_features
        self.head = Head(self.in_features,num_class, activation='mish')
        self.out = nn.Linear(self.in_features, num_class)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.head(x)
        return x

class AttentionResne_t(nn.Module):

    def __init__(self, model_name='resnest50_fast_1s1x64d', normalize_attn=False, num_class=1):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True)
        self.in_features = self.backbone.fc.in_features
        self.head = Head(self.in_features,num_class, activation='mish')
        self.relu = Mish()
        self.maxpool = GeM()
        self.attn1 = AttentionTurtle(256, 1024, 512, 4, normalize_attn=normalize_attn)
        self.attn2 = AttentionTurtle(512, 1024, 512, 2, normalize_attn=normalize_attn)
        self.head = Head(self.in_features,num_class, activation='mish')
        
    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)

        layer1 = self.backbone.layer1(x)
        layer2 = self.backbone.layer2(layer1)
        layer3 = self.backbone.layer3(layer2)
        layer4 = self.backbone.layer4(layer3)
        a1, g1 = self.attn1(layer1, layer3)
        a2, g2 = self.attn2(layer2, layer3)
        g = self.head(layer4)
        g_hat = torch.cat((g,g1,g2), dim=1) # batch_size x C
        out = self.head(g_hat)
        return out

class TripletAttentionResne_t(nn.Module):

    def __init__(self, model_name='resnest50_fast_1s1x64d', num_class=1):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True)
        self.in_features = self.backbone.fc.in_features
        self.head = Head(self.in_features,num_class, activation='mish')
        self.relu = Mish()
        self.maxpool = GeM()
        self.ta1 = TripletAttention(True)
        self.ta2 = TripletAttention(True)
        self.ta3 = TripletAttention(True)
        self.ta4 = TripletAttention(True)
        self.head = Head(self.in_features, num_class, activation='mish')
        
    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)

        layer1 = self.backbone.layer1(x)
        layer1 = self.ta1(layer1)
        layer2 = self.backbone.layer2(layer1)
        layer2 = self.ta2(layer2)
        layer3 = self.backbone.layer3(layer2)
        layer3 = self.ta3(layer3)
        layer4 = self.backbone.layer4(layer3)
        layer4 = self.ta4(layer4)
        out = self.head(layer4)
        return out

class CBAttentionResne_t(nn.Module):

    def __init__(self, model_name='resnest50_fast_1s1x64d', num_class=1):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True)
        self.in_features = self.backbone.fc.in_features
        self.relu = Mish()
        self.maxpool = GeM()
        self.head = Head(self.in_features, num_class, activation='mish')
        self.to_CBAM(self.backbone.layer1)
        self.to_CBAM(self.backbone.layer2)
        self.to_CBAM(self.backbone.layer3)
        self.to_CBAM(self.backbone.layer4)

    def to_CBAM(self, module):
        for i in range(len(module)): 
            dim1 = module[i].conv1.out_channels
            dim2 = module[i].conv2.out_channels
            module[i].conv1.add_module('cbam1', CBAM(dim1))
            module[i].conv2.add_module('cbam2', CBAM(dim2))
        
    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)

        layer1 = self.backbone.layer1(x)
        layer2 = self.backbone.layer2(layer1)
        layer3 = self.backbone.layer3(layer2)
        layer4 = self.backbone.layer4(layer3)
        out = self.head(layer4)
        return out

class BotResne_t(nn.Module):

    def __init__(self, model_name='resnest50_fast_1s1x64d', dim=320, num_class=1):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True)
        self.dim = dim
        self.in_features = self.backbone.fc.in_features
        self.relu = Mish()
        self.maxpool = GeM()

        self.head = Head(self.in_features, num_class, activation='mish')
        # self.to_MHSA(self.backbone.layer1, 4)
        self.to_MHSA(self.backbone.layer2, 8)
        self.to_MHSA(self.backbone.layer3, 16)
        self.to_MHSA(self.backbone.layer4, 32)
    
    def to_MHSA(self, module, factor):
        dim = module[-1].conv2.out_channels 
        module[-1].conv2 = MHSA(dim, self.dim//factor, self.dim//factor)

        
    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)

        layer1 = self.backbone.layer1(x)
        layer2 = self.backbone.layer2(layer1)
        layer3 = self.backbone.layer3(layer2)
        layer4 = self.backbone.layer4(layer3)
        out = self.head(layer4)
        return out
class Mixnet(nn.Module):

    def __init__(self, model_name='mixnet_xxl', num_class=100):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, in_chans=3)
        self.in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(self.in_features, 128)
        self.output = nn.Linear(128, 2)
        self.head = Head(self.in_features,num_class, activation='mish')


    def forward(self, x, meta_data=None):
        x = self.backbone.conv_stem(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.blocks(x)
        x = self.backbone.conv_head(x)
        x = self.backbone.bn2(x)
        x = self.backbone.act2(x)
        x = self.head(x)
        return x