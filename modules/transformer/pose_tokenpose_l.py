from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
# import timm
import math
from .tokenpose_base import TokenPose_L_base
from .hr_base import HRNET_base

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

class AttributeDict(dict):
        def __getattr__(self, attr):
            try:
                return self[attr]
            except:
                raise AttributeError
        def __setattr__(self, attr, value):
            self[attr] = value

def convert_dict_to_attrit_dict(dict_instance):
    att_dict = AttributeDict()
    for key,value in dict_instance.items():
        if isinstance(value, dict):
            value = convert_dict_to_attrit_dict(value)
        att_dict.__setattr__(key, value)
    return att_dict

class TokenPose_L(nn.Module):

    def __init__(self, cfg, **kwargs):

        extra = cfg.MODEL.EXTRA
        self.cfg = cfg

        super(TokenPose_L, self).__init__()

        print(cfg.MODEL)
        ##################################################
        self.pre_feature = HRNET_base(cfg,**kwargs)
        self.transformer = TokenPose_L_base(feature_size=[cfg.MODEL.IMAGE_SIZE[1]//4,cfg.MODEL.IMAGE_SIZE[0]//4],patch_size=[cfg.MODEL.PATCH_SIZE[1],cfg.MODEL.PATCH_SIZE[0]],
                            num_keypoints = cfg.MODEL.NUM_JOINTS,dim =cfg.MODEL.DIM,
                            channels=cfg.MODEL.BASE_CHANNEL,
                            depth=cfg.MODEL.TRANSFORMER_DEPTH,heads=cfg.MODEL.TRANSFORMER_HEADS,
                            mlp_dim = cfg.MODEL.DIM*cfg.MODEL.TRANSFORMER_MLP_RATIO,
                            apply_init=cfg.MODEL.INIT,
                            hidden_heatmap_dim=cfg.MODEL.HIDDEN_HEATMAP_DIM,
                            heatmap_dim=cfg.MODEL.HEATMAP_SIZE[1]*cfg.MODEL.HEATMAP_SIZE[0],
                            heatmap_size=[cfg.MODEL.HEATMAP_SIZE[1],cfg.MODEL.HEATMAP_SIZE[0]],
                            pos_embedding_type=cfg.MODEL.POS_EMBEDDING_TYPE,
                            estimate_jacobian=cfg.MODEL.ESTIMATE_JACOBIAN, temperature=cfg.MODEL.TEMPERATURE)
        ###################################################3

    def forward(self, x):
        if self.cfg.MODEL.DATA_PREPROCESS:
            mean=torch.tensor([0.485, 0.456, 0.406]).to(x.device)
            std=torch.tensor([0.229, 0.224, 0.225]).to(x.device)
            mean=mean.view(1,3,1,1)
            std=std.view(1,3,1,1)
            x = x - mean
            x = x / std
        x = self.pre_feature(x)
        x = self.transformer(x)
        return x

    def init_weights(self, pretrained='', cfg=None):
        self.pre_feature.init_weights(pretrained)


def get_pose_net(cfg, is_train, **kwargs):
    cfg = convert_dict_to_attrit_dict(cfg)
    model = TokenPose_L(cfg, **kwargs)
    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED,cfg)
    return model
