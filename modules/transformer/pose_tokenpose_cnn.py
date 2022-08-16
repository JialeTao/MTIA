# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Yanjie Li (leeyegy@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
# import timm
import math
from .tokenpose_base import TokenPose_TB_base
from .hr_base import HRNET_base
from ..util import make_coordinate_grid

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

class TokenPose_B(nn.Module):

    def __init__(self, cfg, **kwargs):

        extra = cfg.MODEL.EXTRA
        self.cfg = cfg

        super(TokenPose_B, self).__init__()

        print(cfg.MODEL)
        ##################################################
        self.pre_feature = HRNET_base(cfg,**kwargs)
        self.kp = nn.Conv2d(in_channels=cfg.MODEL.BASE_CHANNEL, out_channels=cfg.MODEL.NUM_JOINTS, kernel_size=(7, 7),
                            padding=0)
        self.jacobian = nn.Conv2d(in_channels=cfg.MODEL.BASE_CHANNEL,
                                    out_channels=4, kernel_size=(7, 7), padding=0)
        ###################################################3

    def gaussian2kp(self, heatmap):
            """
            Extract the mean and from a heatmap
            """
            shape = heatmap.shape
            heatmap = heatmap.unsqueeze(-1)
            grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)
            value = (heatmap * grid).sum(dim=(2, 3)) # N * 10 * 2
            kp = {'value': value}

            return kp



    def forward(self, x):
        if self.cfg.MODEL.DATA_PREPROCESS:
            mean=torch.tensor([0.485, 0.456, 0.406]).to(x.device)
            std=torch.tensor([0.229, 0.224, 0.225]).to(x.device)
            mean=mean.view(1,3,1,1)
            std=std.view(1,3,1,1)
            x = x - mean
            x = x / std
        x = self.pre_feature(x)
        heatmap = self.kp(x)
        final_shape = heatmap.shape
        heatmap= heatmap.view(final_shape[0], final_shape[1], -1)
        heatmap= F.softmax(heatmap/ 0.1, dim=2)
        heatmap= heatmap.view(*final_shape)
        out = self.gaussian2kp(heatmap)

        jacobian_map = self.jacobian(x)
        jacobian_map = jacobian_map.reshape(final_shape[0], 1, 4, final_shape[2],final_shape[3])
        heatmap= heatmap.unsqueeze(2)
        jacobian = heatmap * jacobian_map
        jacobian = jacobian.view(final_shape[0], final_shape[1], 4, -1)
        jacobian = jacobian.sum(dim=-1)
        jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 2, 2)
        out['jacobian'] = jacobian
        return out

    def init_weights(self, pretrained=''):
        self.pre_feature.init_weights(pretrained)


def get_pose_net(cfg, is_train, **kwargs):
    cfg = convert_dict_to_attrit_dict(cfg)
    model = TokenPose_B(cfg, **kwargs)
    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model
