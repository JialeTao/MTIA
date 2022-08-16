from argparse import ArgumentParser
from time import gmtime, strftime
import os, sys

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from einops import rearrange, repeat
import imageio
import cv2
import numpy as np

from tqdm import tqdm
import yaml
from shutil import copy

from frames_dataset import FramesDataset
from visualizer import get_local
get_local.activate()
import modules
# from modules.transformer.pose_tokenpose_b  import get_pose_net
from modules import transformer
from modules.generator import OcclusionAwareGenerator
from modules.discriminator import MultiScaleDiscriminator
from modules.bg_motion_predictor import BGMotionPredictor

from torchsummaryX import summary
# from torchsummary import summary
# from torchstat import stat

import time
print("Start : %s" % time.ctime())
#time.sleep(7200)
print("End : %s" % time.ctime())

if __name__ == "__main__":


    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--log_dir", default='./log', help="path to log into")
    parser.add_argument("--oss_dir", default='jiale.tjl/MoTrans/', help="path to upload to oss")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--oss_checkpoint", default=None, help="path to oss checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f)

    device = torch.device('cuda:{}'.format(opt.device_ids[0]))
    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
        log_dir = os.path.join(log_dir, 'vis_att')
    else:
        log_dir = opt.log_dir + '_' + os.path.basename(opt.config).split('.')[0]

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    kp_detector = eval('transformer.'+config['MODEL']['NAME']+'.get_pose_net')(config, is_train=False)
    kp_detector.to(device)
    # kp_detector = torch.nn.DataParallel(kp_detector)
    kp_detector.eval()

    checkpoint = torch.load(opt.checkpoint, map_location='cuda:{}'.format(0))
    kp_detector.load_state_dict(checkpoint['kp_detector'])

    dataset = FramesDataset(is_train=False, **config['dataset_params'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    depth = config['MODEL']['TRANSFORMER_DEPTH']

    jacob_token = kp_detector.transformer.jacobian_token
    num_keypoints = kp_detector.transformer.num_keypoints
    num_patches = kp_detector.transformer.num_patches
    patch_size  = kp_detector.transformer.patch_size

    # jacob_token = kp_detector.features.jacobian_token
    # num_keypoints = kp_detector.features.num_keypoints
    # num_patches = kp_detector.features.num_patches
    # patch_size  = kp_detector.features.patch_size

    for it, x in tqdm(enumerate(dataloader)):
        if it > 0:
            break
        elif it < 0:
            continue
        # if it < 1:
            # continue
        # elif it > 1:
        #     break
        with torch.no_grad():
            # if torch.cuda.is_available():
                # # 1*3*t*h*w
            #     x['video'] = x['video'].cuda()
            for frame_idx in range(x['video'].shape[2]):
                if frame_idx != 30:
                    continue
                    # break
                # if frame_idx > 0:
                #     break
                print(x['name'])
                driving = x['video'][:, :, frame_idx].cuda()
                kp_driving = kp_detector(driving)
                cache = get_local.cache
                print(cache.keys())
                dots = list(cache.values())
                print(len(dots))
                dots = dots[0]
                print(len(dots))
                # print(type(dots))

                kp_self_dot = []
                jacob_self_dot = []
                kp_jacob_cross_dot = []
                kp_fea_cross_att = []
                jacob_fea_cross_att = []
                for dot in dots:
                    # print(len(dot))
                    # bhnn
                    # attn = dot.softmax(dim=-1)[0]
                    attn = dot[0]
                    # print(attn[0,8,512])
                    # print(attn.shape)
                    # if 1:
                        # # print(np.argmax(attn[:,0:num_keypoints,:],axis=2))
                        # att_token = torch.from_numpy(attn[:,0:num_keypoints,:]).softmax(-1)[:,0:num_keypoints,0:num_keypoints].numpy()
                        # att_feat = torch.from_numpy(attn[:,0:num_keypoints,:]).softmax(-1)[:,0:num_keypoints,num_keypoints:].numpy()
                        # print(att_token.sum(-1).mean(-1).mean(), att_feat.sum(-1).mean(-1).mean())
                        # # print(att_token.sum(-1), att_feat.sum(-1))
                        # # print(torch.from_numpy(attn[:,0:num_keypoints,:]).softmax(-1)[:,num_keypoints//2:num_keypoints,num_keypoints//2:num_keypoints].numpy())
                    #     continue
                    if jacob_token:
                        ## h*10*10
                        kp_self_dot.append(attn[:,0:num_keypoints//2,0:num_keypoints//2])
                        jacob_self_dot.append(attn[:,num_keypoints//2:num_keypoints,num_keypoints//2:num_keypoints])
                        kp_jacob_cross_dot.append(attn[:,0:num_keypoints//2,num_keypoints//2:num_keypoints])
                        ## h*10*hw
                        kp_fea_cross_att.append(attn[:,0:num_keypoints//2, num_keypoints:])
                        jacob_fea_cross_att.append(attn[:,num_keypoints//2:num_keypoints, num_keypoints:])
                    else:
                        kp_fea_cross_att.append(attn[:,0:num_keypoints, num_keypoints:])
                        jacob_fea_cross_att.append(attn[:,0:num_keypoints, num_keypoints:])
                # if 1:
                #     break
                ## h*num_layer*10*10
                # kp_self_dot = torch.from_numpy(np.array(kp_self_dot).transpose(1,0,2,3))
                # kp_self_att = kp_self_dot.softmax(-1)
                # jacob_self_dot = torch.from_numpy(np.array(jacob_self_dot).transpose(1,0,2,3))
                # jacob_self_att = jacob_self_dot.softmax(-1)
                # kp_jacob_cross_dot = torch.from_numpy(np.array(kp_jacob_cross_dot).transpose(1,0,2,3))
                # kp_jacob_cross_att = kp_jacob_cross_dot.softmax(-1)

                # kp_self_dot = kp_self_dot.sum(0,keepdim=True)
                # kp_self_att = kp_self_dot.sum(0,keepdim=True)
                # jacob_self_dot = jacob_self_dot.sum(0,keepdim=True)
                # jacob_self_att = jacob_self_att.sum(0,keepdim=True)
                # kp_jacob_cross_dot = kp_jacob_cross_dot.sum(0,keepdim=True)
                # kp_jacob_cross_att = kp_jacob_cross_att.sum(0,keepdim=True)

                # kp_self_dot = kp_self_dot - torch.min(kp_self_dot, dim=-1, keepdim=True)[0]
                # kp_self_dot = kp_self_dot / torch.max(kp_self_dot, dim=-1, keepdim=True)[0]
                # # kp_self_att = kp_self_att - torch.min(kp_self_att, dim=-1, keepdim=True)[0]
                # # kp_self_att = kp_self_att / torch.max(kp_self_att, dim=-1, keepdim=True)[0]
                # jacob_self_dot = jacob_self_dot - torch.min(jacob_self_dot, dim=-1, keepdim=True)[0]
                # jacob_self_dot = jacob_self_dot / torch.max(jacob_self_dot, dim=-1, keepdim=True)[0]
                # # jacob_self_att = jacob_self_att - torch.min(jacob_self_att, dim=-1, keepdim=True)[0]
                # # jaocb_self_att = jacob_self_att / torch.max(jacob_self_att, dim=-1, keepdim=True)[0]
                # kp_jacob_cross_dot = kp_jacob_cross_dot - torch.min(kp_jacob_cross_dot, dim=-1, keepdim=True)[0]
                # kp_jacob_cross_dot = kp_jacob_cross_dot / torch.max(kp_jacob_cross_dot, dim=-1, keepdim=True)[0]
                # # kp_jacob_cross_att = kp_jacob_cross_att - torch.min(kp_jacob_cross_att, dim=-1, keepdim=True)[0]
                # # kp_jacob_cross_att = kp_jacob_cross_att / torch.max(kp_jacob_cross_att, dim=-1, keepdim=True)[0]

                # kp_self_dot = repeat(kp_self_dot.unsqueeze(-1), 'he l k n ()->he l k n p', p=10*10)
                # kp_self_dot = rearrange(kp_self_dot, 'he l k n (p1 p2)->(he k p1) (l n p2)', p1=10)
                # kp_self_att = repeat(kp_self_att.unsqueeze(-1), 'he l k n ()->he l k n p', p=10*10)
                # kp_self_att = rearrange(kp_self_att, 'he l k n (p1 p2)->(he k p1) (l n p2)', p1=10)
                # jacob_self_dot = repeat(jacob_self_dot.unsqueeze(-1), 'he l k n ()->he l k n p', p=10*10)
                # jacob_self_dot = rearrange(jacob_self_dot, 'he l k n (p1 p2)->(he k p1) (l n p2)', p1=10)
                # jacob_self_att = repeat(jacob_self_att.unsqueeze(-1), 'he l k n ()->he l k n p', p=10*10)
                # jacob_self_att = rearrange(jacob_self_att, 'he l k n (p1 p2)->(he k p1) (l n p2)', p1=10)
                # kp_jacob_cross_dot = repeat(kp_jacob_cross_dot.unsqueeze(-1), 'he l k n ()->he l k n p', p=10*10)
                # kp_jacob_cross_dot = rearrange(kp_jacob_cross_dot, 'he l k n (p1 p2)->(he k p1) (l n p2)', p1=10)
                # kp_jacob_cross_att = repeat(kp_jacob_cross_att.unsqueeze(-1), 'he l k n ()->he l k n p', p=10*10)
                # kp_jacob_cross_att = rearrange(kp_jacob_cross_att, 'he l k n (p1 p2)->(he k p1) (l n p2)', p1=10)

                # kp_self_dot[0:-1:100,:] = 0
                # kp_self_dot[:,0:-1:100] = 0
                # kp_self_att[0:-1:100,:] = 0
                # kp_self_att[:,0:-1:100] = 0
                # jacob_self_dot[0:-1:100,:] = 0
                # jacob_self_dot[:,0:-1:100] = 0
                # jacob_self_att[0:-1:100,:] = 0
                # jacob_self_att[:,0:-1:100] = 0
                # kp_jacob_cross_dot[0:-1:100,0:-1:100] = 0
                # kp_jacob_cross_dot[0:-1:100,:] = 0
                # kp_jacob_cross_att[0:-1:100,:] = 0
                # kp_jacob_cross_att[:,0:-1:100] = 0
                # kp_self_dot = cv2.applyColorMap((kp_self_dot.numpy()*255).astype(np.uint8), cv2.COLORMAP_JET)
                # kp_self_att = cv2.applyColorMap((kp_self_att.numpy()*255).astype(np.uint8), cv2.COLORMAP_JET)
                # jacob_self_dot = cv2.applyColorMap((jacob_self_dot.numpy()*255).astype(np.uint8), cv2.COLORMAP_JET)
                # jacob_self_att = cv2.applyColorMap((jacob_self_att.numpy()*255).astype(np.uint8), cv2.COLORMAP_JET)
                # kp_jacob_cross_dot = cv2.applyColorMap((kp_jacob_cross_dot.numpy()*255).astype(np.uint8), cv2.COLORMAP_JET)
                # kp_jacob_cross_att = cv2.applyColorMap((kp_jacob_cross_att.numpy()*255).astype(np.uint8), cv2.COLORMAP_JET)
                # cv2.imwrite(os.path.join(log_dir, 'kp_self_dot.png'), kp_self_dot)
                # cv2.imwrite(os.path.join(log_dir, 'kp_self_att.png'), kp_self_att)
                # cv2.imwrite(os.path.join(log_dir, 'jacob_self_dot.png'), jacob_self_dot)
                # cv2.imwrite(os.path.join(log_dir, 'jacob_self_att.png'), jacob_self_att)
                # cv2.imwrite(os.path.join(log_dir, 'kp_jacob_cross_dot.png'), kp_jacob_cross_dot)
                # cv2.imwrite(os.path.join(log_dir, 'kp_jacob_cross_att.png'), kp_jacob_cross_att)
                # if 1:
                #     break
                ## h*num_layer*10*10

                ## h*num_layer*10*h*w
                kp_fea_cross_dot = torch.from_numpy(np.array(kp_fea_cross_att).transpose(1,0,2,3))
                kp_fea_cross_att = kp_fea_cross_dot.softmax(-1)

                ## sum all head attentions
                kp_fea_cross_dot = kp_fea_cross_dot.sum(0, keepdim=True)
                kp_fea_cross_att = kp_fea_cross_att.sum(0, keepdim=True)
                ## sum all head attentions

                # print(torch.min(kp_fea_cross_att, dim=-1, keepdim=True)[0].shape)
                kp_fea_cross_dot = kp_fea_cross_dot - torch.min(kp_fea_cross_dot, dim=-1, keepdim=True)[0]
                kp_fea_cross_dot = kp_fea_cross_dot / torch.max(kp_fea_cross_dot, dim=-1, keepdim=True)[0]
                # kp_fea_cross_dot = repeat(kp_fea_cross_dot.unsqueeze(-1), 'he l k n ()->he l k n p', p=patch_size[0]*patch_size[1])
                # print(kp_fea_cross_dot.shape)

                kp_fea_cross_att = kp_fea_cross_att - torch.min(kp_fea_cross_att, dim=-1, keepdim=True)[0]
                kp_fea_cross_att = kp_fea_cross_att / torch.max(kp_fea_cross_att, dim=-1, keepdim=True)[0]
                # kp_fea_cross_att = repeat(kp_fea_cross_att.unsqueeze(-1), 'he l k n ()->he l k n p', p=patch_size[0]*patch_size[1])
                # print(kp_fea_cross_dot.shape)

                # print(kp_fea_cross_att[2,:,8,492,0])
                # kp_fea_cross_att_test = rearrange(kp_fea_cross_att.permute(2,0,1,3,4), 'k he l (h w) p->k he l h w p',  h=64//patch_size[0],w=64//patch_size[1])
                # print(kp_fea_cross_att_test[8,0,:,15,12,0])
                # kp_fea_cross_dot = rearrange(kp_fea_cross_dot, 'he l k (h w) (p1 p2)->(he l) k (h p1) (w p2)',  h=64//patch_size[0],w=64//patch_size[1],p1=patch_size[0],p2=patch_size[1])
                print(kp_fea_cross_dot.shape)
                kp_fea_cross_dot = rearrange(kp_fea_cross_dot, 'he l k (h w)->(he l) k h w',  h=64//patch_size[0],w=64//patch_size[1])
                print(kp_fea_cross_dot.shape)
                kp_fea_cross_dot = F.upsample(kp_fea_cross_dot, scale_factor=4*patch_size[0], mode='bilinear')
                kp_fea_cross_dot = rearrange(kp_fea_cross_dot, '(he l) k h w->k (he h) (l w)', he=1,l=depth)
                print(kp_fea_cross_dot.shape)

                # kp_fea_cross_att = rearrange(kp_fea_cross_att, 'he l k (h w) (p1 p2)->(he l) k  (h p1) (w p2)',  h=64//patch_size[0],w=64//patch_size[1],p1=patch_size[0],p2=patch_size[1])
                kp_fea_cross_att = rearrange(kp_fea_cross_att, 'he l k (h w)->(he l) k h w',  h=64//patch_size[0],w=64//patch_size[1])
                kp_fea_cross_att = F.upsample(kp_fea_cross_att, scale_factor=4*patch_size[0], mode='bilinear')
                kp_fea_cross_att = rearrange(kp_fea_cross_att, '(he l) k h w->k (he h) (l w)', he=1,l=depth)
                # kp_fea_cross_att = rearrange(kp_fea_cross_att, 'he l k (h w) (p1 p2)->k (he h p1) (l w p2)',  h=64//patch_size[0],w=64//patch_size[1],p1=patch_size[0],p2=patch_size[1])
                # print(kp_fea_cross_att[8,30,24:-1:64])
                # break
                # kp_fea_cross_att = repeat(kp_fea_cross_att.unsqueeze(-1), 'k h w ()->k h w c', c=3)

                ## repeat patch first then softmax
                # jacob_fea_cross_att = repeat(torch.from_numpy(np.array(jacob_fea_cross_att).transpose(1,0,2,3)).unsqueeze(-1), 'h l k n ()->h l k n p', p=16*16)
                # jacob_fea_cross_att = rearrange(jacob_fea_cross_att, 'he l k (h w) (p1 p2)->he l k (h p1) (w p2)', h=16, p1=16)
                # jacob_fea_cross_att = rearrange(jacob_fea_cross_att, 'he l k h w->he l k (h w)')
                # jacob_fea_cross_att = rearrange(jacob_fea_cross_att, 'he l k (h w)->k (he h) (l w)', h=256)

                # no repeat
                # jacob_fea_cross_att = torch.from_numpy(np.array(jacob_fea_cross_att).transpose(1,0,2,3))
                # jacob_fea_cross_att = rearrange(jacob_fea_cross_att, 'he l k (h w)->(he l) k h w', h=16)
                # jacob_fea_cross_att = F.upsample(jacob_fea_cross_att, scale_factor=16)
                # # jacob_fea_cross_att = rearrange(jacob_fea_cross_att, 'he l k h w->he l k (h w)')
                # jacob_fea_cross_att = rearrange(jacob_fea_cross_att, '(he l) k h w->k (he h) (l w)', he=8)
                ## repeat patch first then softmax

                ## softmax all patch first then repeat every patch
                jacob_fea_cross_dot = torch.from_numpy(np.array(jacob_fea_cross_att).transpose(1,0,2,3))
                jacob_fea_cross_att = jacob_fea_cross_dot.softmax(-1)

                ## sum all head attentions
                jacob_fea_cross_dot = jacob_fea_cross_dot.sum(0, keepdim=True)
                jacob_fea_cross_att = jacob_fea_cross_att.sum(0, keepdim=True)
                ## sum all head attentions

                jacob_fea_cross_dot = jacob_fea_cross_dot - torch.min(jacob_fea_cross_dot, dim=-1, keepdim=True)[0]
                jacob_fea_cross_dot = jacob_fea_cross_dot / torch.max(jacob_fea_cross_dot, dim=-1, keepdim=True)[0]
                # jacob_fea_cross_dot = repeat(jacob_fea_cross_dot.unsqueeze(-1), 'h l k n ()->h l k n p', p=patch_size[0]*patch_size[1])

                jacob_fea_cross_att = jacob_fea_cross_att - torch.min(jacob_fea_cross_att, dim=-1, keepdim=True)[0]
                jacob_fea_cross_att = jacob_fea_cross_att / torch.max(jacob_fea_cross_att, dim=-1, keepdim=True)[0]
                # jacob_fea_cross_att = repeat(jacob_fea_cross_att.unsqueeze(-1), 'h l k n ()->h l k n p', p=patch_size[0]*patch_size[1])

                # jacob_fea_cross_dot = rearrange(jacob_fea_cross_dot, 'he l k (h w) (p1 p2)->(he l) k  (h p1) (w p2)',  h=64//patch_size[0],w=64//patch_size[1],p1=patch_size[0],p2=patch_size[1])
                jacob_fea_cross_dot = rearrange(jacob_fea_cross_dot, 'he l k (h w)->(he l) k h w',  h=64//patch_size[0],w=64//patch_size[1])
                jacob_fea_cross_dot = F.upsample(jacob_fea_cross_dot, scale_factor=4*patch_size[0], mode='bilinear')
                jacob_fea_cross_dot = rearrange(jacob_fea_cross_dot, '(he l) k h w->k (he h) (l w)', he=1,l=depth)

                # jacob_fea_cross_att = rearrange(jacob_fea_cross_att, 'he l k (h w) (p1 p2)->(he l) k  (h p1) (w p2)',  h=64//patch_size[0],w=64//patch_size[1],p1=patch_size[0],p2=patch_size[1])
                jacob_fea_cross_att = rearrange(jacob_fea_cross_att, 'he l k (h w)->(he l) k h w',  h=64//patch_size[0],w=64//patch_size[1])
                jacob_fea_cross_att = F.upsample(jacob_fea_cross_att, scale_factor=4*patch_size[0], mode='bilinear')
                jacob_fea_cross_att = rearrange(jacob_fea_cross_att, '(he l) k h w->k (he h) (l w)', he=1,l=depth)
                # jacob_fea_cross_att = rearrange(jacob_fea_cross_att, 'he l k (h w) (p1 p2)->k (he h p1) (l w p2)', h=64//patch_size[0],w=64//patch_size[1],p1=patch_size[0],p2=patch_size[1])
                ## softmax all patch first then repeat every patch

                # # jacob_fea_cross_att = repeat(jacob_fea_cross_att.unsqueeze(-1), 'k h w ()->k h w c', c=3)
                # # kp_fea_cross_att = np.array(kp_fea_cross_att).transpose(1,0,2,3).reshape(8,12,10,16,16)
                # # jacob_fea_cross_att = np.array(jacob_fea_cross_att).transpose(1,0,2,3).reshape(8,12,10,16,16)
                kp_fea_dot = kp_fea_cross_dot.numpy()
                jacob_fea_dot = jacob_fea_cross_dot.numpy()
                kp_fea_att = kp_fea_cross_att.numpy()
                jacob_fea_att = jacob_fea_cross_att.numpy()

                print(driving.shape)
                driving_raw = driving[0].permute(1,2,0).data.cpu().numpy()
                driving_raw = np.uint8(np.array(driving_raw)[:,:,::-1])
                imageio.imsave(os.path.join(log_dir, 'driving.png'), driving_raw)

                driving = repeat(driving, '() c h w->n c h w', n=1*depth)
                driving = rearrange(driving, '(he l) c h w->(he h) (l w) c', he=1, l=depth)
                driving = driving.data.cpu().numpy().astype(np.uint8)

                for i in range(num_keypoints // 2):
                # for i in range(num_keypoints):
                    kp_dot = kp_fea_dot[i,:,:]
                    jacob_dot = jacob_fea_dot[i,:,:]
                    kp_att = kp_fea_att[i,:,:]
                    jacob_att = jacob_fea_att[i,:,:]

                    kp_dot[0:-1:256,:] = 0
                    kp_dot[:,0:-1:256] = 0
                    jacob_dot[0:-1:256,:] = 0
                    jacob_dot[:,0:-1:256] = 0
                    kp_att[0:-1:256,:] = 0
                    kp_att[:,0:-1:256] = 0
                    jacob_att[0:-1:256,:] = 0
                    jacob_att[:,0:-1:256] = 0
                    # if i == 8:
                    #     print(kp_att[30,24:-1:64])
                    kp_dot = cv2.applyColorMap((kp_dot*255).astype(np.uint8), cv2.COLORMAP_JET)
                    jacob_dot = cv2.applyColorMap((jacob_dot*255).astype(np.uint8), cv2.COLORMAP_JET)
                    kp_att = cv2.applyColorMap((kp_att*255).astype(np.uint8), cv2.COLORMAP_JET)
                    jacob_att = cv2.applyColorMap((jacob_att*255).astype(np.uint8), cv2.COLORMAP_JET)

                    # jacob_dot = np.concatenate((driving_raw, jacob_dot),axis=1)
                    # jacob_att = np.concatenate((driving_raw, jacob_att),axis=1)



                    # print(driving.shape)
                    # print(kp_dot.shape)
                    # img_kp_dot = kp_dot * 0.3 + driving
                    # img_kp_att = kp_att * 0.3 + driving
                    # img_jacob_dot = jacob_dot * 0.3 + driving
                    # img_jacob_att = jacob_att * 0.3 + driving
                    # if i == 8:
                    #     print(kp_att[30,24:-1:64])
                    if not os.path.exists(os.path.join(log_dir, str(i))):
                        os.makedirs(os.path.join(log_dir, str(i)))
                    cv2.imwrite(os.path.join(log_dir, str(i), 'kp_cross_dot.png'), kp_dot)
                    cv2.imwrite(os.path.join(log_dir, str(i), 'kp_cross_att.png'), kp_att)
                    cv2.imwrite(os.path.join(log_dir, str(i), 'jacob_cross_dot.png'), jacob_dot)
                    cv2.imwrite(os.path.join(log_dir, str(i), 'jacob_cross_att.png'), jacob_att)

                    # cv2.imwrite(os.path.join(log_dir, str(i), 'img_kp_dot.png'), img_kp_dot)
                    # cv2.imwrite(os.path.join(log_dir, str(i), 'img_kp_att.png'), img_kp_att)
                    # cv2.imwrite(os.path.join(log_dir, str(i), 'img_jacob_dot.png'), img_jacob_dot)
                    # cv2.imwrite(os.path.join(log_dir, str(i), 'img_jacob_att.png'), img_jacob_att)
    # # oss_uploader()




























