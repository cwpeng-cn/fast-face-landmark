
import os
import cv2
import torch
import logging
import argparse
import numpy as np
from utils import constants

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.utils.data import random_split
from torch.cuda.amp import GradScaler, autocast

from datasets.aflw_dataset import AFLWDataset
from model.pfld import PFLDInference
from loss.weight_loss import WeightLoss
from utils.avg_meter import AverageMeter

dataset=AFLWDataset()
train_ds,val_ds=random_split(dataset,lengths=[int(len(dataset)*0.95),len(dataset)-int(len(dataset)*0.95)])
val_dataloader = DataLoader(val_ds,
                        batch_size=1,
                        shuffle=False,
                        num_workers=1,
                        drop_last=False)

device="cpu"
pfld_backbone = PFLDInference().to(device)
checkpoint = torch.load("checkpoint/snapshot/checkpoint_epoch_94.pth.tar", map_location=device)
pfld_backbone.load_state_dict(checkpoint['pfld_backbone'])
pfld_backbone.eval()

def torch2numpy(tensor_img):
    assert len(tensor_img.shape)==3
    np_img=tensor_img.numpy()
    np_img=np.transpose(np_img,(1,2,0))
    np_img=np_img*constants.IMG_NORM_STD+constants.IMG_NORM_MEAN
    np_img=(np_img*255).astype(np.uint8)
    return np_img.copy()

with torch.no_grad():
    for i,item in enumerate(val_dataloader):
        img=item["img"].to(device)
        landmark_gt=item["keypoints"].to(device)
        visable=item["visable"].to(device)[0]

        pre_landmark = pfld_backbone(img)[0]
        keypoint=(pre_landmark.numpy()+1)*constants.IMG_RES/2
        
        np_img=torch2numpy(img[0])
        
        for k in range(len(visable)):
            if visable[k]==1:
                if 0<min(keypoint[k])<max(keypoint[k])<112:
                    cv2.circle(np_img,[int(keypoint[k][0]),int(keypoint[k][1])],2,[255,0,0],-1)
                else:
                    print(min(keypoint[k]),max(keypoint[k]))

        cv2.imwrite("processed/{}.jpg".format(i),np_img)
        cv2.waitKey(500)
        