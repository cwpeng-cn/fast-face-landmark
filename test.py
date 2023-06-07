# -*- encoding: utf-8 -*-
'''
Description      : test
Time             :2023/05/24 13:21:45
Author           :cwpeng
email            :cw.peng@foxmail.com
'''

import cv2
import torch
import numpy as np
from utils import constants
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from utils.imutils import torch2numpy
from dataset.aflw_dataset import AFLWDataset
from model.mobileone import mobileone, reparameterize_model

dataset = AFLWDataset(is_train=False)
train_ds, val_ds = random_split(dataset,
                                lengths=[
                                    int(len(dataset) * 0.95),
                                    len(dataset) - int(len(dataset) * 0.95)
                                ])
val_dataloader = DataLoader(val_ds,
                            batch_size=1,
                            shuffle=False,
                            num_workers=1,
                            drop_last=False)

device = "cpu"

# net = PFLDInference().to(device)
# checkpoint = torch.load("checkpoint/pfld_epoch_142.pth.tar", map_location=device)
# net.load_state_dict(checkpoint['pfld_backbone'])

net = mobileone(num_classes=68 * 2, inference_mode=False, variant="s1")
checkpoint = torch.load("checkpoint/snapshot/checkpoint_epoch_150.pth.tar",
                        map_location=device)
net.load_state_dict(checkpoint['net'])
net = reparameterize_model(net)
net.eval()

with torch.no_grad():
    for i, item in enumerate(val_dataloader):
        img = item["img"].to(device)
        landmark_gt = item["keypoints"].to(device)
        visable = item["visable"].to(device)[0]

        pre_landmarks, pre_visables = net(img)
        pre_landmark, pre_visable = pre_landmarks[0], pre_visables[0]

        keypoint = (pre_landmark.numpy() + 1) * constants.IMG_RES / 2

        print((F.sigmoid(pre_visable) > 0.5).int())
        print(visable)
        print("-------------------------------")

        np_img = torch2numpy(img[0])

        for k in range(len(visable)):
            if visable[k] == 1:
                if 0 < min(keypoint[k]) < max(keypoint[k]) < 112:
                    cv2.circle(np_img,
                               [int(keypoint[k][0]),
                                int(keypoint[k][1])], 2, [255, 0, 0], -1)
                else:
                    # print(min(keypoint[k]),max(keypoint[k]))
                    pass

        cv2.imwrite("processed/{}.jpg".format(i), np_img)
        cv2.waitKey(500)
