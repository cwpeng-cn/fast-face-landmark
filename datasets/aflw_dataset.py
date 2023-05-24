# -*- encoding: utf-8 -*-
'''
Description      : AFLW dataset
Time             :2023/05/24 13:24:02
Author           :cwpeng
email            :cw.peng@foxmail.com
'''

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from utils import constants
from utils.imutils import crop, transform


class AFLWDataset(Dataset):

    def __init__(self, use_augmentation=True, is_train=True):
        super(AFLWDataset, self).__init__()
        self.is_train = is_train
        self.use_augmentation = use_augmentation
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
        self.annots = np.load(constants.DS_ANNOT_PATH,allow_pickle=True)      
        print(">>> AFLWDataset:: Loading {} samples".format(len(self.annots)))
        
    def augm_params(self):
        """ augmentation parameters."""
        sc = 1           # scaling
        rot = 0          # rotation
        pn = np.ones(3)  # per channel pixel-noise
        
        if self.is_train and self.use_augmentation:        
            pn = np.random.uniform(1-constants.noise_factor, 1+constants.noise_factor, 3)
            rot = min(2*constants.rot_factor,max(-2*constants.rot_factor, np.random.randn()*constants.rot_factor))
            sc = min(1+constants.scale_factor,
                    max(1-constants.scale_factor, np.random.randn()*constants.scale_factor+1))
            rot=0
        return pn, rot, sc

    def rgb_processing(self, rgb_img, center, scale, rot, pn):

        """Process rgb image and do augmentation."""        
        rgb_img = crop(rgb_img, center, scale, [constants.IMG_RES, constants.IMG_RES], rot=rot)

        if rgb_img is None:
            return None
        
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
        rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
        rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))

        rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
        return rgb_img

    def j2d_processing(self, kp, center, scale, r=0):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i] = transform(kp[i]+1, center, scale, 
                                  [constants.IMG_RES, constants.IMG_RES], rot=r)
        # convert to normalized coordinates
        kp = 2.*kp/constants.IMG_RES - 1.         #-1 to 1
        kp = kp.astype('float32')
        return kp

    def __getitem__(self, index):
       
        item={}
        annot=self.annots[index]
        pn, rot, sc = self.augm_params()
        
        img = cv2.imread(annot["img_path"])[:,:,::-1].copy().astype(np.float32)     ##Note: BGR to RGB. We always use RGB
        points = annot["landmark"].copy() 
        center = np.array([annot["center_x"],annot["center_y"]])
      
        h,w=annot["h"],annot["w"]
        scale=max(h,w)/constants.IMG_RES
        visable = annot["visable"]

        img = self.rgb_processing(img, center, sc*scale, rot, pn)
        img = torch.from_numpy(img).float()

        item['img'] = self.normalize_img(img)
        item['keypoints'] = torch.from_numpy(self.j2d_processing(points, center, scale*sc, rot)).float()       #Processing to make in bbox space
        item['center'] = center.astype(np.float32)
        item['scale'] = float(sc * scale)
        item['visable'] = visable

        return item

    def __len__(self):
        return len(self.annots)
    

def torch2numpy(tensor_img):
    assert len(tensor_img.shape)==3
    np_img=tensor_img.numpy()
    np_img=np.transpose(np_img,(1,2,0))
    np_img=np_img*constants.IMG_NORM_STD+constants.IMG_NORM_MEAN
    np_img=(np_img*255).astype(np.uint8)
    return np_img

if __name__=="__main__":
    dataset=AFLWDataset()
    for i in range(100):
        item=dataset[i]
        img=item["img"]
        np_img=torch2numpy(img)
        keypoint=(item["keypoints"]+1)*constants.IMG_RES/2
        visable=item["visable"]
        for k in range(len(visable)):
            if visable[k]==1:
                # print([int(keypoint[k][0]),int(keypoint[k][1])])
                if 0<min(keypoint[k])<max(keypoint[k])<112:
                    np_img=cv2.circle(np_img,[int(keypoint[k][0]),int(keypoint[k][1])],2,[255,0,0],-1)
                else:
                    print(min(keypoint[k]),max(keypoint[k]))

        cv2.imwrite("processed/{}.jpg".format(i),np_img)
        cv2.waitKey(500)
        