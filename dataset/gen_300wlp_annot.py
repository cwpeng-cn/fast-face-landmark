# -*- encoding: utf-8 -*-
# -*- encoding: utf-8 -*-
'''
Description      :dataset download:
                  https://drive.google.com/file/d/0B7OEHD3T4eCkVGs0TkhUWFN6N1k/view?resourcekey=0-WT5tO4TOCbNZY6r6z6WmOA
Time             :2023/06/01 13:39:04
Author           :cwpeng
email            :cw.peng@foxmail.com
'''

import os
import glob
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat

FOLDER = "/home/cwpeng/projects/data/300W_LP/"


def load_annot(annot_path):
    c_mat = loadmat(annot_path)
    landmarks = np.array(np.transpose(c_mat['pt2d'])).astype('float').reshape(
        -1, 2)
    return landmarks


infos = []
for CLASS in os.listdir(FOLDER):
    if CLASS in ["Code", "landmarks"] or "Flip" in CLASS:
        continue
    print(CLASS)
    CLASS_PATH = os.path.join(FOLDER, CLASS)     
    for name in tqdm(glob.glob(os.path.join(CLASS_PATH, "*.jpg"))):
        if name.endswith("_0.jpg"):
            img_path = os.path.join(CLASS_PATH, name)
            annot_path = os.path.join(CLASS_PATH,
                                        name.replace("jpg", "mat"))
            landmark = load_annot(annot_path)
            visable = np.ones(68,dtype="uint8")  #TODO
            visable_landmark = landmark[visable == 1]
            for p in landmark:
                if p[0]<=0 or p[1]<=0:
                    print("hah")
            center_x = int(visable_landmark[:, 0].mean())
            center_y = int(visable_landmark[:, 1].mean())
            w = visable_landmark[:, 0].max() - visable_landmark[:, 0].min()
            h = visable_landmark[:, 1].max() - visable_landmark[:, 1].min()
            info = {
                "img_path": img_path,
                "visable": visable,
                "landmark": landmark,
                "center_x": center_x,
                "center_y": center_y,
                "w": w,
                "h": h,
                "weight": 1.0,  #TODO
                'has_visable': 0
            }
            infos.append(info)

print("300w-lp dataset samples: ",len(infos))
np.save("./dataset/300wlp_annot.npy", infos)
print("annnot saved: ./dataset/300wlp_annot.npy")