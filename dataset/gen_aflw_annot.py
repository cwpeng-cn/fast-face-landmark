# -*- encoding: utf-8 -*-
'''
Description      : used to generate data,
                   dataset download: https://github.com/abhi1kumar/MERL-RAV_dataset
Time             :2023/05/24 13:24:27
Author           :cwpeng
email            :cw.peng@foxmail.com
'''

import os
import glob
import numpy as np
from tqdm import tqdm

FOLDER = "/home/cwpeng/projects/data/merl_rav_organized/"


def load_annot(annot_path):
    with open(annot_path, 'r') as f:
        points, visables = [], []
        lines = f.readlines()
        for i in range(3, 71):
            data = lines[i].split("\n")[0].split(" ")
            point = [float(d) for d in data]
            visable = 1
            if point[0] == -1 and point[1] == -1:
                visable = 0
                point = [0, 0]
            point = [abs(point[0]), abs(point[1])]
            points.append(point)
            visables.append(visable)
        points = np.array(points)
        visables = np.array(visables)
    return points, visables

weights={"frontal":1,"right":1.2,"left":1.2,"lefthalf":1.5,"righthalf":1.5}
infos = []
for CLASS in os.listdir(FOLDER):
    CLASS_PATH = os.path.join(FOLDER, CLASS)
    for train_test in os.listdir(CLASS_PATH):
        train_test_path = os.path.join(CLASS_PATH, train_test)
        for name in tqdm(glob.glob(os.path.join(train_test_path, "*.jpg"))):
            img_path = os.path.join(train_test_path, name)
            annot_path = os.path.join(train_test_path,
                                      name.replace("jpg", "pts"))
            landmark, visable = load_annot(annot_path)
            visable_landmark = landmark[visable == 1]
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
                "weight":weights[CLASS],
                'has_visable': 0
            }
            infos.append(info)

print("aflw dataset samples: ",len(infos))
np.save("./dataset/aflw_annot.npy", infos)
print("annnot saved: ./dataset/alfw_annot.npy")