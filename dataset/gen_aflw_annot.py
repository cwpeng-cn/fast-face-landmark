# -*- encoding: utf-8 -*-
'''
Description      : used to generate data,
                   dataset download: https://github.com/abhi1kumar/MERL-RAV_dataset
Time             :2023/05/24 13:24:27
Author           :cwpeng
email            :cw.peng@foxmail.com
'''

import os
import cv2
import time
import glob
import lmdb
import pickle
import numpy as np
from tqdm import tqdm


FOLDER = "/home/cwpeng/projects/data/merl_rav_organized/"
SAVE_PATH = "./dataset/aflw.lmdb"
COMPRESS_LEVEL = 1
WEIGHTS = {
    "frontal": 1.0,
    "right": 1.0,
    "left": 1.0,
    "lefthalf": 1.5,
    "righthalf": 1.5
}
meta_infos = {}


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


def estimate_mapsize():
    map_size = 0
    for CLASS in os.listdir(FOLDER):
        CLASS_PATH = os.path.join(FOLDER, CLASS)
        for train_test in os.listdir(CLASS_PATH):
            train_test_path = os.path.join(CLASS_PATH, train_test)
            img_path_list = glob.glob(os.path.join(train_test_path, "*.jpg"))
            if len(img_path_list):
                img_path = os.path.join(train_test_path,img_path_list[0])  #select a image
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                _, img_byte = cv2.imencode(
                    '.jpg', img, [cv2.IMWRITE_PNG_COMPRESSION, COMPRESS_LEVEL])
                data_size_per_img = img_byte.nbytes
                map_size += data_size_per_img * len(img_path_list)
    return map_size * 10


if __name__ == "__main__":
    mapsize=estimate_mapsize()
    env=lmdb.open(SAVE_PATH,map_size=mapsize)
    txn = env.begin(write=True)
    count=0
    for CLASS in os.listdir(FOLDER):
        CLASS_PATH = os.path.join(FOLDER, CLASS)
        for train_test in os.listdir(CLASS_PATH):
            train_test_path = os.path.join(CLASS_PATH, train_test)
            for name in tqdm(glob.glob(os.path.join(train_test_path,
                                                    "*.jpg"))):
                img_path = os.path.join(train_test_path, name)
                annot_path = os.path.join(train_test_path,
                                          name.replace("jpg", "pts"))
                landmark, visable = load_annot(annot_path)
                visable_landmark = landmark[visable == 1]
                center_x = int(visable_landmark[:, 0].mean())
                center_y = int(visable_landmark[:, 1].mean())
                w = visable_landmark[:, 0].max() - visable_landmark[:, 0].min()
                h = visable_landmark[:, 1].max() - visable_landmark[:, 1].min()

                random_suffix = str(time.time()).split(".")[-1]
                key = os.path.basename(img_path).split(".")[0] + random_suffix

                info = {
                    "img_path": img_path,
                    "visable": visable,
                    "landmark": landmark,
                    "center_x": center_x,
                    "center_y": center_y,
                    "w": w,
                    "h": h,
                    "weight": WEIGHTS[CLASS],
                    'has_visable': 0
                }

                meta_infos[key] = info
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                _,img_byte=cv2.imencode('.jpg', img, [cv2.IMWRITE_PNG_COMPRESSION, COMPRESS_LEVEL])
                txn.put(key.encode('ascii'), img_byte)
                count+=1
                if count % 100 == 0:
                    txn.commit()
                    txn = env.begin(write=True)
    
    txn.commit()
    env.close()

    with open(os.path.join(SAVE_PATH, "meta_info.pkl"), 'wb') as f:
        pickle.dump(meta_infos, f)