import cv2
import torch
import numpy as np
import torch.nn.functional as F

from utils import constants
from utils.imutils import torch2numpy, rgb_processing
from faceboxes.facebox_dnn import facebox_dnn
from model.mobileone import mobileone, reparameterize_model

from torchvision.transforms import Normalize

normalize_img = Normalize(mean=constants.IMG_NORM_MEAN,
                          std=constants.IMG_NORM_STD)

device = "cpu"
net = mobileone(num_classes=68 * 2, inference_mode=False, variant="s1")
checkpoint = torch.load("checkpoint/snapshot/checkpoint_epoch_150.pth.tar",
                        map_location=device)
net.load_state_dict(checkpoint['net'])
net = reparameterize_model(net)
net.eval()


def main():
    facebox_detect = facebox_dnn()
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        ori_img=img.copy()

        if not ret: break
        height, width = img.shape[:2]
        bounding_boxes, _ = facebox_detect.get_face(img.copy())
        if len(bounding_boxes) == 0:
            continue
        # bounding_boxes, landmarks = detect_faces(img)
        box = bounding_boxes[0]
        x1, y1, x2, y2 = (box[:4] + 0.5).astype(np.int32)
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        cx = x1 + w // 2
        cy = y1 + h // 2
        scale = max(h, w) / constants.IMG_RES * 0.8

        img = rgb_processing(img, [cx, cy], scale, 0, np.ones(3))
        img = torch.from_numpy(img).float()
        img = normalize_img(img)

        pre_landmarks, pre_visables = net(img.unsqueeze(0))

        pre_visables = F.sigmoid(pre_visables)
        pre_landmark, pre_visable = pre_landmarks[0], pre_visables[0]

        np_img = torch2numpy(img)
        pre_landmark = (pre_landmark) / 2 * scale * constants.IMG_RES
        pre_landmark = pre_landmark.detach().numpy()+np.array([cx, cy])

        for k in range(len(pre_visable)):
            if pre_visable[k] > 0.3:
                cv2.circle(
                    ori_img,
                    [int(pre_landmark[k][0]),
                        int(pre_landmark[k][1])], 2, [255, 0, 0], -1)
            else:
                pass

        # cv2.imwrite('processed/{}.jpg'.format(0),ori_img)
        cv2.imshow('hah',ori_img)
        if cv2.waitKey(10) == 27:
            break


main()