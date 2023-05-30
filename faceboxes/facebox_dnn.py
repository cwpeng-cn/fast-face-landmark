import cv2
import torch
import time
import numpy as np


class facebox_dnn():
    def __init__(self, threshold=0.05):
        self.net = cv2.dnn.readNetFromCaffe('faceboxes/faceboxes_deploy.prototxt', 'faceboxes/faceboxes.caffemodel')
        self.conf_threshold = threshold

    def detect(self, frame):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, None, [104, 117, 123], False, False)
        self.net.setInput(blob)
        detections = self.net.forward()
        face_rois = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
                face_rois.append(frame[y1:y2, x1:x2])
        return frameOpencvDnn, face_rois

    def get_face(self, frame):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, None, [104, 117, 123], False, False)
        self.net.setInput(blob)
        detections = self.net.forward()
        boxs, face_rois = [], []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                boxs.append((x1, y1, x2, y2))
                face_rois.append(frame[y1:y2, x1:x2])
        return np.array(boxs), face_rois


if __name__ == "__main__":
    facebox_detect = facebox_dnn()
    cap = cv2.VideoCapture(0)
    ret,frame=cap.read()
    while ret:
        ret,frame=cap.read()
        start=time.time()
        drawimg, face_rois = facebox_detect.detect(frame)
        print((time.time()-start)*1000)
        cv2.imshow('detect', drawimg)
        cv2.waitKey(10)
    cv2.destroyAllWindows()
