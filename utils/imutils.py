# Copyright (c) Facebook, Inc. and its affiliates.

#Modified from https://github.com/nkolot/SPIN/blob/master/LICENSE
"""
This file contains functions that are used to perform data augmentation.
"""
import torch
import numpy as np
import scipy.misc
import cv2
from utils import constants
from scipy import ndimage

from torchvision.transforms import Normalize


def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = constants.IMG_RES * scale  #h becomes the original bbox max(height, min).
    t = np.zeros((3, 3))
    t[0, 0] = float(
        res[1]
    ) / h  #This becomes a scaling factor to rescale original bbox -> res size (default: 112x112)
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1


def crop(img, center, scale, res, rot=0):
    """Crop image according to the supplied bounding box."""
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1)) - 1
    # Bottom right point
    br = np.array(
        transform([res[0] + 1, res[1] + 1], center, scale, res, invert=1)) - 1

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if new_shape[0] > 15000 or new_shape[1] > 15000:
        print("Image Size Too Big!  scale{}, new_shape{} br{}, ul{}".format(
            scale, new_shape, br, ul))
        return None

    if len(img.shape) > 2:
        new_shape += [img.shape[2]]

    new_img = np.zeros(new_shape, dtype=np.uint8)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    # print("{} vs {}  || {} vs {}".format(new_y[1] - new_y[0] , old_y[1] - old_y[0], new_x[1] - new_x[0], old_x[1] -old_x[0] )   )
    if new_y[1] - new_y[0] != old_y[1] - old_y[0] or new_x[1] - new_x[
            0] != old_x[1] - old_x[0] or new_y[1] - new_y[0] < 0 or new_x[
                1] - new_x[0] < 0:
        print("Warning: maybe person is out of image boundary!")
        return None
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1],
                                                        old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        new_img = scipy.ndimage.interpolation.rotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]

    new_img = cv2.resize(new_img, tuple(res))

    return new_img


def uncrop(img, center, scale, orig_shape, rot=0, is_rgb=True):
    """'Undo' the image cropping/resizing.
    This function is used when evaluating mask/part segmentation.
    """
    res = img.shape[:2]
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1)) - 1
    # Bottom right point
    br = np.array(
        transform([res[0] + 1, res[1] + 1], center, scale, res, invert=1)) - 1
    # size of cropped image
    crop_shape = [br[1] - ul[1], br[0] - ul[0]]

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(orig_shape, dtype=np.uint8)
    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], orig_shape[1]) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], orig_shape[0]) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(orig_shape[1], br[0])
    old_y = max(0, ul[1]), min(orig_shape[0], br[1])
    img = cv2.resize(img, crop_shape, interp='nearest')
    new_img[old_y[0]:old_y[1], old_x[0]:old_x[1]] = img[new_y[0]:new_y[1],
                                                        new_x[0]:new_x[1]]
    return new_img


def flip_img(img):
    """Flip rgb images or masks.
    channels come last, e.g. (256,256,3).
    """
    img = np.fliplr(img)
    return img


def torch2numpy(tensor_img):
    assert len(tensor_img.shape) == 3
    np_img = tensor_img.numpy()
    np_img = np.transpose(np_img, (1, 2, 0))
    np_img = np_img * constants.IMG_NORM_STD + constants.IMG_NORM_MEAN
    np_img = (np_img * 255).astype(np.uint8)
    return np_img.copy()

def rgb_processing(rgb_img, center, scale, rot, pn):
    """Process rgb image and do augmentation."""
    rgb_img = crop(rgb_img,
                    center,
                    scale, [constants.IMG_RES, constants.IMG_RES],
                    rot=rot)

    if rgb_img is None:
        return None

    # in the rgb image we add pixel noise in a channel-wise manner
    rgb_img[:, :,
            0] = np.minimum(255.0, np.maximum(0.0,
                                                rgb_img[:, :, 0] * pn[0]))
    rgb_img[:, :,
            1] = np.minimum(255.0, np.maximum(0.0,
                                                rgb_img[:, :, 1] * pn[1]))
    rgb_img[:, :,
            2] = np.minimum(255.0, np.maximum(0.0,
                                                rgb_img[:, :, 2] * pn[2]))

    rgb_img = np.transpose(rgb_img.astype('float32'), (2, 0, 1)) / 255.0
    return rgb_img
