import os
import math
from datetime import datetime
import numpy as np
import cv2 as cv
from skimage.measure import compare_psnr, compare_ssim
from torchvision.utils import make_grid


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print("Path already exists. Rename exist path to [{:s}]".format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def tensor2img(tensor, out_type=np.uint8, min_max=(0,1)):
    '''
    Convert a Torch Tensor into an image Numpy array
    :param tensor: 4D(B, C, H, W), 3D(C, H, W) or 2D(H, W), any range, RGB channel order
    :param out_type: np.uint8 (default)
    :param min_max: output Tensor value range.
    :return: 3D(H, W, C) or 2D(H, W) range from 0 to 255 with type of np.uint8
    '''
    tensor = tensor.squeeze().float.cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1]-min_max[0])
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0)) # HWC BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0)) # HWC BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            "Only support 4D, 3D and 2D tensor. But received with dimension: {:d}".format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # ** Important. Numpy.uint8() WILL NOT round by default
    return img_np.astype(out_type)


def save_img(img, img_path):
    cv.imwrite(img_path, img)


def psnr(im1, im2):
    # mse = np.power(im1 - im2, 2).mean()
    # psnr = 20 * np.log10(255.0) - 10 * np.log10(mse)
    im1 = np.maximum(np.minimum(im1, 1.0), 0.0)
    im2 = np.maximum(np.minimum(im2, 1.0), 0.0)
    return compare_psnr(im1, im2)


def ssim(im1, im2):
    im1 = np.maximum(np.minimum(im1, 1.0), 0.0)
    im2 = np.maximum(np.minimum(im2, 1.0), 0.0)
    return compare_ssim(im1, im2, win_size=11, data_range=1, gaussian_weights=True)
