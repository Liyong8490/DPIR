import os.path
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import data.util as util


class DenoiseDataset(data.Dataset):
    """
    Read original images and generate noisy image with
    gaussian white noise with given range.
    """
    def __init__(self, opt):
        super(DenoiseDataset, self).__init__()
        self.opt = opt
        self.label_root = None
        self.noisy_root = None
        # environment for lmdb
        if 'data_root' in opt
        self.env, self.label_root = util.get_image_paths(opt['data_type'], opt['data_root'])
        assert self.paths, "Error: Dataset path is empty."

        self.random_scale_list = [1]

    def __getitem__(self, index):
        scale = self.opt['scale']
        im_size = self.opt['im_size']
        path = self.paths[index]
        img = util.read_img(self.env, path)
        if self.opt['phase'] != 'train':
            img = util.modcrop(img, scale)
        if self.opt['color']:
            img = util.channel_convert(img.shape[2], self.opt['color'], [img])[0]

        if self.paths:


