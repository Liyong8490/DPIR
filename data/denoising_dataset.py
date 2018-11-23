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
        # environment for lmdb
        self.env_lb, self.label_root = util.get_image_paths(opt['data_type'], opt['label_root'])
        self.env_ni, self.noisy_root = util.get_image_paths(opt['data_type'], opt['noisy_root'])

        assert self.label_root, "Error: Dataset path is empty."

        if self.label_root and self.noisy_root:
            assert len(self.label_root) == len(self.noisy_root), \
                "Label and noisy dataset have different number of images: {}, {}.".format(
                    len(self.label_root), len(self.noisy_root))

        self.random_scale_list = [1]

    def __getitem__(self, index):
        noisy_path_ = None
        nlv = self.opt['noise_level']
        im_size = self.opt['im_size']
        label_path_ = self.label_root[index]
        img_label = util.read_img(self.env_lb, label_path_)
        # force to 3 channels
        if img_label.ndim == 2:
            img_label = cv2.cvtColor(img_label, cv2.COLOR_GRAY2BGR)
        if self.opt['color']:
            img_label = util.channel_convert(img_label.shape[2], self.opt['color'], [img_label])[0]

        # get noisy image
        if self.noisy_root:
            noisy_path_ = self.noisy_root[index]
            img_noisy = util.read_img(self.env_ni, noisy_path_)
            if img_noisy.ndim == 2:
                img_noisy = cv2.cvtColor(img_noisy, cv2.COLOR_GRAY2BGR)
        else:  # add noise on-the-fly
            # random sample noise level from given range
            if self.opt['phase'] == 'train':
                random_scale = random.choice(self.random_scale_list)
                H_s, W_s, C_s = img_label.shape

                def _mod(n, random_scale_, thres):
                    rlt = int(n * random_scale_)
                    return thres if rlt < thres else rlt

                H_s = _mod(H_s, random_scale, im_size)
                W_s = _mod(W_s, random_scale, im_size)
                img_label = cv2.resize(np.copy(img_label), (W_s, H_s), interpolation=cv2.INTER_LINEAR)
            if img_label.ndim == 2:
                img_label = np.expand_dims(img_label, axis=2)
            # add random noise to generate img_noisy
            sigma = nlv[random.randrange(len(nlv))]
            noise = sigma / 255.0 * np.random.normal(size=img_label.shape)
            img_noisy = (img_label + noise) * 255.0
            img_noisy = img_noisy.round()
            img_noisy[img_noisy > 255] = 255
            img_noisy[img_noisy < 0] = 0
            img_noisy /= 255.0

        if self.opt['phase'] == 'train':
            shape = img_label.shape
            H = shape[0]
            W = shape[1]
            # randomly crop
            rnd_h = random.randint(0, max(0, H - im_size))
            rnd_w = random.randint(0, max(0, W - im_size))
            img_noisy = img_noisy[rnd_h:rnd_h + im_size, rnd_w:rnd_w + im_size, :]
            img_label = img_label[rnd_h:rnd_h + im_size, rnd_w:rnd_w + im_size, :]

            # augmentation - flip, rotate
            img_noisy, img_label = util.augment([img_noisy, img_label], self.opt['use_flip'], self.opt['use_rot'])

            # change color space if necessary
        if self.opt['color']:
            _, _, C = img_noisy.shape
            img_noisy = util.channel_convert(C, self.opt['color'], [img_noisy])[0]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_label.shape[2] == 3:
            img_label = img_label[:, :, [2, 1, 0]]
            img_noisy = img_noisy[:, :, [2, 1, 0]]
        img_label = torch.from_numpy(np.ascontiguousarray(np.transpose(img_label, (2, 0, 1)))).float()
        img_noisy = torch.from_numpy(np.ascontiguousarray(np.transpose(img_noisy, (2, 0, 1)))).float()

        if noisy_path_ is None:
            noisy_path_ = label_path_
        return {'NI': img_noisy, 'LB': img_label, 'NI_path': noisy_path_, 'LB_path': label_path_}

    def __len__(self):
        return len(self.label_root)
