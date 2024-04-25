import cv2
import torch
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

from os.path import join

from dkm.models.model_zoo.DKMv3 import DKMv3

from demo import preprocess, read_image, fast_make_matching_figure, fast_make_matching_overlay, compute_geom, \
    warp_images


class DiffExtractor:
    def __init__(self, checkpoints_path='weights/gim_dkm_100h.ckpt', device='cpu'):
        self.dense_certainty = None
        self.dense_matches = None
        self.device = device
        model = DKMv3(weights=None, h=672, w=896)
        state_dict = torch.load(checkpoints_path, map_location='cpu')
        if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('model.'):
                state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)
            if 'encoder.net.fc' in k:
                state_dict.pop(k)
        model.load_state_dict(state_dict)
        model = model.eval().to(device)
        self.model = model
        self.follow_up = False

    def extract(self, path0, path1, threshold=0.1, size_ratio=1.0, mask_path=None, follow_up=False) -> np.ndarray:
        self.follow_up = follow_up
        image0 = read_image(path0)
        image1 = read_image(path1)
        image0 = cv2.resize(image0, (int(image0.shape[1] * size_ratio), int(image0.shape[0] * size_ratio)))
        image1 = cv2.resize(image1, (int(image1.shape[1] * size_ratio), int(image1.shape[0] * size_ratio)))
        image0, scale0 = preprocess(image0)
        image1, scale1 = preprocess(image1)

        image0 = image0.to(self.device)[None]
        image1 = image1.to(self.device)[None]

        self.data = dict(color0=image0, color1=image1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.dense_matches, self.dense_certainty = self.model.match(image0, image1)

        height0, width0 = image0.shape[-2:]
        points = self.dense_matches.reshape(-1, 4)[:, :2]
        certainty = self.dense_certainty.reshape(-1)
        points = torch.stack((width0 * (points[:, 0] + 1) / 2, height0 * (points[:, 1] + 1) / 2), dim=-1)
        points = torch.round(points).int()
        points = torch.clip(points[:, 0] + points[:, 1] * width0, 0, height0 * width0 - 1)
        points = points[certainty > threshold].long()
        img = torch.ones(height0 * width0, dtype=torch.uint8, device=self.device) * 255
        img[points] = 0
        img = img.reshape(height0, width0)

        if mask_path:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (width0, height0))
            mask = torch.tensor(mask, device=self.device)
            img = torch.min(img, mask)

        if follow_up:
            height0, width0 = self.data['color0'].shape[-2:]
            height1, width1 = self.data['color1'].shape[-2:]
            sparse_matches, mconf = self.model.sample(self.dense_matches, self.dense_certainty, 5000)
            kpts0 = sparse_matches[:, :2]
            kpts0 = torch.stack((
                width0 * (kpts0[:, 0] + 1) / 2, height0 * (kpts0[:, 1] + 1) / 2), dim=-1, )
            kpts1 = sparse_matches[:, 2:]
            kpts1 = torch.stack((
                width1 * (kpts1[:, 0] + 1) / 2, height1 * (kpts1[:, 1] + 1) / 2), dim=-1, )
            b_ids = torch.where(mconf[None])[0]

            # robust fitting
            _, mask = cv2.findFundamentalMat(kpts0.cpu().detach().numpy(),
                                             kpts1.cpu().detach().numpy(),
                                             cv2.USAC_MAGSAC, ransacReprojThreshold=1.0,
                                             confidence=0.999999, maxIters=10000)
            mask = mask.ravel() > 0

            self.data.update({
                'hw0_i': self.data['color0'].shape[-2:],
                'hw1_i': self.data['color1'].shape[-2:],
                'mkpts0_f': kpts0,
                'mkpts1_f': kpts1,
                'm_bids': b_ids,
                'mconf': mconf,
                'inliers': mask,
            })

        return img.cpu().numpy()

    def warp(self):
        if not self.follow_up:
            raise ValueError('Please set follow_up=True in the extract method')
        geom_info = compute_geom(self.data)
        warped_image = warp_images(self.data['color0'], self.data['color1'], geom_info,
                                   "Homography")

        return warped_image

    def match(self):
        if not self.follow_up:
            raise ValueError('Please set follow_up=True in the extract method')
        alpha = 0.5
        out = fast_make_matching_figure(self.data, b_id=0)
        overlay = fast_make_matching_overlay(self.data, b_id=0)
        out = cv2.addWeighted(out, 1 - alpha, overlay, alpha, 0)[..., ::-1]
        return out


if __name__ == '__main__':
    import os

    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    output = 'output_tmp'
    os.makedirs(output, exist_ok=True)
    path = 'data'
    path_a = join(path, 'output_A')
    path_b = join(path, 'output_B')
    path_mask = join(path, 'mask')
    name_a = os.listdir(path_a)

    extractor = DiffExtractor(device=device)
    for i in range(len(name_a)):
        img_a = join(path_a, name_a[i])
        img_b = join(path_b, name_a[i])
        mask = join(path_mask, name_a[i])
        diff = extractor.extract(img_a, img_b, threshold=0.1, mask_path=mask)
        cv2.imwrite(join(output, name_a[i]), diff)
        print(f'{name_a[i]} done')
