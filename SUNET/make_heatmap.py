import json

import numpy as np
import torch


def read_keypoints(path):
    f = open(path)
    t = json.load(f)
    return t



def make_heatmap(path, size, sigma=10):
    sqr_sig = sigma ** 2
    kernel = np.zeros([9 * sigma, 9 * sigma])
    for i in range(9 * sigma):
        for j in range(9 * sigma):
            kernel[i, j] = 1 / (2 * np.pi * sqr_sig) * \
                           np.exp(
                               -((i - (9 * sigma - 1) / 2) ** 2 + (j - (9 * sigma - 1) / 2) ** 2) / (
                                       2 * sqr_sig))
    kernel = torch.tensor(kernel) / (1 / (2 * np.pi * sqr_sig))
    keypoints = read_keypoints(path)['people'][0]['pose_keypoints_2d']
    keypoints = torch.tensor(np.array(keypoints)).view(-1, 3)
    heatmap = torch.zeros(25, size[0] + 18 * sigma, size[1] + 18 * sigma)
    for i, keypoint in enumerate(keypoints):
        if keypoint[2] > 0:
            heatmap[i, 9 * sigma + int(torch.round(keypoint[1])) - int(
                round(9 * sigma / 2)):9 * sigma + int(
                torch.round(keypoint[1])) - int(
                round(9 * sigma / 2)) + 9 * sigma,
            9 * sigma + int(torch.round(keypoint[0])) - int(
                round(9 * sigma / 2)):9 * sigma + int(
                torch.round(keypoint[0])) - int(
                round(9 * sigma / 2)) + 9 * sigma] = kernel
    heatmap = heatmap[:, 9 * sigma:-9 * sigma, 9 * sigma:-9 * sigma]
    return heatmap
