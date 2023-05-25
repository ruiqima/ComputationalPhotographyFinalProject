import numpy as np
import cv2
from scipy.interpolate import interpn
from skimage import io
import math
from scipy import signal

def g(x, sigma):
    return np.exp(- (x ** 2) / (2 * (sigma ** 2))) / (2 * np.pi * (sigma ** 2))

def get_difference(I1, I2):
    d = I1 - I2
    Id = (d - np.min(d)) / (np.max(d) - np.min(d))
    print('# of 0 in Id: {}'.format(np.count_nonzero(Id == 0)))
    io.imsave('difference.png', Id)


def bilateral_filtering(I, ksize, sigma_s, sigma_r):
    # padding the edges
    half_ksize = (ksize - 1) // 2
    I_padded = np.pad(I, ((half_ksize, half_ksize), (half_ksize, half_ksize), (0, 0)), 'edge')

    # calculate G_sigma_s
    points = np.array(np.meshgrid(
                np.arange(- half_ksize, half_ksize + 1), 
                np.arange(- half_ksize, half_ksize + 1))
            ).T.reshape(-1, 2).reshape(ksize, ksize, 2)
    l2_norm = np.linalg.norm(points - np.array([0, 0]), axis = 2)
    print('l2_norm: ')
    # print(l2_norm)
    G_s = g(l2_norm, sigma_s)
    print('G_s: ')
    # print(G_s)

    def compute_filtered_value(p_i, p_j, k):
        '''
        p_i and p_j are on the padded image
        k is the current color channel
        '''
        print('----------start computing ({}, {}, {})----------'.format(p_i, p_j, k))
        if p_i < half_ksize or p_i + half_ksize + 1 > I_padded.shape[0]:
            return 0
        if p_j < half_ksize or p_j + half_ksize + 1 > I_padded.shape[1]:
            return 0

        # qs is the coordinates on the original unpadded I
        qs = np.array(
            np.meshgrid(
                np.arange(p_i - half_ksize, p_i + half_ksize + 1), 
                np.arange(p_j - half_ksize, p_j + half_ksize + 1))
            ).T.reshape(-1, 2).reshape(ksize, ksize, 2)

        # calculate G_sigma_r
        range_difference = I_padded[qs[:, :, 0], qs[:, :, 1], k] - I_padded[p_i, p_j, k]
        # print('range_difference: ')
        # print(range_difference)
        G_r = g(range_difference, sigma_r)
        # print('G_r: ')
        # print(G_r)

        # calculate joint G (w)
        w = G_s * G_r

        # apply gaussian filter to point p_i, p_j and get updated value for position p
        filtered_value_at_p = np.sum(w * I_padded[qs[:, :, 0], qs[:, :, 1], k]) / np.sum(w)
        # print('filtered_value_at_p: {}'.format(filtered_value_at_p))
        # print('----------finish computing ({}, {}, {})----------'.format(p_i, p_j, k))
        return filtered_value_at_p

    

    vectorized_compute_filtered_value = np.vectorize(compute_filtered_value)
    I_filtered = vectorized_compute_filtered_value(
        np.arange(I_padded.shape[0])[:, np.newaxis, np.newaxis],
        np.arange(I_padded.shape[1])[np.newaxis, :, np.newaxis],
        np.arange(I_padded.shape[2])[np.newaxis, np.newaxis, :])

    crop_indices = (slice(half_ksize, -half_ksize), slice(half_ksize, -half_ksize), slice(None))
    I_filtered = I_filtered[crop_indices]
    return I_filtered
    


KERNEL = 7
SIGMA_S = 1
SIGMA_R = 1

# I = np.load("data/one_tiff.npy")
I = io.imread("test_data/small_2.jpg")
print(I.shape)
# io.imsave('I.png', I)
# print('# of 0 in I: {}'.format(np.count_nonzero(I == 0)))
bilateral_filtered = bilateral_filtering(I, KERNEL, SIGMA_S, SIGMA_R)

print('# of 0 in bilateral_filtered: {}'.format(np.count_nonzero(bilateral_filtered == 0)))
io.imsave('bilateral_filtered.png', bilateral_filtered)
get_difference(bilateral_filtered, I)
