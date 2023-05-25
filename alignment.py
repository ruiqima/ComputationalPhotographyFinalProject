import numpy as np
from skimage import io, color
import cv2


'''
1. We can find the brightess pixel: the north star, i.e. the center of rotation
2. Find two corresponding point and calculate rotation matrix. How?

'''

def overlap_images(imgs):
    print(imgs.shape)
    res = np.sum(imgs, axis = 0)
    print(res.shape)
    # the brightess pixel is the north star
    gray = color.rgb2gray(res)
    flat_gray = gray.flatten()
    brightest_index = np.argmax(flat_gray)
    brightest_coords = np.unravel_index(brightest_index, gray.shape)
    print(brightest_coords)
    res[brightest_coords[0], brightest_coords[1], :] = [1, 0, 0]
    io.imsave('overlap.png', res)


imgs = np.load("data/tiffs.npy")
print(imgs.shape)
